"""GenAI spans emitted by the shared HTTP model transport."""

import json

import httpx2
import msgspec
import pytest
from msgtrace.sdk.tracer import tracer_manager
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind, StatusCode

from msgflux.exceptions import ModelProviderHTTPError
from msgflux.envs import envs
from msgflux.models.http_transport import HTTPTransport
from msgflux.models.model_credentials import (
    ModelCredentialResolver,
    ResolvedModelCredentials,
)
from msgflux.models.providers.vllm import VLLMTextClassifier
from msgflux.models.providers.ollama import OllamaChatCompletion
from msgflux.nn.modules.module import Module
from msgflux.telemetry import Spans
from msgflux.telemetry.context import active_span


class _Credentials(ModelCredentialResolver):
    def resolve(self, owner):
        return ResolvedModelCredentials(headers={"Authorization": "secret-token"})


class _Owner:
    provider = "openai"
    model_id = "test-model"
    credential_resolver = _Credentials()
    sampling_params = {"base_url": "https://api.example.com/v1"}

    def _raise_if_aborted(self):
        pass


@pytest.fixture
def spans(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracer_manager, "_tracer", provider.get_tracer("test"))
    yield exporter
    provider.shutdown()


def test_request_records_genai_attributes_and_parent(spans):
    attempts = 0

    def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx2.Response(429, headers={"retry-after": "0"})
        return httpx2.Response(
            200,
            json={
                "id": "resp_123",
                "model": "served-model",
                "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "Hello!"}}
                ],
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        transport = HTTPTransport(client=client, max_retries=1)
        with active_span(Spans.init_flow("test-flow")):
            transport.request(
                _Owner(),
                "/chat/completions",
                json={"model": "requested-model", "temperature": 0.5},
            )

    parent, model = spans.get_finished_spans()[0:2]
    assert model.name == "test-flow"
    assert parent.name == "chat requested-model"
    assert parent.kind is SpanKind.CLIENT
    assert parent.parent.span_id == model.context.span_id
    assert attempts == 2
    assert parent.attributes["gen_ai.operation.name"] == "chat"
    assert parent.attributes["gen_ai.provider.name"] == "openai"
    assert parent.attributes["gen_ai.request.model"] == "requested-model"
    assert parent.attributes["gen_ai.request.temperature"] == 0.5
    assert parent.attributes["gen_ai.response.id"] == "resp_123"
    assert parent.attributes["gen_ai.response.model"] == "served-model"
    assert parent.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert parent.attributes["gen_ai.usage.input_tokens"] == 5
    assert parent.attributes["gen_ai.usage.output_tokens"] == 2
    assert json.loads(parent.attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "Hello!"}]}
    ]
    assert "secret-token" not in repr(parent.attributes)


@pytest.mark.parametrize(
    ("endpoint", "params", "level"),
    [
        ("/chat/completions", {"reasoning_effort": "high"}, "high"),
        ("/responses", {"reasoning": {"effort": "low"}}, "low"),
        ("/chat/completions", {"reasoning": {"effort": "medium"}}, "medium"),
        ("/api/chat", {"think": "high"}, None),
    ],
)
def test_request_records_only_sent_reasoning_level(spans, endpoint, params, level):
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json={}))
    ) as client:
        HTTPTransport(client=client).request(
            _Owner(), endpoint, json={"model": "test-model", **params}
        )

    (span,) = spans.get_finished_spans()
    assert span.attributes.get("gen_ai.request.reasoning.level") == level


def test_gemini_transport_records_nested_thinking_level_without_schema(spans):
    class GeminiOwner(_Owner):
        provider = "gemini"

    response_payload = {
        "id": "gemini-response",
        "model": "gemini-served-model",
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        "choices": [
            {"finish_reason": "stop", "message": {"content": "structured output"}}
        ],
    }
    with httpx2.Client(
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(200, json=response_payload)
        )
    ) as client:
        HTTPTransport(client=client).request(
            GeminiOwner(),
            "/chat/completions",
            json={
                "model": "gemini-requested-model",
                "extra_body": {
                    "google": {"thinking_config": {"thinking_level": "low"}}
                },
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "private-schema",
                        "schema": {"type": "object"},
                    },
                },
            },
        )

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.request.reasoning.level"] == "low"
    assert span.attributes["gen_ai.request.model"] == "gemini-requested-model"
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 3
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "structured output"}],
        }
    ]
    assert "private-schema" not in repr(span.attributes)


def test_gemini_provider_records_translated_effort_over_http_transport(
    spans, monkeypatch
):
    from msgflux.models.chat_transport import HTTPChatTransport
    from msgflux.models.providers.gemini import GeminiChatCompletion

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    observed = {}

    def handler(request):
        observed["body"] = json.loads(request.content)
        return httpx2.Response(
            200,
            json={
                "id": "gemini-response",
                "model": "gemini-served-model",
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "OK"},
                    }
                ],
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model = GeminiChatCompletion(
            model_id="gemini-3.6-flash",
            reasoning_effort="low",
            chat_transport=HTTPChatTransport(client=client),
        )
        response = model("Reply with exactly OK")
        assert response.consume() == "OK"

    assert observed["body"]["extra_body"]["google"]["thinking_config"] == {
        "thinking_level": "low",
        "include_thoughts": True,
    }
    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.request.reasoning.level"] == "low"
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 3


def test_non_gemini_transport_ignores_gemini_thinking_config(spans):
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json={}))
    ) as client:
        HTTPTransport(client=client).request(
            _Owner(),
            "/chat/completions",
            json={
                "model": "test-model",
                "extra_body": {
                    "google": {"thinking_config": {"thinking_level": "high"}}
                },
            },
        )

    (span,) = spans.get_finished_spans()
    assert "gen_ai.request.reasoning.level" not in span.attributes


@pytest.mark.parametrize(
    ("think", "level", "enabled"),
    [("high", "high", None), (True, None, True), (False, None, False)],
)
def test_ollama_native_think_preserves_level_or_boolean(spans, think, level, enabled):
    class OllamaOwner(_Owner):
        provider = "ollama"

    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json={}))
    ) as client:
        HTTPTransport(client=client).request(
            OllamaOwner(), "/api/chat", json={"model": "test-model", "think": think}
        )

    (span,) = spans.get_finished_spans()
    assert span.attributes.get("gen_ai.request.reasoning.level") == level
    assert span.attributes.get("ollama.request.think") == enabled


def test_stream_span_ends_when_closed_and_records_final_usage(spans):
    body = b'data: {"choices":[],"usage":{"input_tokens":3,"output_tokens":4}}\n\n'

    def handler(_request):
        return httpx2.Response(200, content=body)

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        transport = HTTPTransport(client=client)
        stream = transport.stream(
            _Owner(),
            "/responses",
            json={"model": "test-model", "stream": True},
            iterate=lambda response: (
                {"usage": {"input_tokens": 3, "output_tokens": 4}}
                for _ in response.iter_lines()
            ),
        )
        assert next(stream)["usage"]["output_tokens"] == 4
        assert not spans.get_finished_spans()
        stream.close()

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.usage.input_tokens"] == 3
    assert span.attributes["gen_ai.usage.output_tokens"] == 4


@pytest.mark.asyncio
async def test_async_request_records_error_status(spans):
    async def handler(_request):
        return httpx2.Response(400, json={"error": {"message": "invalid model"}})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        transport = HTTPTransport(async_client=client, max_retries=0)
        with pytest.raises(ModelProviderHTTPError):
            await transport.arequest(
                _Owner(), "/embeddings", json={"model": "test-model"}
            )

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.operation.name"] == "embeddings"
    assert span.status.status_code is StatusCode.ERROR
    assert len(span.events) == 1


@pytest.mark.asyncio
async def test_async_stream_records_usage_and_closes_on_early_exit(spans):
    closed = []

    class Body(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield b"one"
            yield b"two"

        async def aclose(self):
            closed.append(True)

    async def handler(_request):
        return httpx2.Response(200, stream=Body())

    async def iterate(response):
        async for _ in response.aiter_bytes():
            yield {"usage": {"prompt_tokens": 7, "completion_tokens": 2}}

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        transport = HTTPTransport(async_client=client)
        stream = transport.astream(
            _Owner(), "/chat/completions", json={"model": "test-model"}, iterate=iterate
        )
        assert (await anext(stream))["usage"]["prompt_tokens"] == 7
        assert not spans.get_finished_spans()
        await stream.aclose()

    assert closed == [True]
    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 2


def test_chat_stream_assembles_output_and_finish_reasons(spans):
    frames = [
        {"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, content=b"ok"))
    ) as client:
        transport = HTTPTransport(client=client)
        assert (
            list(
                transport.stream(
                    _Owner(),
                    "/chat/completions",
                    iterate=lambda _: iter(frames),
                )
            )
            == frames
        )

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "Hello"}]}
    ]


def test_structured_response_preserves_json_text(spans):
    payload = {
        "choices": [
            {
                "message": {"content": '{"answer":"Paris","count":1}'},
                "finish_reason": "stop",
            }
        ]
    }
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=payload))
    ) as client:
        HTTPTransport(client=client).request(_Owner(), "/chat/completions")

    (span,) = spans.get_finished_spans()
    text = json.loads(span.attributes["gen_ai.output.messages"])[0]["parts"][0][
        "content"
    ]
    assert json.loads(text) == {"answer": "Paris", "count": 1}


def test_tool_call_is_recorded_as_genai_output_part(spans):
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location":"Paris"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=payload))
    ) as client:
        HTTPTransport(client=client).request(_Owner(), "/chat/completions")

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.response.finish_reasons"] == ("tool_calls",)
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "tool_call",
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": {"location": "Paris"},
                }
            ],
        }
    ]


def test_streamed_tool_call_assembles_arguments(spans):
    frames = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":',
                                },
                            }
                        ]
                    },
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '"Paris"}'}}
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, content=b"ok"))
    ) as client:
        list(
            HTTPTransport(client=client).stream(
                _Owner(), "/chat/completions", iterate=lambda _: iter(frames)
            )
        )

    (span,) = spans.get_finished_spans()
    part = json.loads(span.attributes["gen_ai.output.messages"])[0]["parts"][0]
    assert part["name"] == "get_weather"
    assert part["arguments"] == {"location": "Paris"}


def test_responses_function_call_is_recorded(spans):
    payload = {
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "get_weather",
                "arguments": '{"location":"Paris"}',
            }
        ],
    }
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=payload))
    ) as client:
        HTTPTransport(client=client).request(_Owner(), "/responses")

    (span,) = spans.get_finished_spans()
    part = json.loads(span.attributes["gen_ai.output.messages"])[0]["parts"][0]
    assert part == {
        "type": "tool_call",
        "id": "call_2",
        "name": "get_weather",
        "arguments": {"location": "Paris"},
    }


@pytest.mark.asyncio
async def test_responses_stream_uses_final_output_and_status(spans):
    frames = [
        {"type": "response.output_text.delta", "output_index": 0, "delta": "Part"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Part done"}],
                    }
                ],
            },
        },
    ]

    async def iterate(_):
        for frame in frames:
            yield frame

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, content=b"ok"))
    ) as client:
        transport = HTTPTransport(async_client=client)
        assert [
            frame
            async for frame in transport.astream(
                _Owner(), "/responses", iterate=iterate
            )
        ] == frames

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.response.status"] == "completed"
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "Part done"}]}
    ]


def test_model_output_capture_can_be_disabled(spans, monkeypatch):
    monkeypatch.setattr(envs, "telemetry_capture_model_output", False)
    with httpx2.Client(
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "secret"}, "finish_reason": "length"}
                    ]
                },
            )
        )
    ) as client:
        HTTPTransport(client=client).request(_Owner(), "/chat/completions")

    (span,) = spans.get_finished_spans()
    assert "gen_ai.output.messages" not in span.attributes
    assert span.attributes["gen_ai.response.finish_reasons"] == ("length",)


@pytest.mark.parametrize(
    ("status", "details", "reason", "span_status"),
    [
        ("incomplete", {"reason": "max_output_tokens"}, "length", StatusCode.UNSET),
        ("failed", None, "error", StatusCode.ERROR),
    ],
)
def test_responses_terminal_status_sets_stop_reason(
    spans, status, details, reason, span_status
):
    payload = {
        "status": status,
        "incomplete_details": details,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Partial"}],
            }
        ],
    }
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=payload))
    ) as client:
        HTTPTransport(client=client).request(_Owner(), "/responses")

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.response.finish_reasons"] == (reason,)
    assert span.attributes["gen_ai.response.status"] == status
    assert span.status.status_code is span_status


def test_module_span_is_parent_of_model_span(spans):
    class ChatModule(Module):
        def __init__(self, transport):
            super().__init__()
            self.transport = transport

        def forward(self):
            return self.transport.request(
                _Owner(), "/chat/completions", json={"model": "test-model"}
            )

    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json={}))
    ) as client:
        ChatModule(HTTPTransport(client=client))()

    model, flow = spans.get_finished_spans()
    assert model.kind is SpanKind.CLIENT
    assert model.parent.span_id == flow.context.span_id


def test_legacy_httpx_model_records_span(spans):
    with httpx2.Client(
        transport=httpx2.MockTransport(
            lambda _: httpx2.Response(200, json={"data": [{"label": "yes"}]})
        )
    ) as client:
        model = VLLMTextClassifier("classifier", retry=False)
        model.client.close()
        model.client = client
        result = model("example")

    assert result.data == ["yes"]
    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.provider.name"] == "vllm"
    assert span.attributes["gen_ai.request.model"] == "classifier"


def test_ollama_native_schema_records_output_and_usage(spans, monkeypatch):
    class Answer(msgspec.Struct):
        answer: str
        count: int

    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = OllamaChatCompletion(model_id="qwen3:8b", retry=False)
    payload = {
        "model": "qwen3:8b",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 11,
        "eval_count": 7,
        "message": {
            "role": "assistant",
            "content": '{"answer":"Paris","count":1}',
        },
    }
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=payload))
    ) as client:
        model.native_http_transport = HTTPTransport(client=client, max_retries=0)
        result = model("Answer with Paris and count 1", generation_schema=Answer)

    assert result.consume().answer == "Paris"
    (span,) = spans.get_finished_spans()
    assert span.name == "chat qwen3:8b"
    assert span.attributes["gen_ai.provider.name"] == "ollama"
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert span.attributes["gen_ai.usage.input_tokens"] == 11
    assert span.attributes["gen_ai.usage.output_tokens"] == 7
    output = json.loads(span.attributes["gen_ai.output.messages"])
    assert json.loads(output[0]["parts"][0]["content"]) == {
        "answer": "Paris",
        "count": 1,
    }


def test_ollama_native_stream_records_tool_call(spans, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = OllamaChatCompletion(model_id="qwen3:8b", retry=False)
    chunks = [
        {
            "model": "qwen3:8b",
            "message": {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "lookup", "arguments": {"sku": "1842"}}}
                ],
            },
            "done": False,
        },
        {
            "model": "qwen3:8b",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 3,
            "eval_count": 2,
        },
    ]
    body = b"\n".join(json.dumps(chunk).encode() for chunk in chunks) + b"\n"
    with httpx2.Client(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, content=body))
    ) as client:
        model.native_http_transport = HTTPTransport(client=client, max_retries=0)
        stream = model._execute_model(messages=[], model="qwen3:8b", stream=True)
        assert len(list(stream)) == 2

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert span.attributes["gen_ai.usage.output_tokens"] == 2
    tool = json.loads(span.attributes["gen_ai.output.messages"])[0]["parts"][0]
    assert tool["type"] == "tool_call"
    assert tool["name"] == "lookup"
    assert tool["arguments"] == {"sku": "1842"}


@pytest.mark.asyncio
async def test_ollama_native_async_response_records_output(spans, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = OllamaChatCompletion(model_id="qwen3:8b", retry=False)
    payload = {
        "model": "qwen3:8b",
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": "Hello"},
    }
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=payload))
    ) as client:
        model.native_http_transport = HTTPTransport(async_client=client, max_retries=0)
        result = await model.acall("Say hello")

    assert result.data == "Hello"
    (span,) = spans.get_finished_spans()
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "Hello"}]}
    ]


@pytest.mark.asyncio
async def test_ollama_native_async_stream_records_output(spans, monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = OllamaChatCompletion(model_id="qwen3:8b", retry=False)
    chunks = [
        {"message": {"content": "Hel"}, "done": False},
        {
            "message": {"content": "lo"},
            "done": True,
            "done_reason": "stop",
        },
    ]
    body = b"\n".join(json.dumps(chunk).encode() for chunk in chunks) + b"\n"
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, content=body))
    ) as client:
        model.native_http_transport = HTTPTransport(async_client=client, max_retries=0)
        stream = await model._aexecute_model(messages=[], model="qwen3:8b", stream=True)
        assert len([item async for item in stream]) == 2

    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "Hello"}]}
    ]


@pytest.mark.asyncio
async def test_legacy_httpx_model_records_async_span(spans):
    async def handler(_request):
        return httpx2.Response(200, json={"data": [{"label": "yes"}]})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model = VLLMTextClassifier("classifier", retry=False)
        await model.aclient.aclose()
        model.aclient = client
        result = await model.acall("example")

    assert result.data == ["yes"]
    (span,) = spans.get_finished_spans()
    assert span.attributes["gen_ai.request.model"] == "classifier"
