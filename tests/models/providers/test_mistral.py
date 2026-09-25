"""Tests for the Mistral AI OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import asyncio
import json

import httpx2
import msgspec
import pytest
from msgtrace.sdk.tracer import tracer_manager
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from msgflux.chat_messages import ChatMessages
from msgflux.models.chat_transport import HTTPChatTransport
from msgflux.tools import ToolCatalogEntry, ToolCatalogView, ToolRef
from tests.models._chat_transport import EndpointMockTransport


@pytest.fixture(autouse=True)
def mistral_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")


@pytest.fixture
def mock_mistral_client():
    from msgflux.models.providers.mistral import MistralChatCompletion

    client = MagicMock()
    async_client = MagicMock()
    transport = EndpointMockTransport(client.return_value, async_client.return_value)
    with patch.object(MistralChatCompletion, "chat_transport", transport):
        yield client


@pytest.fixture
def mistral_spans(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracer_manager, "_tracer", provider.get_tracer("mistral-test"))
    yield exporter
    provider.shutdown()


def test_mistral_shared_http_transport_traces_schema_request(mistral_spans):
    from msgflux.models.providers.mistral import MistralChatCompletion

    class Answer(msgspec.Struct):
        answer: str
        count: int

    payload = {
        "id": "mistral-response",
        "model": "mistral-small-latest",
        "usage": {"prompt_tokens": 8, "completion_tokens": 3, "total_tokens": 11},
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"answer":"Paris","count":1}',
                },
            }
        ],
    }
    requests = []

    def handler(request):
        requests.append(json.loads(request.read()))
        return httpx2.Response(200, json=payload)

    model = MistralChatCompletion(
        model_id="mistral-small-latest",
        reasoning_effort="high",
        temperature=0.25,
    )
    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model.chat_transport = HTTPChatTransport(client=client)
        response = model("Return Paris and count 1", generation_schema=Answer)

    assert response.consume().answer == "Paris"
    assert requests[0]["model"] == "mistral-small-latest"
    assert requests[0]["reasoning_effort"] == "high"
    assert requests[0]["temperature"] == 0.25
    assert requests[0]["response_format"]["type"] == "json_schema"

    (span,) = mistral_spans.get_finished_spans()
    assert span.attributes["gen_ai.provider.name"] == "mistral"
    assert span.attributes["gen_ai.request.reasoning.level"] == "high"
    assert span.attributes["gen_ai.request.temperature"] == 0.25
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert span.attributes["gen_ai.usage.input_tokens"] == 8
    assert span.attributes["gen_ai.usage.output_tokens"] == 3
    output = json.loads(span.attributes["gen_ai.output.messages"])
    assert json.loads(output[0]["parts"][0]["content"]) == {
        "answer": "Paris",
        "count": 1,
    }


@pytest.mark.asyncio
async def test_mistral_async_http_transport_traces_tool_schema(mistral_spans):
    from msgflux.models.providers.mistral import MistralChatCompletion

    payload = {
        "id": "mistral-tool-response",
        "model": "mistral-small-latest",
        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "lookup_inventory",
                                "arguments": '{"sku":"1842"}',
                            },
                        }
                    ],
                },
            }
        ],
    }
    requests = []

    async def handler(request):
        requests.append(json.loads(request.read()))
        return httpx2.Response(200, json=payload)

    catalog = ToolCatalogView(
        library_id="warehouse_tools",
        thread_id="thread_1",
        entries=(
            ToolCatalogEntry(
                ref=ToolRef(library_id="warehouse_tools", tool_id="lookup_inventory"),
                description="Look up a SKU.",
                input_schema={
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                },
                strict=True,
            ),
        ),
        choice="lookup_inventory",
    )
    model = MistralChatCompletion(model_id="mistral-small-latest")
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model.chat_transport = HTTPChatTransport(async_client=client)
        await model.acall("Look up SKU-1842", tool_catalog=catalog)

    assert requests[0]["tools"][0]["function"]["name"] == "lookup_inventory"
    assert requests[0]["tool_choice"]["function"]["name"] == "lookup_inventory"

    (span,) = mistral_spans.get_finished_spans()
    assert span.attributes["gen_ai.response.finish_reasons"] == ("tool_calls",)
    assert span.attributes["gen_ai.usage.input_tokens"] == 9
    assert span.attributes["gen_ai.usage.output_tokens"] == 4
    output = json.loads(span.attributes["gen_ai.output.messages"])
    assert output[0]["parts"][0] == {
        "type": "tool_call",
        "name": "lookup_inventory",
        "id": "call_1",
        "arguments": {"sku": "1842"},
    }


@pytest.mark.asyncio
async def test_mistral_async_stream_transport_traces_structured_output(mistral_spans):
    from msgflux.models.providers.mistral import MistralChatCompletion

    events = [
        {
            "id": "mistral-stream-response",
            "model": "mistral-small-latest",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": "Check."}],
                            },
                            {"type": "text", "text": "OK"},
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "mistral-stream-response",
            "model": "mistral-small-latest",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
        },
    ]
    body = (
        b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        + b"data: [DONE]\n\n"
    )

    async def handler(_request):
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
        )

    model = MistralChatCompletion(model_id="mistral-small-latest")
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        model.chat_transport = HTTPChatTransport(async_client=client)
        response = await model.acall("Reply with OK", stream=True)
        chunks = [chunk async for chunk in response.consume()]

    assert chunks == ["OK"]
    assert response.reasoning == "Check."
    (span,) = mistral_spans.get_finished_spans()
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert span.attributes["gen_ai.usage.input_tokens"] == 6
    assert span.attributes["gen_ai.usage.output_tokens"] == 2
    output = json.loads(span.attributes["gen_ai.output.messages"])
    assert output[0]["parts"] == [{"type": "text", "content": "OK"}]


def test_mistral_sync_stream_transport_traces_structured_output(mistral_spans):
    from msgflux.models.providers.mistral import MistralChatCompletion

    events = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": [
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": "Check."}],
                            },
                            {"type": "text", "text": "OK"},
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
        },
    ]
    body = (
        b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        + b"data: [DONE]\n\n"
    )

    model = MistralChatCompletion(model_id="mistral-small-latest")
    with httpx2.Client(
        transport=httpx2.MockTransport(
            lambda _request: httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body,
            )
        )
    ) as client:
        model.chat_transport = HTTPChatTransport(client=client)
        response = model("Reply with OK", stream=True)

        async def collect():
            return [chunk async for chunk in response.consume()]

        chunks = asyncio.run(collect())

    assert chunks == ["OK"]
    assert response.reasoning == "Check."
    (span,) = mistral_spans.get_finished_spans()
    assert span.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert span.attributes["gen_ai.usage.input_tokens"] == 6
    assert span.attributes["gen_ai.usage.output_tokens"] == 2
    output = json.loads(span.attributes["gen_ai.output.messages"])
    assert output[0]["parts"] == [{"type": "text", "content": "OK"}]


def test_mistral_defaults_to_chat_completions():
    from msgflux.models.providers.mistral import MistralChatCompletion

    model = MistralChatCompletion(model_id="mistral-small-latest")

    assert model.provider == "mistral"
    assert model.api_mode == "chat_completions"
    assert model.supported_api_modes == ("chat_completions",)


def test_mistral_reads_base_url_and_api_key():
    from msgflux.models.providers.mistral import MistralChatCompletion

    model = MistralChatCompletion(model_id="mistral-small-latest")

    assert model._get_base_url() == "https://api.mistral.ai/v1"
    assert model._get_api_key() == "test-key"


def test_mistral_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.mistral import MistralChatCompletion

    monkeypatch.delenv("MISTRAL_API_KEY")

    with pytest.raises(ValueError, match="MISTRAL_API_KEY"):
        MistralChatCompletion(model_id="mistral-small-latest")


def test_mistral_models_registered():
    from msgflux.models.registry import model_registry

    assert "mistral" in model_registry.get("chat_completion", {})


def test_mistral_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("mistral/mistral-small-latest")

    assert model.provider == "mistral"
    assert model.model_id == "mistral-small-latest"


def test_mistral_rejects_responses_api_mode():
    from msgflux.models.providers.mistral import MistralChatCompletion

    with pytest.raises(ValueError, match="responses"):
        MistralChatCompletion(model_id="mistral-small-latest", api_mode="responses")


def test_mistral_structured_content_round_trip(mock_mistral_client):
    from msgflux.models.providers.mistral import MistralChatCompletion

    mock_mistral_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=[
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": "Check digits."}],
                            },
                            {"type": "text", "text": "OK"},
                        ],
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
    )
    model = MistralChatCompletion(
        model_id="mistral-small-latest", reasoning_effort="high"
    )
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Check digits."

    request = mock_mistral_client.return_value.chat.completions.create.call_args.kwargs
    assert request["reasoning_effort"] == "high"


def test_mistral_plain_string_content_still_works(mock_mistral_client):
    from msgflux.models.providers.mistral import MistralChatCompletion

    mock_mistral_client.return_value.chat.completions.create.return_value = (
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content="OK",
                        tool_calls=None,
                        audio=None,
                        annotations=None,
                    ),
                )
            ],
        )
    )
    model = MistralChatCompletion(model_id="mistral-small-latest")
    response = model("Reply with exactly: OK")

    assert response.consume() == "OK"
    assert response.reasoning is None


def test_mistral_replays_thinking_as_native_chunk():
    from msgflux.models.providers.mistral import MistralChatCompletion

    model = MistralChatCompletion(model_id="mistral-small-latest")
    messages = ChatMessages()
    messages.add_reasoning("private chain")
    messages.add_assistant("first answer")
    messages.add_user("follow up")

    params = model._build_generation_params(
        messages,
        system_prompt=None,
        prefilling=None,
        tool_catalog=None,
    )

    assert params["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": [{"type": "text", "text": "private chain"}],
                },
                {"type": "text", "text": "first answer"},
            ],
        },
        {"role": "user", "content": "follow up"},
    ]


def test_mistral_stream_expands_list_delta(mock_mistral_client):
    from msgflux.models.providers.mistral import MistralChatCompletion

    model = MistralChatCompletion(model_id="mistral-small-latest")
    chunk = SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason=None,
                logprobs=None,
                delta=SimpleNamespace(
                    content=[
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "Check."}],
                        },
                        {"type": "text", "text": "OK"},
                    ],
                    tool_calls=None,
                ),
            )
        ],
    )

    expanded = model._expand_mistral_chunk(chunk)

    assert len(expanded) == 2
    assert expanded[0].choices[0].delta.reasoning_content == "Check."
    assert expanded[0].choices[0].delta.content is None
    assert expanded[1].choices[0].delta.content == "OK"


def test_mistral_api_key_env_override(monkeypatch):
    from msgflux.models.providers.mistral import MistralChatCompletion

    monkeypatch.setenv("MISTRAL_ACME_KEY", "mistral-acme-key")
    model = MistralChatCompletion(
        model_id="mistral-small-latest", api_key_env="MISTRAL_ACME_KEY"
    )

    assert model._get_api_key() == "mistral-acme-key"
