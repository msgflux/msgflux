"""Tests for the native Google Gemini Interactions provider."""

import asyncio
import json

import httpx2
import msgspec
import pytest

import msgflux as mf
from msgflux.chat_messages import ChatMessages
from msgflux.exceptions import ModelProviderHTTPError
from msgflux.models.http_transport import HTTPTransport
from msgflux.models.providers.gemini import GeminiChatCompletion
from msgflux.tools import ToolCatalog, ToolOutcome

THOUGHT_SIGNATURE = "opaque-thought-signature"


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com")


def usage(
    *,
    input_tokens=6,
    output_tokens=1,
    thought_tokens=4,
    cached_tokens=0,
):
    return {
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_thought_tokens": thought_tokens,
        "total_cached_tokens": cached_tokens,
        "total_tokens": input_tokens + output_tokens + thought_tokens,
    }


def text_payload(
    text="OK",
    *,
    thought="Thinking.",
    signature=THOUGHT_SIGNATURE,
    status="completed",
):
    steps = []
    if thought is not None or signature is not None:
        step = {"type": "thought"}
        if thought is not None:
            step["summary"] = [{"type": "text", "text": thought}]
        if signature is not None:
            step["signature"] = signature
        steps.append(step)
    steps.append({"type": "model_output", "content": [{"type": "text", "text": text}]})
    return {
        "object": "interaction",
        "model": "gemini-3.8-flash",
        "status": status,
        "steps": steps,
        "usage": usage(),
    }


def tool_payload():
    return {
        "object": "interaction",
        "model": "gemini-3.8-flash",
        "status": "requires_action",
        "steps": [
            {
                "type": "thought",
                "summary": [{"type": "text", "text": "Calling inventory."}],
                "signature": THOUGHT_SIGNATURE,
            },
            {
                "type": "function_call",
                "id": "call_1",
                "name": "inventory_lookup",
                "arguments": {
                    "sku": "BOLT-17",
                    "quantity": 3,
                    "active": True,
                    "labels": ["rush", "blue"],
                    "note": None,
                },
            },
        ],
        "usage": usage(input_tokens=161, output_tokens=3, thought_tokens=12),
    }


def catalog(*, choice=None):
    return ToolCatalog.from_function_schemas(
        [
            {
                "type": "function",
                "function": {
                    "name": "inventory_lookup",
                    "description": "Check inventory with typed filters.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sku": {"type": "string"},
                            "quantity": {"type": "integer"},
                            "active": {"type": "boolean"},
                            "labels": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["rush", "blue"],
                                },
                            },
                            "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                        },
                        "required": [
                            "sku",
                            "quantity",
                            "active",
                            "labels",
                            "note",
                        ],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        choice=choice,
    )


def sse(frames):
    return "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames)


def stream_frames(*, summary="Thinking.", tool_arguments=False):
    frames = [
        {
            "event_type": "interaction.created",
            "interaction": {
                "object": "interaction",
                "model": "gemini-3.8-flash",
            },
        },
        {
            "event_type": "interaction.status_update",
            "status": "in_progress",
        },
        {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
    ]
    if summary is not None:
        frames.append(
            {
                "event_type": "step.delta",
                "index": 0,
                "delta": {
                    "type": "thought_summary",
                    "content": {"type": "text", "text": summary},
                },
            }
        )
    frames.append(
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {
                "type": "thought_signature",
                "signature": THOUGHT_SIGNATURE,
            },
        }
    )
    frames.append({"event_type": "step.stop", "index": 0})
    if tool_arguments:
        frames.extend(
            [
                {
                    "event_type": "step.start",
                    "index": 1,
                    "step": {
                        "type": "function_call",
                        "id": "call_1",
                        "name": "inventory_lookup",
                    },
                },
                {
                    "event_type": "step.delta",
                    "index": 1,
                    "delta": {
                        "type": "arguments_delta",
                        "arguments": '{"sku":"BOLT-17",',
                    },
                },
                {
                    "event_type": "step.delta",
                    "index": 1,
                    "delta": {
                        "type": "arguments_delta",
                        "arguments": '"quantity":3,"active":true,',
                    },
                },
                {
                    "event_type": "step.delta",
                    "index": 1,
                    "delta": {
                        "type": "arguments_delta",
                        "arguments": ('"labels":["rush","blue"],"note":null}'),
                    },
                },
                {"event_type": "step.stop", "index": 1},
            ]
        )
        status = "requires_action"
    else:
        frames.extend(
            [
                {
                    "event_type": "step.start",
                    "index": 1,
                    "step": {"type": "model_output"},
                },
                {
                    "event_type": "step.delta",
                    "index": 1,
                    "delta": {"type": "text", "text": "OK"},
                },
                {"event_type": "step.stop", "index": 1},
            ]
        )
        status = "completed"
    frames.append(
        {
            "event_type": "interaction.completed",
            "interaction": {
                "object": "interaction",
                "model": "gemini-3.8-flash",
                "status": status,
                "usage": usage(),
            },
        }
    )
    return frames


def test_defaults_registry_and_factory():
    model = GeminiChatCompletion(model_id="gemini-3.8-flash")

    assert model.provider == "gemini"
    assert model.api_mode == "interactions"
    assert model.supported_api_modes == ("interactions",)
    assert model.store is False
    assert model.api_version == "v1"
    assert model._get_base_url() == "https://generativelanguage.googleapis.com"
    assert model._get_api_key() == "fake-gemini-key"
    assert model._uses_canonical_history is True

    from msgflux.models.registry import model_registry

    assert "gemini" in model_registry.get("chat_completion", {})
    built = mf.Model.chat_completion("gemini/gemini-3.8-flash")
    assert built.provider == "gemini"
    assert built.api_mode == "interactions"


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiChatCompletion(model_id="gemini-3.8-flash")


def test_native_text_request_response_and_usage():
    requests = []

    def handler(request):
        requests.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": json.loads(request.content),
            }
        )
        return httpx2.Response(200, json=text_payload())

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model = GeminiChatCompletion(
            model_id="gemini-3.8-flash",
            reasoning_effort="low",
            max_tokens=512,
            temperature=0.2,
            stop=["END"],
            retry=False,
        )
        model.http_transport = HTTPTransport(client=client, max_retries=0)
        response = model("Reply with exactly OK", system_prompt="Be terse.")

    assert response.consume() == "OK"
    assert response.reasoning == "Thinking."
    assert response.response_type == "text_generation"
    assert response.metadata.usage.input_tokens == 6
    assert response.metadata.usage.output_tokens == 5
    assert response.metadata.usage.input_tokens_details.cached_tokens == 0
    assert response.metadata.usage.output_tokens_details.reasoning_tokens == 4
    assert response.metadata.finish_reason == "stop"

    request = requests[0]
    assert request["url"] == (
        "https://generativelanguage.googleapis.com/v1/interactions"
    )
    assert request["headers"]["x-goog-api-key"] == "fake-gemini-key"
    body = request["body"]
    assert body == {
        "model": "gemini-3.8-flash",
        "input": [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "Reply with exactly OK"}],
            }
        ],
        "store": False,
        "system_instruction": "Be terse.",
        "generation_config": {
            "max_output_tokens": 512,
            "temperature": 0.2,
            "stop_sequences": ["END"],
            "thinking_level": "low",
            "thinking_summaries": "auto",
        },
        "stream": False,
    }
    assert "messages" not in body

    history = response.history_items
    assert history[0]["text"] == "Thinking."
    assert history[0]["provider_state"]["data"] == text_payload()["steps"][0]
    assert history[1]["content"] == "OK"
    assert history[1]["provider_state"]["data"] == text_payload()["steps"][1]


@pytest.mark.parametrize("restore", [False, True])
def test_thought_and_model_output_steps_replay_after_restore(restore):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx2.Response(200, json=text_payload())
        return httpx2.Response(200, json=text_payload(text="FOLLOW-UP"))

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model = GeminiChatCompletion(
            model_id="gemini-3.8-flash",
            reasoning_effort="low",
            retry=False,
        )
        model.http_transport = HTTPTransport(client=client, max_retries=0)
        first = model("Reply with exactly OK")
        assert first.consume() == "OK"

        history = ChatMessages(
            json.loads(
                json.dumps(
                    [
                        {"role": "user", "content": "Reply with exactly OK"},
                        *first.history_items,
                        {"role": "user", "content": "Continue."},
                    ]
                )
            )
        )
        if restore:
            serialized = model.serialize()
            model = mf.Model.from_serialized(
                serialized["provider"],
                serialized["model_type"],
                serialized["state"],
            )
            model.http_transport = HTTPTransport(client=client, max_retries=0)
        second = model(history, system_prompt="Be terse.")
        assert second.consume() == "FOLLOW-UP"

    replay = requests[1]
    assert replay["system_instruction"] == "Be terse."
    assert replay["input"] == [
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "Reply with exactly OK"}],
        },
        {
            "type": "thought",
            "signature": THOUGHT_SIGNATURE,
            "summary": [{"type": "text", "text": "Thinking."}],
        },
        {
            "type": "model_output",
            "content": [{"type": "text", "text": "OK"}],
        },
        {
            "type": "user_input",
            "content": [{"type": "text", "text": "Continue."}],
        },
    ]


def test_native_tools_forced_choice_and_tool_history_replay():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx2.Response(200, json=tool_payload())
        return httpx2.Response(200, json=text_payload(text="Available: 7"))

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model = GeminiChatCompletion(
            model_id="gemini-3.8-flash",
            reasoning_effort="low",
            retry=False,
        )
        model.http_transport = HTTPTransport(client=client, max_retries=0)
        response = model(
            "Call inventory_lookup for BOLT-17.",
            tool_catalog=catalog(choice="inventory_lookup"),
        )

        assert response.response_type == "tool_call"
        (intent,) = list(response.get_tool_intents())
        assert intent.id == "call_1"
        assert intent.name == "inventory_lookup"
        assert intent.arguments == {
            "sku": "BOLT-17",
            "quantity": 3,
            "active": True,
            "labels": ["rush", "blue"],
            "note": None,
        }
        assert response.reasoning == "Calling inventory."
        (call_item,) = [
            item
            for item in response.history_items
            if item.get("type") == "function_call"
        ]
        assert call_item["provider_state"]["data"] == tool_payload()["steps"][1]

        outcomes = response.render_tool_outcomes(
            [
                ToolOutcome.completed(
                    intent,
                    {"available": 7, "warehouse": "A1"},
                )
            ]
        )
        assert [item["type"] for item in outcomes] == ["function_call_output"]
        assert not any(
            item.get("type") == "function_call"
            or (item.get("type") == "message" and item.get("role") == "assistant")
            for item in outcomes
        )

        history = ChatMessages(
            json.loads(
                json.dumps(
                    [
                        {
                            "role": "user",
                            "content": "Call inventory_lookup for BOLT-17.",
                        },
                        *response.history_items,
                        *outcomes,
                    ]
                )
            )
        )
        final = model(history, tool_catalog=catalog(choice="none"))
        assert final.consume() == "Available: 7"

    first_request = requests[0]
    assert first_request["tools"] == [
        {
            "type": "function",
            "name": "inventory_lookup",
            "description": "Check inventory with typed filters.",
            "parameters": catalog().portable_schemas()[0]["function"]["parameters"],
        }
    ]
    assert "function" not in first_request["tools"][0]
    assert first_request["generation_config"]["tool_choice"] == {
        "allowed_tools": {"mode": "any", "tools": ["inventory_lookup"]}
    }
    replay = requests[1]
    assert [step["type"] for step in replay["input"]] == [
        "user_input",
        "thought",
        "function_call",
        "function_result",
    ]
    assert replay["input"][1]["signature"] == THOUGHT_SIGNATURE
    assert replay["input"][2]["arguments"] == {
        "sku": "BOLT-17",
        "quantity": 3,
        "active": True,
        "labels": ["rush", "blue"],
        "note": None,
    }
    assert replay["input"][3]["type"] == "function_result"
    assert replay["input"][3]["call_id"] == "call_1"
    assert replay["input"][3]["name"] == "inventory_lookup"
    assert json.loads(replay["input"][3]["result"]) == {
        "available": 7,
        "warehouse": "A1",
    }
    assert sum(step["type"] == "function_call" for step in replay["input"]) == 1
    assert sum(step["type"] == "function_result" for step in replay["input"]) == 1
    assert replay["generation_config"]["tool_choice"] == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("summary", ["Thinking.", None])
async def test_stream_thought_signature_and_text(asynchronous, summary):
    data = sse(stream_frames(summary=summary))

    def handler(_request):
        return httpx2.Response(
            200,
            text=data,
            headers={"content-type": "text/event-stream"},
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler)
        ) as aclient:
            model = GeminiChatCompletion(
                model_id="gemini-3.8-flash",
                reasoning_effort="low",
                retry=False,
            )
            model.http_transport = HTTPTransport(
                client=client,
                async_client=aclient,
                max_retries=0,
            )
            response = (
                await model.acall("Reply with exactly OK", stream=True)
                if asynchronous
                else model("Reply with exactly OK", stream=True)
            )

            async def consume():
                return [chunk async for chunk in response.consume()]

            chunks = await asyncio.wait_for(consume(), timeout=2)

    assert chunks == ["OK"]
    assert response.response_type == "text_generation"
    assert response.error is None
    assert response.reasoning == summary
    assert response.metadata.usage.input_tokens == 6
    assert response.metadata.usage.output_tokens == 5
    assert response.metadata.usage.output_tokens_details.reasoning_tokens == 4
    items = response.chat_accumulator.snapshot()
    assert [item["type"] for item in items] == ["reasoning", "message"]
    assert items[0]["provider_state"]["data"]["signature"] == THOUGHT_SIGNATURE
    assert items[0]["provider_state"]["data"]["type"] == "thought"
    assert items[1]["content"] == "OK"
    assert sum(item.get("type") == "reasoning" for item in items) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stream_nested_tool_arguments_and_single_call(asynchronous):
    data = sse(stream_frames(summary="Thinking.", tool_arguments=True))

    def handler(_request):
        return httpx2.Response(
            200,
            text=data,
            headers={"content-type": "text/event-stream"},
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler)
        ) as aclient:
            model = GeminiChatCompletion(
                model_id="gemini-3.8-flash",
                reasoning_effort="low",
                retry=False,
            )
            model.http_transport = HTTPTransport(
                client=client,
                async_client=aclient,
                max_retries=0,
            )
            response = (
                await model.acall(
                    "Call inventory_lookup for BOLT-17.",
                    stream=True,
                    tool_catalog=catalog(),
                )
                if asynchronous
                else model(
                    "Call inventory_lookup for BOLT-17.",
                    stream=True,
                    tool_catalog=catalog(),
                )
            )

            async def consume():
                return [chunk async for chunk in response.consume()]

            chunks = await asyncio.wait_for(consume(), timeout=2)

    assert chunks == []
    assert response.response_type == "tool_call"
    assert response.error is None
    intents = list(response.get_tool_intents())
    assert len(intents) == 1
    assert intents[0].name == "inventory_lookup"
    assert intents[0].arguments == {
        "sku": "BOLT-17",
        "quantity": 3,
        "active": True,
        "labels": ["rush", "blue"],
        "note": None,
    }
    items = response.chat_accumulator.snapshot()
    assert [item["type"] for item in items] == ["reasoning", "function_call"]
    assert sum(item["type"] == "function_call" for item in items) == 1
    assert items[1]["arguments"] == (
        '{"sku":"BOLT-17","quantity":3,"active":true,'
        '"labels":["rush","blue"],"note":null}'
    )
    assert items[0]["provider_state"]["data"]["signature"] == THOUGHT_SIGNATURE


def test_stream_full_function_arguments_without_deltas():
    frames = [
        {
            "event_type": "step.start",
            "index": 0,
            "step": {
                "type": "function_call",
                "id": "call_1",
                "name": "inventory_lookup",
                "arguments": {"sku": "BOLT-17", "quantity": 3},
            },
        },
        {"event_type": "step.stop", "index": 0},
        {
            "event_type": "interaction.completed",
            "interaction": {
                "status": "requires_action",
                "usage": usage(input_tokens=4, output_tokens=2, thought_tokens=0),
            },
        },
    ]
    data = sse(frames)

    def handler(_request):
        return httpx2.Response(
            200,
            text=data,
            headers={"content-type": "text/event-stream"},
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model = GeminiChatCompletion(model_id="gemini-3.8-flash", retry=False)
        model.http_transport = HTTPTransport(client=client, max_retries=0)
        response = model("Call inventory_lookup.", stream=True)
        assert asyncio.run(_consume(response)) == []

    (intent,) = list(response.get_tool_intents())
    assert intent.arguments == {"sku": "BOLT-17", "quantity": 3}
    assert len(response.chat_accumulator.snapshot()) == 1


async def _consume(response):
    return [chunk async for chunk in response.consume()]


def test_model_restore_preserves_native_configuration():
    model = GeminiChatCompletion(
        model_id="gemini-3.8-flash",
        reasoning_effort="high",
        store=True,
        thinking_summaries="none",
        api_version="v1",
        retry=False,
    )
    serialized = model.serialize()
    restored = mf.Model.from_serialized(
        serialized["provider"],
        serialized["model_type"],
        serialized["state"],
    )

    assert restored.provider == "gemini"
    assert restored.api_mode == "interactions"
    assert restored.store is True
    assert restored.thinking_summaries == "none"
    assert restored.api_version == "v1"
    assert restored.sampling_run_params["reasoning_effort"] == "high"
    assert restored.api_adapter.name == "interactions"
    assert isinstance(restored.http_transport, HTTPTransport)


def test_image_input_is_sent_as_native_image_content():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=text_payload(thought=None, signature=None))

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model = GeminiChatCompletion(model_id="gemini-3.8-flash", retry=False)
        model.http_transport = HTTPTransport(client=client, max_retries=0)
        response = model(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                }
            ]
        )
        assert response.consume() == "OK"

    assert requests[0]["input"] == [
        {
            "type": "user_input",
            "content": [
                {"type": "text", "text": "Describe."},
                {"type": "image", "data": "AAAA", "mime_type": "image/png"},
            ],
        }
    ]


def test_http_error_is_structured():
    def handler(_request):
        return httpx2.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "Invalid argument.",
                    "status": "INVALID_ARGUMENT",
                }
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model = GeminiChatCompletion(model_id="gemini-3.8-flash", retry=False)
        model.http_transport = HTTPTransport(client=client, max_retries=0)
        with pytest.raises(ModelProviderHTTPError) as error:
            model("Hello")

    assert error.value.status_code == 400
    assert error.value.description == "Invalid argument."
    assert error.value.provider == "gemini"
    assert error.value.model_id == "gemini-3.8-flash"


def test_unsupported_options_are_rejected_explicitly():
    with pytest.raises(ValueError, match="parallel_tool_calls=False"):
        GeminiChatCompletion(
            model_id="gemini-3.8-flash",
            parallel_tool_calls=False,
        )
    with pytest.raises(ValueError, match="reasoning_max_tokens"):
        GeminiChatCompletion(
            model_id="gemini-3.8-flash",
            reasoning_max_tokens=100,
        )

    model = GeminiChatCompletion(model_id="gemini-3.8-flash", retry=False)
    with pytest.raises(ValueError, match="prefilling"):
        model("Hello", prefilling="Sure")
    with pytest.raises(ValueError, match="cannot disable thinking"):
        model(
            "Hello",
            extra_body={"generation_config": {"thinking_level": "none"}},
        )
    with pytest.raises(ValueError, match="extra_body"):
        model("Hello", extra_body={"background": True})
    with pytest.raises(ValueError, match="extra_body"):
        model("Hello", previous_interaction_id="interaction_1")


@pytest.mark.asyncio
async def test_async_nonstream_text_uses_async_transport():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=text_payload())

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as aclient:
        model = GeminiChatCompletion(
            model_id="gemini-3.8-flash",
            reasoning_effort="low",
            retry=False,
        )
        model.http_transport = HTTPTransport(async_client=aclient, max_retries=0)
        response = await model.acall("Reply with exactly OK")

    assert response.consume() == "OK"
    assert response.reasoning == "Thinking."
    assert requests[0]["stream"] is False
    assert requests[0]["store"] is False


def test_structured_output_uses_native_response_format():
    requests = []

    class Answer(msgspec.Struct):
        answer: str

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "status": "completed",
                "steps": [
                    {
                        "type": "model_output",
                        "content": [{"type": "text", "text": '{"answer":"OK"}'}],
                    }
                ],
                "usage": usage(),
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model = GeminiChatCompletion(model_id="gemini-3.8-flash", retry=False)
        model.http_transport = HTTPTransport(client=client, max_retries=0)
        response = model("Return JSON.", generation_schema=Answer)

    assert response.response_type == "structured"
    assert response.consume() == {"answer": "OK"}
    assert requests[0]["response_format"] == {
        "type": "text",
        "mime_type": "application/json",
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    }


def test_stream_tool_outcomes_replay_without_duplicate_function_call():
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if body.get("stream"):
            return httpx2.Response(
                200,
                text=sse(stream_frames(summary="Thinking.", tool_arguments=True)),
                headers={"content-type": "text/event-stream"},
            )
        return httpx2.Response(200, json=text_payload(text="Available: 7"))

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model = GeminiChatCompletion(
            model_id="gemini-3.8-flash",
            reasoning_effort="low",
            retry=False,
        )
        model.http_transport = HTTPTransport(client=client, max_retries=0)
        response = model(
            "Call inventory_lookup for BOLT-17.",
            stream=True,
            tool_catalog=catalog(),
        )
        asyncio.run(_consume(response))

        snapshot = response.chat_accumulator.snapshot()
        (intent,) = list(response.get_tool_intents())
        outcomes = response.render_tool_outcomes(
            [
                ToolOutcome.completed(
                    intent,
                    {"available": 7, "warehouse": "A1"},
                )
            ]
        )
        assert [item["type"] for item in outcomes] == ["function_call_output"]
        assert not any(item.get("type") == "function_call" for item in outcomes)

        messages = ChatMessages(
            json.loads(
                json.dumps(
                    [
                        {
                            "role": "user",
                            "content": "Call inventory_lookup for BOLT-17.",
                        },
                        *snapshot,
                        *outcomes,
                    ]
                )
            )
        )
        final = model(messages, tool_catalog=catalog(choice="none"))
        assert final.consume() == "Available: 7"

    replay = requests[1]
    types = [step["type"] for step in replay["input"]]
    assert types.count("function_call") == 1
    assert types.count("function_result") == 1
    assert types == ["user_input", "thought", "function_call", "function_result"]
    assert replay["input"][1]["signature"] == THOUGHT_SIGNATURE
    assert replay["input"][2]["id"] == "call_1"
    assert replay["input"][3]["call_id"] == "call_1"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_failed_interaction_resource_raises(asynchronous):
    def handler(_request):
        return httpx2.Response(
            200,
            json={
                "status": "failed",
                "steps": [],
                "usage": usage(),
                "errors": [{"code": "INTERNAL", "message": "provider boom"}],
            },
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler)
        ) as aclient:
            model = GeminiChatCompletion(model_id="gemini-3.8-flash", retry=False)
            model.http_transport = HTTPTransport(
                client=client,
                async_client=aclient,
                max_retries=0,
            )
            with pytest.raises(ModelProviderHTTPError) as error:
                if asynchronous:
                    await model.acall("Hello")
                else:
                    model("Hello")

    assert "provider boom" in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_truncated_stream_without_completion_is_an_error(asynchronous):
    frames = stream_frames(summary="Thinking.")
    frames = [
        frame for frame in frames if frame.get("event_type") != "interaction.completed"
    ]

    def handler(_request):
        return httpx2.Response(
            200,
            text=sse(frames),
            headers={"content-type": "text/event-stream"},
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler)
        ) as aclient:
            model = GeminiChatCompletion(
                model_id="gemini-3.8-flash",
                reasoning_effort="low",
                retry=False,
            )
            model.http_transport = HTTPTransport(
                client=client,
                async_client=aclient,
                max_retries=0,
            )
            response = (
                await model.acall("Hello", stream=True)
                if asynchronous
                else model("Hello", stream=True)
            )
            with pytest.raises(
                ModelProviderHTTPError,
                match=r"before interaction\.completed",
            ):
                await asyncio.wait_for(_consume(response), timeout=2)

    assert response.error is not None
    assert "before interaction.completed" in str(response.error)


def test_terminal_stream_chunk_carries_interaction_identity():
    frames = [
        {
            "event_type": "interaction.created",
            "interaction": {
                "id": "v1_interaction",
                "model": "gemini-3.8-flash",
                "status": "in_progress",
            },
        },
        {
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "model_output"},
        },
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "text", "text": "OK"},
        },
        {"event_type": "step.stop", "index": 0},
        {
            "event_type": "interaction.completed",
            "interaction": {
                "id": "v1_interaction",
                "model": "gemini-3.8-flash",
                "status": "completed",
                "usage": usage(),
            },
        },
    ]
    model = GeminiChatCompletion(model_id="gemini-3.8-flash", retry=False)

    chunks = list(model._expand_interaction_stream(iter(frames)))

    terminal = chunks[-1]
    assert terminal["id"] == "v1_interaction"
    assert terminal["model"] == "gemini-3.8-flash"
    assert terminal["status"] == "completed"
    assert terminal["usage"]["total_input_tokens"] == 6


def test_stream_preserves_multi_part_thought_summary_and_signature():
    frames = [
        {
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "thought"},
        },
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {
                "type": "thought_summary",
                "content": {"type": "text", "text": "First."},
            },
        },
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {
                "type": "thought_summary",
                "content": {"type": "text", "text": "Second."},
            },
        },
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {
                "type": "thought_signature",
                "signature": THOUGHT_SIGNATURE,
            },
        },
        {"event_type": "step.stop", "index": 0},
        {
            "event_type": "interaction.completed",
            "interaction": {"status": "completed", "usage": usage()},
        },
    ]
    model = GeminiChatCompletion(model_id="gemini-3.8-flash", retry=False)

    chunks = list(model._expand_interaction_stream(iter(frames)))

    thought_delta = next(
        chunk.choices[0].delta
        for chunk in chunks
        if chunk.choices[0].delta.get("thought_step") is not None
    )
    assert thought_delta["thought_step"]["signature"] == THOUGHT_SIGNATURE
    assert thought_delta["thought_step"]["summary"] == [
        {"type": "text", "text": "First."},
        {"type": "text", "text": "Second."},
    ]
