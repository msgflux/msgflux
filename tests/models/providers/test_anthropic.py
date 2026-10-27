"""Tests for the Anthropic Messages API provider."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest

from msgflux.chat_messages import ChatMessages


@pytest.fixture(autouse=True)
def anthropic_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")


@pytest.fixture
def mock_anthropic_clients():
    client = MagicMock()
    aclient = MagicMock()
    with (
        patch("msgflux.models.http_transport.httpx2.Client", return_value=client),
        patch(
            "msgflux.models.http_transport.httpx2.AsyncClient",
            return_value=aclient,
        ),
    ):
        yield client, aclient


def _native_text_response(**overrides):
    payload = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-4-8",
        "content": [
            {
                "type": "thinking",
                "thinking": "Checking the request.",
                "signature": "sig-1",
            },
            {"type": "text", "text": "OK"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }
    payload.update(overrides)
    return payload


def test_anthropic_defaults_to_messages_mode(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    model = AnthropicChatCompletion(model_id="claude-opus-4-8")

    assert model.provider == "anthropic"
    assert model.api_mode == "anthropic_messages"
    assert model.supported_api_modes == ("anthropic_messages",)


def test_anthropic_reads_base_url_and_api_key(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    model = AnthropicChatCompletion(model_id="claude-opus-4-8")

    assert model._get_base_url() == "https://api.anthropic.com"
    assert model._get_api_key() == "test-key"
    assert model._native_url() == "https://api.anthropic.com/v1/messages"
    assert model._native_headers() == {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    assert model.credential_resolver.resolve(model).headers == {"x-api-key": "test-key"}


def test_anthropic_missing_api_key_raises(monkeypatch):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    monkeypatch.delenv("ANTHROPIC_API_KEY")

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        AnthropicChatCompletion(model_id="claude-opus-4-8")


def test_anthropic_models_registered():
    from msgflux.models.registry import model_registry

    assert "anthropic" in model_registry.get("chat_completion", {})


def test_anthropic_resolves_through_model_factory():
    import msgflux as mf

    model = mf.Model.chat_completion("anthropic/claude-opus-4-8")

    assert model.provider == "anthropic"
    assert model.model_id == "claude-opus-4-8"


def test_anthropic_text_thinking_round_trip(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    client, _ = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _native_text_response()
    client.request.return_value = response

    model = AnthropicChatCompletion(model_id="claude-opus-4-8", reasoning_effort="high")
    result = model("Reply with exactly: OK")

    assert result.consume() == "OK"
    assert result.reasoning == "Checking the request."
    assert result.history_items == [
        {
            "type": "reasoning",
            "role": "assistant",
            "text": "Checking the request.",
            "provider_state": {
                "provider": "anthropic",
                "api_mode": "anthropic_messages",
                "codec": "anthropic_thinking",
                "data": {
                    "type": "thinking",
                    "thinking": "Checking the request.",
                    "signature": "sig-1",
                },
            },
        }
    ]

    body = client.request.call_args.kwargs["json"]
    assert client.request.call_args.args[:2] == (
        "POST",
        "https://api.anthropic.com/v1/messages",
    )
    assert client.request.call_args.kwargs["headers"] == {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        "x-api-key": "test-key",
    }
    assert body["model"] == "claude-opus-4-8"
    assert body["max_tokens"] == 4096
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}
    assert "cache_control" not in body


def test_anthropic_async_request_uses_shared_transport(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    _, aclient = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _native_text_response()
    aclient.request = AsyncMock(return_value=response)

    model = AnthropicChatCompletion(model_id="claude-opus-4-8")
    result = asyncio.run(model.acall("Reply with exactly: OK"))

    assert result.consume() == "OK"
    assert aclient.request.await_args.args[:2] == (
        "POST",
        "https://api.anthropic.com/v1/messages",
    )
    assert aclient.request.await_args.kwargs["headers"]["x-api-key"] == "test-key"


def test_anthropic_sync_stream_uses_shared_transport(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    client, _ = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    response.iter_lines.return_value = [
        "event: message_start",
        'data: {"message":{"usage":{"input_tokens":7}}}',
        "",
        "event: content_block_start",
        'data: {"index":0,"content_block":{"type":"text","text":""}}',
        "",
        "event: content_block_delta",
        'data: {"index":0,"delta":{"type":"text_delta","text":"OK"}}',
        "",
        "event: content_block_stop",
        'data: {"index":0}',
        "",
        "event: message_delta",
        'data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}',
        "",
    ]
    client.stream.return_value.__enter__.return_value = response

    model = AnthropicChatCompletion(model_id="claude-opus-4-8")
    chunks = list(
        model._execute_model(messages=[{"role": "user", "content": "hi"}], stream=True)
    )

    assert any(chunk.choices[0].delta.content == "OK" for chunk in chunks)
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert client.stream.call_args.args[:2] == (
        "POST",
        "https://api.anthropic.com/v1/messages",
    )
    assert client.stream.call_args.kwargs["headers"]["x-api-key"] == "test-key"


def test_anthropic_async_stream_uses_shared_transport(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    _, aclient = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200

    async def lines():
        for line in (
            "event: content_block_start",
            'data: {"index":0,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"index":0,"delta":{"type":"text_delta","text":"OK"}}',
            "",
            "event: content_block_stop",
            'data: {"index":0}',
            "",
            "event: message_delta",
            'data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}',
            "",
        ):
            yield line

    response.aiter_lines.return_value = lines()
    aclient.stream.return_value.__aenter__.return_value = response
    model = AnthropicChatCompletion(model_id="claude-opus-4-8")

    async def collect():
        return [
            chunk
            async for chunk in await model._aexecute_model(
                messages=[{"role": "user", "content": "hi"}], stream=True
            )
        ]

    chunks = asyncio.run(collect())

    assert any(chunk.choices[0].delta.content == "OK" for chunk in chunks)
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert aclient.stream.call_args.args[:2] == (
        "POST",
        "https://api.anthropic.com/v1/messages",
    )
    assert aclient.stream.call_args.kwargs["headers"]["x-api-key"] == "test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_anthropic_public_stream_dispatch(asynchronous):
    from msgflux.models.http_transport import HTTPTransport
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    events = [
        ("message_start", {"message": {"usage": {"input_tokens": 7}}}),
        (
            "content_block_start",
            {"index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "text_delta", "text": "OK"}},
        ),
        ("content_block_stop", {"index": 0}),
        (
            "message_delta",
            {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
        ),
        ("message_stop", {}),
    ]
    body = "".join(
        f"event: {kind}\ndata: {json.dumps(data)}\n\n" for kind, data in events
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler)
        ) as aclient:
            model = AnthropicChatCompletion(model_id="claude-haiku-5-5", retry=False)
            model.http_transport = HTTPTransport(
                client=client, async_client=aclient, max_retries=0
            )
            response = (
                await model.acall("Hello", stream=True)
                if asynchronous
                else model("Hello", stream=True)
            )

            async def consume():
                return [chunk async for chunk in response.consume()]

            assert await asyncio.wait_for(consume(), timeout=2) == ["OK"]
    assert str(requests[0].url) == "https://api.anthropic.com/v1/messages"
    assert json.loads(requests[0].content)["stream"] is True


def test_anthropic_redacted_thinking_round_trip(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    client, _ = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _native_text_response(
        content=[
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": "OK"},
        ]
    )
    client.request.return_value = response

    model = AnthropicChatCompletion(model_id="claude-opus-4-8")
    result = model("Reply with exactly: OK")

    assert result.consume() == "OK"
    assert result.reasoning is None
    assert result.history_items == [
        {
            "type": "reasoning",
            "role": "assistant",
            "provider_state": {
                "provider": "anthropic",
                "api_mode": "anthropic_messages",
                "codec": "anthropic_thinking",
                "data": {"type": "redacted_thinking", "data": "opaque"},
            },
        }
    ]


def test_anthropic_tool_use_replay(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    client, _ = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _native_text_response(
        content=[
            {
                "type": "thinking",
                "thinking": "Need the weather.",
                "signature": "sig-2",
            },
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ],
        stop_reason="tool_use",
    )
    client.request.return_value = response

    model = AnthropicChatCompletion(model_id="claude-opus-4-8")
    result = model("What is the weather in Paris?")

    assert result.response_type == "tool_call"

    messages = ChatMessages()
    messages.add_user("What is the weather in Paris?")
    for item in result.history_items:
        messages.append(item)
    intents = list(result.get_tool_intents())
    for intent in intents:
        messages.append(
            {
                "type": "function_call",
                "call_id": intent.id,
                "name": intent.name,
                "arguments": intent.arguments,
            }
        )
    messages.add_tool(intents[0].id, "sunny")

    params = model._build_generation_params(
        messages, system_prompt=None, prefilling=None, tool_catalog=None
    )
    body = model._to_anthropic_body({**model.sampling_run_params, **params})

    assert body["messages"] == [
        {"role": "user", "content": "What is the weather in Paris?"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "Need the weather.",
                    "signature": "sig-2",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "sunny",
                }
            ],
        },
    ]


def test_anthropic_effort_none_disables_thinking(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    model = AnthropicChatCompletion(model_id="claude-sonnet-5", reasoning_effort="none")

    assert model._thinking_request_config() == {"thinking": {"type": "disabled"}}


def test_anthropic_budget_mode_uses_manual_thinking(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    model = AnthropicChatCompletion(
        model_id="claude-sonnet-4-5", reasoning_max_tokens=4000
    )

    assert model._thinking_request_config() == {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 4000,
            "display": "summarized",
        }
    }


def test_anthropic_effort_and_budget_conflict(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    with pytest.raises(ValueError, match="cannot be used together"):
        AnthropicChatCompletion(
            model_id="claude-sonnet-4-5",
            reasoning_effort="high",
            reasoning_max_tokens=4000,
        )


def test_anthropic_prompt_cache_sends_top_level_breakpoint(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    model = AnthropicChatCompletion(model_id="claude-opus-4-8", prompt_cache=True)
    params = model._build_generation_params(
        "hi", system_prompt=None, prefilling=None, tool_catalog=None
    )
    body = model._to_anthropic_body({**model.sampling_run_params, **params})

    assert body["cache_control"] == {"type": "ephemeral"}


def test_anthropic_usage_aggregates_cache_counters(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    client, _ = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _native_text_response(
        usage={
            "input_tokens": 17,
            "output_tokens": 20,
            "cache_creation_input_tokens": 1370,
            "cache_read_input_tokens": 0,
        }
    )
    client.request.return_value = response

    model = AnthropicChatCompletion(model_id="claude-opus-4-8")
    result = model("hi")

    usage = result.metadata.usage
    assert usage.input_tokens == 1387
    assert usage.input_tokens_details.cache_write_tokens == 1370


def test_anthropic_structured_output_raises(mock_anthropic_clients):
    import msgspec

    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    class Answer(msgspec.Struct):
        text: str

    model = AnthropicChatCompletion(model_id="claude-opus-4-8")

    with pytest.raises(ValueError, match="generation_schema"):
        model("hi", generation_schema=Answer)


def test_anthropic_http_error_mapping(mock_anthropic_clients):
    from msgflux.exceptions import ModelProviderHTTPError
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    client, _ = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 402
    response.headers = {"request-id": "req_1"}
    response.json.return_value = {
        "type": "error",
        "error": {"type": "billing_error", "message": "No credits."},
    }
    response.raise_for_status.side_effect = httpx2.HTTPStatusError(
        "billing error", request=MagicMock(), response=response
    )
    client.request.return_value = response

    model = AnthropicChatCompletion(model_id="claude-opus-4-8")

    with pytest.raises(ModelProviderHTTPError, match="HTTP 402") as exc_info:
        model("hi")

    assert exc_info.value.error_type == "billing_error"
    assert exc_info.value.request_id == "req_1"


def test_anthropic_stream_thinking_and_text(mock_anthropic_clients):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    model = AnthropicChatCompletion(model_id="claude-opus-4-8")
    events = [
        ("message_start", {"message": {"usage": {"input_tokens": 10}}}),
        (
            "content_block_start",
            {"index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "thinking_delta", "thinking": "Check."}},
        ),
        (
            "content_block_delta",
            {"index": 0, "delta": {"type": "signature_delta", "signature": "sig-9"}},
        ),
        ("content_block_stop", {"index": 0}),
        (
            "content_block_start",
            {"index": 1, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"index": 1, "delta": {"type": "text_delta", "text": "OK"}},
        ),
        ("content_block_stop", {"index": 1}),
        (
            "message_delta",
            {
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 12},
            },
        ),
    ]

    chunks = list(model._expand_stream_events(iter(events)))
    kinds = [
        (
            chunk.choices[0].delta.get("thinking"),
            chunk.choices[0].delta.get("content"),
            bool(chunk.choices[0].delta.get("thinking_block")),
        )
        for chunk in chunks
    ]

    assert kinds[0] == ("Check.", None, False)
    assert kinds[1] == (None, None, True)
    assert kinds[2] == (None, "OK", False)
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage["input_tokens"] == 10
    assert chunks[-1].usage["output_tokens"] == 12

    state = chunks[1].choices[0].delta["thinking_block"]
    assert model.reasoning_codec.extract_state(
        {"thinking_block": state}, serialize=lambda value: value
    ) == {"type": "thinking", "thinking": "Check.", "signature": "sig-9"}


def test_anthropic_state_does_not_replay_for_other_providers(
    mock_anthropic_clients, monkeypatch
):
    import msgflux as mf

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    other = mf.Model.chat_completion("openai/gpt-5", api_mode="chat_completions")
    messages = ChatMessages(
        [
            {
                "type": "reasoning",
                "role": "assistant",
                "text": "summary",
                "provider_state": {
                    "provider": "anthropic",
                    "api_mode": "anthropic_messages",
                    "codec": "anthropic_thinking",
                    "data": {
                        "type": "thinking",
                        "thinking": "summary",
                        "signature": "sig-1",
                    },
                },
            },
            {"role": "assistant", "content": "answer"},
        ]
    )

    replay = messages.to_chatml(
        provider="openai",
        api_mode="chat_completions",
        reasoning_codec=other.reasoning_codec,
    )

    assert replay == [{"role": "assistant", "content": "answer"}]


def test_anthropic_api_key_env_override(mock_anthropic_clients, monkeypatch):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    monkeypatch.setenv("ANTHROPIC_ACME_KEY", "anthropic-acme-key")
    model = AnthropicChatCompletion(
        model_id="claude-opus-4-8", api_key_env="ANTHROPIC_ACME_KEY"
    )

    assert model._get_api_key() == "anthropic-acme-key"


def _search_catalog(*, choice=None, include_search=True):
    from msgflux.tools.catalog import ToolCatalogEntry, ToolCatalogView, ToolRef

    library_id = "warehouse_tools"
    entries = [
        ToolCatalogEntry(
            ref=ToolRef(library_id=library_id, tool_id="lookup_inventory"),
            description="Look up a SKU.",
            input_schema={
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Warehouse SKU.",
                    }
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
            deferred=True,
        ),
        ToolCatalogEntry(
            ref=ToolRef(library_id=library_id, tool_id="current_stock"),
            description="Read current stock levels.",
            input_schema={"type": "object", "properties": {}},
            deferred=True,
            loaded=True,
        ),
        ToolCatalogEntry(
            ref=ToolRef(library_id=library_id, tool_id="archive_lookup"),
            description="Search archived warehouse records.",
            input_schema={"type": "object", "properties": {}},
            deferred=True,
        ),
    ]
    if include_search:
        entries.append(
            ToolCatalogEntry(
                ref=ToolRef(library_id=library_id, tool_id="tool_search"),
                description="Search deferred tools.",
                input_schema={"type": "object", "properties": {}},
                catalog_role="search",
            )
        )
    return ToolCatalogView(
        library_id=library_id,
        thread_id="thread_search",
        entries=tuple(entries),
        choice=choice,
    )


def _native_search_response(*, content, stop_reason="tool_use"):
    return _native_text_response(content=content, stop_reason=stop_reason)


def _weather_search_blocks(*, include_tool_use=True):
    blocks = [
        {"type": "text", "text": "I will search for a weather tool."},
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "tool_search_tool_bm25",
            "input": {"query": "weather", "limit": 3},
        },
        {
            "type": "tool_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": {
                "type": "tool_search_tool_search_result",
                "tool_references": [
                    {"type": "tool_reference", "tool_name": "get_weather"}
                ],
            },
        },
        {"type": "text", "text": "I found a weather tool."},
    ]
    if include_tool_use:
        blocks.append(
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            }
        )
    return blocks


def test_anthropic_native_tool_search_compiles_all_tools(
    mock_anthropic_clients,
):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5")
    catalog = _search_catalog()

    params = model._build_generation_params(
        "Check SKU-1842",
        system_prompt=None,
        prefilling=None,
        tool_catalog=catalog,
    )
    body = model._to_anthropic_body({**model.sampling_run_params, **params})

    assert body["tools"][0] == {
        "type": "tool_search_tool_bm25_20251119",
        "name": "tool_search_tool_bm25",
    }
    tools = {tool["name"]: tool for tool in body["tools"][1:]}
    assert set(tools) == {"lookup_inventory", "current_stock", "archive_lookup"}
    assert tools["lookup_inventory"]["defer_loading"] is True
    assert tools["archive_lookup"]["defer_loading"] is True
    assert tools["current_stock"]["defer_loading"] is True
    assert tools["lookup_inventory"]["input_schema"] == {
        "type": "object",
        "properties": {
            "sku": {
                "type": "string",
                "description": "Warehouse SKU.",
            }
        },
        "required": ["sku"],
        "additionalProperties": False,
    }
    assert body["tool_choice"] == {"type": "auto"}


def test_anthropic_native_tool_search_keeps_loaded_deferred_but_exposes_selected(
    mock_anthropic_clients,
):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion
    from msgflux.tools.catalog import ToolChoice

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5")
    catalog = _search_catalog(choice=ToolChoice(mode="tool", name="lookup_inventory"))

    params = model._build_generation_params(
        "Check SKU-1842",
        system_prompt=None,
        prefilling=None,
        tool_catalog=catalog,
    )
    body = model._to_anthropic_body({**model.sampling_run_params, **params})
    tools = {tool["name"]: tool for tool in body["tools"][1:]}

    assert "defer_loading" not in tools["lookup_inventory"]
    assert tools["current_stock"]["defer_loading"] is True
    assert body["tool_choice"] == {"type": "tool", "name": "lookup_inventory"}


@pytest.mark.parametrize(
    "native_tools,override,expected",
    [
        (True, None, True),
        (True, True, True),
        (True, False, False),
        (False, True, False),
    ],
)
def test_anthropic_native_search_respects_explicit_model_capability(
    native_tools, override, expected
):
    from msgflux.models.chat_capabilities import ChatModelCapabilities
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    model = AnthropicChatCompletion(
        model_id="claude-haiku-5-5",
        native_tools=native_tools,
        model_capabilities=ChatModelCapabilities(hosted_tool_search=override),
    )
    params = model._build_generation_params("Find stock", None, None, _search_catalog())
    body = model._to_anthropic_body({**model.sampling_run_params, **params})
    assert model.supports_native_tool_search() is expected
    assert (
        any(
            tool.get("type") == "tool_search_tool_bm25_20251119"
            for tool in body["tools"]
        )
        is expected
    )


def test_anthropic_native_tools_false_uses_portable_search(
    mock_anthropic_clients,
):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5", native_tools=False)
    catalog = _search_catalog()

    params = model._build_generation_params(
        "Check SKU-1842",
        system_prompt=None,
        prefilling=None,
        tool_catalog=catalog,
    )
    body = model._to_anthropic_body({**model.sampling_run_params, **params})

    assert [tool["name"] for tool in body["tools"]] == [
        "current_stock",
        "tool_search",
    ]
    assert not any(tool.get("defer_loading") for tool in body["tools"])


def test_anthropic_native_tool_search_preserves_blocks_and_replays(
    mock_anthropic_clients,
):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    client, _ = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _native_search_response(
        content=_weather_search_blocks()
    )
    client.request.return_value = response

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5")
    result = model("What is the weather in Paris?")

    assert result.response_type == "tool_call"
    intents = list(result.get_tool_intents())
    assert [(intent.id, intent.name) for intent in intents] == [
        ("toolu_1", "get_weather")
    ]

    search_items = [
        item
        for item in result.history_items
        if item["type"] in {"tool_search_call", "tool_search_output"}
    ]
    function_calls = [
        item for item in result.history_items if item["type"] == "function_call"
    ]
    assert [item["type"] for item in search_items] == [
        "tool_search_call",
        "tool_search_output",
    ]
    assert search_items[0]["provider_state"]["data"] == {
        "type": "server_tool_use",
        "id": "srvtoolu_1",
        "name": "tool_search_tool_bm25",
        "input": {"query": "weather", "limit": 3},
    }
    assert search_items[1]["provider_state"]["data"]["content"] == {
        "type": "tool_search_tool_search_result",
        "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
    }
    assert [item["call_id"] for item in function_calls] == ["toolu_1"]
    assert (
        function_calls[0]["provider_state"]["data"]["anthropic_content_blocks"]
        == _weather_search_blocks()[:-1]
    )

    messages = ChatMessages()
    messages.add_user("What is the weather in Paris?")
    for item in result.history_items:
        messages.append(item)
    messages.add_tool("toolu_1", "sunny")

    params = model._build_generation_params(
        messages, system_prompt=None, prefilling=None, tool_catalog=None
    )
    body = model._to_anthropic_body({**model.sampling_run_params, **params})

    assert body["messages"][1] == {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I will search for a weather tool."},
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "tool_search_tool_bm25",
                "input": {"query": "weather", "limit": 3},
            },
            {
                "type": "tool_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": {
                    "type": "tool_search_tool_search_result",
                    "tool_references": [
                        {"type": "tool_reference", "tool_name": "get_weather"}
                    ],
                },
            },
            {"type": "text", "text": "I found a weather tool."},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
            },
        ],
    }
    assert body["messages"][2] == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny"}
        ],
    }
    assert "srvtoolu_1" not in json.dumps(body["messages"][2])


def test_anthropic_native_tool_search_text_turn_replays_once(
    mock_anthropic_clients,
):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    client, _ = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _native_search_response(
        content=_weather_search_blocks(include_tool_use=False),
        stop_reason="end_turn",
    )
    client.request.return_value = response

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5")
    result = model("Find a weather tool")

    assert result.response_type == "text_generation"
    message_items = [item for item in result.history_items if item["type"] == "message"]
    assert len(message_items) == 1
    assert message_items[0]["content"] == (
        "I will search for a weather tool.I found a weather tool."
    )
    assert message_items[0]["provider_state"]["data"][
        "anthropic_content_blocks"
    ] == _weather_search_blocks(include_tool_use=False)

    messages = ChatMessages()
    messages.add_user("Find a weather tool")
    for item in result.history_items:
        messages.append(item)

    params = model._build_generation_params(
        messages, system_prompt=None, prefilling=None, tool_catalog=None
    )
    body = model._to_anthropic_body({**model.sampling_run_params, **params})

    assert body["messages"][1]["content"] == _weather_search_blocks(
        include_tool_use=False
    )


def test_anthropic_stream_preserves_server_tool_blocks(
    mock_anthropic_clients,
):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion
    from msgflux.models.response import ModelStreamResponse
    from msgflux.models.timing import ModelRequestTimer

    client, _ = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    result_block = {
        "type": "tool_search_tool_result",
        "tool_use_id": "srvtoolu_1",
        "content": {
            "type": "tool_search_tool_search_result",
            "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
        },
    }
    response.iter_lines.return_value = [
        "event: message_start",
        'data: {"message":{"usage":{"input_tokens":7}}}',
        "",
        "event: content_block_start",
        'data: {"index":0,"content_block":{"type":"text","text":""}}',
        "",
        "event: content_block_delta",
        'data: {"index":0,"delta":{"type":"text_delta","text":"Searching."}}',
        "",
        "event: content_block_stop",
        'data: {"index":0}',
        "",
        "event: content_block_start",
        'data: {"index":1,"content_block":{"type":"server_tool_use",'
        '"id":"srvtoolu_1","name":"tool_search_tool_bm25","input":{}}}',
        "",
        "event: content_block_delta",
        'data: {"index":1,"delta":{"type":"input_json_delta",'
        '"partial_json":"{\\"query\\":\\"weather\\"}"}}',
        "",
        "event: content_block_stop",
        'data: {"index":1}',
        "",
        "event: content_block_start",
        "data: " + json.dumps({"index": 2, "content_block": result_block}),
        "",
        "event: content_block_stop",
        'data: {"index":2}',
        "",
        "event: content_block_start",
        'data: {"index":3,"content_block":{"type":"tool_use",'
        '"id":"toolu_1","name":"get_weather","input":{}}}',
        "",
        "event: content_block_delta",
        'data: {"index":3,"delta":{"type":"input_json_delta",'
        '"partial_json":"{\\"city\\":\\"Paris\\"}"}}',
        "",
        "event: content_block_stop",
        'data: {"index":3}',
        "",
        "event: message_delta",
        'data: {"delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}',
        "",
    ]
    client.stream.return_value.__enter__.return_value = response

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5")
    stream_response = ModelStreamResponse()
    model._stream_generate(
        messages=[{"role": "user", "content": "Weather?"}],
        stream=True,
        stream_response=stream_response,
        _request_timer=ModelRequestTimer(),
    )

    items = stream_response.chat_accumulator.snapshot()
    assert [item["type"] for item in items] == [
        "tool_search_call",
        "tool_search_output",
        "function_call",
    ]
    assert items[0]["provider_state"]["data"]["id"] == "srvtoolu_1"
    assert items[1]["provider_state"]["data"]["content"] == result_block["content"]
    assert items[2]["call_id"] == "toolu_1"
    assert items[2]["provider_state"]["data"]["anthropic_content_blocks"] == [
        {"type": "text", "text": "Searching."},
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "tool_search_tool_bm25",
            "input": {"query": "weather"},
        },
        result_block,
    ]

    messages = ChatMessages()
    messages.add_user("Weather?")
    for item in items:
        messages.append(item)
    messages.add_tool("toolu_1", "sunny")
    params = model._build_generation_params(
        messages, system_prompt=None, prefilling=None, tool_catalog=None
    )
    body = model._to_anthropic_body({**model.sampling_run_params, **params})
    assert body["messages"][1]["content"][0] == {
        "type": "text",
        "text": "Searching.",
    }
    assert body["messages"][1]["content"][2] == result_block
    assert body["messages"][1]["content"][-1] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "get_weather",
        "input": {"city": "Paris"},
    }


def test_anthropic_native_blocks_do_not_replay_for_other_providers(
    mock_anthropic_clients,
):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion

    client, _ = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = _native_search_response(
        content=_weather_search_blocks(include_tool_use=False),
        stop_reason="end_turn",
    )
    client.request.return_value = response

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5")
    result = model("Find a weather tool")
    messages = ChatMessages()
    messages.add_user("Find a weather tool")
    for item in result.history_items:
        messages.append(item)

    replay = messages.to_chatml(provider="openai", api_mode="chat_completions")

    assert replay == [
        {"role": "user", "content": "Find a weather tool"},
        {
            "role": "assistant",
            "content": "I will search for a weather tool.I found a weather tool.",
        },
    ]


def test_anthropic_stream_empty_tool_input_still_emits_call(
    mock_anthropic_clients,
):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion
    from msgflux.models.response import ModelStreamResponse
    from msgflux.models.timing import ModelRequestTimer

    client, _ = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    response.iter_lines.return_value = [
        "event: content_block_start",
        'data: {"index":0,"content_block":{"type":"tool_use",'
        '"id":"toolu_empty","name":"current_stock","input":{}}}',
        "",
        "event: content_block_stop",
        'data: {"index":0}',
        "",
        "event: message_delta",
        'data: {"delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}',
        "",
    ]
    client.stream.return_value.__enter__.return_value = response

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5")
    stream_response = ModelStreamResponse()
    model._stream_generate(
        messages=[{"role": "user", "content": "Stock?"}],
        stream=True,
        stream_response=stream_response,
        _request_timer=ModelRequestTimer(),
    )

    items = stream_response.chat_accumulator.snapshot()
    assert items == [
        {
            "type": "function_call",
            "call_id": "toolu_empty",
            "name": "current_stock",
            "arguments": "{}",
        }
    ]
    assert stream_response.response_type == "tool_call"


def test_anthropic_stream_search_text_turn_replays_once(
    mock_anthropic_clients,
):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion
    from msgflux.models.response import ModelStreamResponse
    from msgflux.models.timing import ModelRequestTimer

    client, _ = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    result_block = {
        "type": "tool_search_tool_result",
        "tool_use_id": "srvtoolu_text",
        "content": {
            "type": "tool_search_tool_search_result",
            "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
        },
    }
    response.iter_lines.return_value = [
        "event: content_block_start",
        'data: {"index":0,"content_block":{"type":"text","text":""}}',
        "",
        "event: content_block_delta",
        'data: {"index":0,"delta":{"type":"text_delta","text":"Searching."}}',
        "",
        "event: content_block_stop",
        'data: {"index":0}',
        "",
        "event: content_block_start",
        'data: {"index":1,"content_block":{"type":"server_tool_use",'
        '"id":"srvtoolu_text","name":"tool_search_tool_bm25","input":{}}}',
        "",
        "event: content_block_delta",
        'data: {"index":1,"delta":{"type":"input_json_delta",'
        '"partial_json":"{\\"query\\":\\"weather\\"}"}}',
        "",
        "event: content_block_stop",
        'data: {"index":1}',
        "",
        "event: content_block_start",
        "data: " + json.dumps({"index": 2, "content_block": result_block}),
        "",
        "event: content_block_stop",
        'data: {"index":2}',
        "",
        "event: content_block_start",
        'data: {"index":3,"content_block":{"type":"text","text":""}}',
        "",
        "event: content_block_delta",
        'data: {"index":3,"delta":{"type":"text_delta","text":"No tool needed."}}',
        "",
        "event: content_block_stop",
        'data: {"index":3}',
        "",
        "event: message_delta",
        'data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}',
        "",
    ]
    client.stream.return_value.__enter__.return_value = response

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5")
    stream_response = ModelStreamResponse()
    model._stream_generate(
        messages=[{"role": "user", "content": "Weather?"}],
        stream=True,
        stream_response=stream_response,
        _request_timer=ModelRequestTimer(),
    )

    items = stream_response.chat_accumulator.snapshot()
    assert [item["type"] for item in items] == [
        "tool_search_call",
        "tool_search_output",
        "message",
    ]
    assert items[2]["provider_state"]["data"]["anthropic_content_blocks"] == [
        {"type": "text", "text": "Searching."},
        {
            "type": "server_tool_use",
            "id": "srvtoolu_text",
            "name": "tool_search_tool_bm25",
            "input": {"query": "weather"},
        },
        result_block,
        {"type": "text", "text": "No tool needed."},
    ]

    messages = ChatMessages()
    messages.add_user("Weather?")
    for item in items:
        messages.append(item)
    params = model._build_generation_params(
        messages, system_prompt=None, prefilling=None, tool_catalog=None
    )
    body = model._to_anthropic_body({**model.sampling_run_params, **params})
    assert (
        body["messages"][1]["content"]
        == items[2]["provider_state"]["data"]["anthropic_content_blocks"]
    )


def test_anthropic_async_stream_preserves_server_tool_blocks(
    mock_anthropic_clients,
):
    from msgflux.models.providers.anthropic import AnthropicChatCompletion
    from msgflux.models.response import ModelStreamResponse
    from msgflux.models.timing import ModelRequestTimer

    _, aclient = mock_anthropic_clients
    response = MagicMock()
    response.status_code = 200
    result_block = {
        "type": "tool_search_tool_result",
        "tool_use_id": "srvtoolu_async",
        "content": {
            "type": "tool_search_tool_search_result",
            "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
        },
    }
    lines = [
        "event: content_block_start",
        'data: {"index":0,"content_block":{"type":"server_tool_use",'
        '"id":"srvtoolu_async","name":"tool_search_tool_bm25","input":{}}}',
        "",
        "event: content_block_delta",
        'data: {"index":0,"delta":{"type":"input_json_delta",'
        '"partial_json":"{\\"query\\":\\"weather\\"}"}}',
        "",
        "event: content_block_stop",
        'data: {"index":0}',
        "",
        "event: content_block_start",
        "data: " + json.dumps({"index": 1, "content_block": result_block}),
        "",
        "event: content_block_stop",
        'data: {"index":1}',
        "",
        "event: message_delta",
        'data: {"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}',
        "",
    ]

    async def event_lines():
        for line in lines:
            yield line

    response.aiter_lines.return_value = event_lines()
    aclient.stream.return_value.__aenter__.return_value = response

    model = AnthropicChatCompletion(model_id="claude-haiku-5-5")
    stream_response = ModelStreamResponse(mode="async")

    async def run():
        await model._astream_generate(
            messages=[{"role": "user", "content": "Weather?"}],
            stream=True,
            stream_response=stream_response,
            _request_timer=ModelRequestTimer(),
        )

    asyncio.run(run())

    items = stream_response.chat_accumulator.snapshot()
    assert [item["type"] for item in items] == [
        "tool_search_call",
        "tool_search_output",
        "message",
    ]
    assert items[0]["provider_state"]["data"]["id"] == "srvtoolu_async"
    assert items[1]["provider_state"]["data"]["content"] == result_block["content"]
    assert items[2]["provider_state"]["data"]["anthropic_content_blocks"] == [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_async",
            "name": "tool_search_tool_bm25",
            "input": {"query": "weather"},
        },
        result_block,
    ]
