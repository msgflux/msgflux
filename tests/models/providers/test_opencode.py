"""Tests for the OpenCode gateway providers (Go and Zen)."""

import json
from dataclasses import dataclass

import httpx2
import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.models.chat_transport import HTTPChatTransport
from msgflux.models.http_transport import HTTPTransport
from msgflux.models.providers.anthropic import AnthropicChatCompletion
from msgflux.runtime.context import thread_context
from msgflux.tools import ToolCatalog, ToolOutcome

API_MODES = ("chat_completions", "responses", "anthropic_messages")


@dataclass(frozen=True)
class _ProviderCase:
    provider: str
    display_name: str
    api_key_env: str
    base_url_env: str
    base_url: str
    anthropic_base: str
    facade_name: str
    backend_name: str


GO = _ProviderCase(
    provider="opencode-go",
    display_name="OpenCode Go",
    api_key_env="OPENCODE_GO_API_KEY",
    base_url_env="OPENCODE_GO_BASE_URL",
    base_url="https://opencode.ai/inference/go/openai/v1",
    anthropic_base="https://opencode.ai/inference/go/anthropic",
    facade_name="OpenCodeGoChatCompletion",
    backend_name="_OpenCodeGoAnthropicMessages",
)
ZEN = _ProviderCase(
    provider="opencode-zen",
    display_name="OpenCode Zen",
    api_key_env="OPENCODE_ZEN_API_KEY",
    base_url_env="OPENCODE_ZEN_BASE_URL",
    base_url="https://opencode.ai/inference/openai/v1",
    anthropic_base="https://opencode.ai/inference/anthropic",
    facade_name="OpenCodeZenChatCompletion",
    backend_name="_OpenCodeZenAnthropicMessages",
)
PROVIDER_CASES = [pytest.param(GO, id=GO.provider), pytest.param(ZEN, id=ZEN.provider)]


def _other_provider(case):
    return ZEN if case.provider == GO.provider else GO


def _classes(case):
    from msgflux.models.providers import opencode

    return getattr(opencode, case.facade_name), getattr(opencode, case.backend_name)


@pytest.fixture(autouse=True)
def opencode_env(monkeypatch):
    """Keep every provider's environment isolated before each test."""
    for env_name in (
        GO.api_key_env,
        ZEN.api_key_env,
        GO.base_url_env,
        ZEN.base_url_env,
        "OPENCODE_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(env_name, raising=False)


@pytest.fixture(params=PROVIDER_CASES)
def provider(request, monkeypatch):
    case = request.param
    monkeypatch.setenv(case.api_key_env, "test-key")
    return case


def _model(provider, *, client=None, async_client=None, **kwargs):
    facade, _ = _classes(provider)
    return facade(
        model_id="deepseek-v4.1-flash",
        chat_transport=HTTPChatTransport(
            client=client,
            async_client=async_client,
            max_retries=0,
        ),
        **kwargs,
    )


def _native_model(provider, *, client=None, async_client=None, **kwargs):
    facade, _ = _classes(provider)
    model = facade(
        model_id="anthropic/claude-sonnet-4-6",
        api_mode="anthropic_messages",
        retry=False,
        **kwargs,
    )
    model.http_transport = HTTPTransport(
        client=client,
        async_client=async_client,
        max_retries=0,
    )
    return model


def _completion(*, content="OK", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl_opencode",
        "object": "chat.completion",
        "created": 0,
        "model": "deepseek-v4.1-flash",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def _responses_completion(text="OK", *, output=None):
    return {
        "id": "resp_opencode",
        "object": "response",
        "status": "completed",
        "model": "deepseek-v4.1-flash",
        "output": output
        or [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }


def _anthropic_completion(
    *,
    content=None,
    stop_reason="end_turn",
    usage=None,
    model="anthropic/claude-sonnet-4-6",
):
    return {
        "id": "msg_opencode",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content or [{"type": "text", "text": "OK"}],
        "stop_reason": stop_reason,
        "usage": usage or {"input_tokens": 3, "output_tokens": 2},
    }


def _sse(*events):
    payloads = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return payloads + "data: [DONE]\n\n"


def _stream_text(text):
    return _sse(
        {
            "id": "chatcmpl_opencode_stream",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "deepseek-v4.1-flash",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": text},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl_opencode_stream",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "deepseek-v4.1-flash",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    )


def _responses_stream(text):
    return _sse(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_stream",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        },
        {
            "type": "response.output_text.delta",
            "output_index": 0,
            "delta": text,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": "msg_stream",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_stream",
                "status": "completed",
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "total_tokens": 5,
                },
            },
        },
    )


def _anthropic_sse(events):
    return "".join(
        f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events
    )


def _anthropic_text_stream(text):
    return _anthropic_sse(
        [
            ("message_start", {"message": {"usage": {"input_tokens": 3}}}),
            (
                "content_block_start",
                {"index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"index": 0, "delta": {"type": "text_delta", "text": text}},
            ),
            ("content_block_stop", {"index": 0}),
            (
                "message_delta",
                {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}},
            ),
            ("message_stop", {}),
        ]
    )


TOOL_STREAM_SIGNATURE = "sig-opencode-stream"

TOOL_STREAM_EVENTS = [
    ("message_start", {"message": {"usage": {"input_tokens": 4}}}),
    (
        "content_block_start",
        {"index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    ),
    (
        "content_block_delta",
        {
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Need the order."},
        },
    ),
    (
        "content_block_delta",
        {
            "index": 0,
            "delta": {
                "type": "signature_delta",
                "signature": TOOL_STREAM_SIGNATURE,
            },
        },
    ),
    ("content_block_stop", {"index": 0}),
    (
        "content_block_start",
        {
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_stream_1",
                "name": "submit_order",
                "input": {},
            },
        },
    ),
    (
        "content_block_delta",
        {
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"quantity":'},
        },
    ),
    (
        "content_block_delta",
        {
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "2}"},
        },
    ),
    ("content_block_stop", {"index": 1}),
    (
        "message_delta",
        {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 8}},
    ),
    ("message_stop", {}),
]


def _tool_catalog():
    return ToolCatalog.from_function_schemas(
        schemas=[
            {
                "type": "function",
                "function": {
                    "name": "submit_order",
                    "description": "Submit a synthetic order.",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"quantity": {"type": "integer"}},
                        "required": ["quantity"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        choice="submit_order",
    )


def _request_body(request):
    return json.loads(request.content)


def test_defaults_and_factory_registration(provider):
    import msgflux as mf
    from msgflux.accounts.providers import get_api_key_env

    facade, backend = _classes(provider)

    model = mf.Model.chat_completion(f"{provider.provider}/deepseek-v4.1-flash")

    assert isinstance(model, facade)
    assert model.provider == provider.provider
    assert model.display_name == provider.display_name
    assert model.model_id == "deepseek-v4.1-flash"
    assert model.api_mode == "chat_completions"
    assert model.supported_api_modes == (
        "chat_completions",
        "responses",
        "anthropic_messages",
    )
    assert isinstance(model.chat_transport, HTTPChatTransport)
    assert model._get_base_url() == provider.base_url
    assert model._get_api_key() == "test-key"
    assert model.api_key_env == provider.api_key_env
    assert get_api_key_env(provider.provider) == provider.api_key_env

    responses = mf.Model.chat_completion(
        f"{provider.provider}/deepseek-v4.1-flash", api_mode="responses"
    )
    assert isinstance(responses, facade)
    assert responses.api_mode == "responses"
    assert responses._get_base_url() == model._get_base_url()

    native = mf.Model.chat_completion(
        f"{provider.provider}/anthropic/claude-sonnet-4-6",
        api_mode="anthropic_messages",
    )
    assert isinstance(native, backend)
    assert isinstance(native, AnthropicChatCompletion)
    assert not isinstance(native, facade)
    assert native.provider == provider.provider
    assert native.api_mode == "anthropic_messages"
    assert native.supported_api_modes == ("anthropic_messages",)
    assert native.api_key_env == provider.api_key_env
    assert native._native_url() == f"{provider.anthropic_base}/v1/messages"
    assert native.credential_resolver.resolve(native).headers == {
        "Authorization": "Bearer test-key"
    }

    with pytest.raises(
        ValueError,
        match=("supported modes: chat_completions, responses, anthropic_messages"),
    ):
        facade(model_id="deepseek-v4.1-flash", api_mode="interactions")


def test_registry_and_accounts_expose_both_providers_without_private_classes():
    from msgflux.accounts.providers import get_api_key_env
    from msgflux.models.providers import opencode
    from msgflux.models.registry import model_registry

    registry = model_registry["chat_completion"]

    assert registry[GO.provider] is opencode.OpenCodeGoChatCompletion
    assert registry[ZEN.provider] is opencode.OpenCodeZenChatCompletion
    assert (
        opencode.OpenCodeGoChatCompletion._anthropic_messages_backend
        is opencode._OpenCodeGoAnthropicMessages
    )
    assert (
        opencode.OpenCodeZenChatCompletion._anthropic_messages_backend
        is opencode._OpenCodeZenAnthropicMessages
    )

    private_classes = {
        opencode._BaseOpenCode,
        opencode._OpenCodeSessions,
        opencode._OpenCodeAnthropicMessages,
        opencode._OpenCodeChatCompletion,
        opencode._OpenCodeGoConfig,
        opencode._OpenCodeZenConfig,
        opencode._OpenCodeGoAnthropicMessages,
        opencode._OpenCodeZenAnthropicMessages,
    }
    assert private_classes.isdisjoint(registry.values())

    assert get_api_key_env(GO.provider) == GO.api_key_env
    assert get_api_key_env(ZEN.provider) == ZEN.api_key_env


def test_base_url_override_and_missing_key(provider, monkeypatch):
    facade, _ = _classes(provider)

    monkeypatch.setenv(provider.base_url_env, "https://gateway.example/openai/v1")
    chat = _model(provider)
    native = _native_model(provider)

    assert chat._get_base_url() == "https://gateway.example/openai/v1"
    assert native._get_base_url() == "https://gateway.example/anthropic"
    assert native._native_url() == ("https://gateway.example/anthropic/v1/messages")

    monkeypatch.setenv(provider.base_url_env, "https://gateway.example/v1")
    native = _native_model(provider)
    assert native._get_base_url() == "https://gateway.example"
    assert native._native_url() == "https://gateway.example/v1/messages"

    monkeypatch.delenv(provider.base_url_env)
    monkeypatch.delenv(provider.api_key_env)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-decoy")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-decoy")

    with pytest.raises(ValueError, match=provider.api_key_env):
        facade(model_id="deepseek-v4.1-flash")
    with pytest.raises(ValueError, match=provider.api_key_env):
        facade(
            model_id="anthropic/claude-sonnet-4-6",
            api_mode="anthropic_messages",
        )


@pytest.mark.parametrize("api_mode", API_MODES)
def test_missing_key_does_not_fall_back_to_generic_or_other_provider_key(
    provider, monkeypatch, api_mode
):
    import msgflux as mf

    other = _other_provider(provider)
    monkeypatch.delenv(provider.api_key_env)
    monkeypatch.setenv("OPENCODE_API_KEY", "generic-plan-key")
    monkeypatch.setenv(other.api_key_env, "other-provider-key")

    with pytest.raises(ValueError, match=provider.api_key_env):
        mf.Model.chat_completion(
            f"{provider.provider}/deepseek-v4.1-flash", api_mode=api_mode
        )


def test_base_url_override_applies_to_actual_requests(provider, monkeypatch):
    monkeypatch.setenv(provider.base_url_env, "https://gateway.example/openai/v1")
    captured = []

    def handler(request):
        captured.append(request)
        if str(request.url).endswith("/chat/completions"):
            return httpx2.Response(200, json=_completion())
        if str(request.url).endswith("/responses"):
            return httpx2.Response(200, json=_responses_completion())
        return httpx2.Response(200, json=_anthropic_completion())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    chat = _model(provider, client=client)
    responses = _model(provider, client=client, api_mode="responses")
    native = _native_model(provider, client=client)
    try:
        chat("chat")
        responses("responses")
        native("anthropic")
    finally:
        client.close()

    assert [str(request.url) for request in captured] == [
        "https://gateway.example/openai/v1/chat/completions",
        "https://gateway.example/openai/v1/responses",
        "https://gateway.example/anthropic/v1/messages",
    ]


def test_sync_completion_uses_official_endpoint_and_headers(provider):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_completion())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    model = _model(provider, client=client)
    try:
        response = model("Reply with exactly: OK")
    finally:
        client.close()

    assert response.consume() == "OK"
    request = captured[0]
    assert request.method == "POST"
    assert str(request.url) == f"{provider.base_url}/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["user-agent"] == "msgflux"
    assert request.headers["x-opencode-session"] == model._fallback_session_id
    body = _request_body(request)
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Reply with exactly: OK"}]


@pytest.mark.asyncio
async def test_async_completion_uses_official_endpoint_and_headers(provider):
    captured = []

    async def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_completion(content="ASYNC"))

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    model = _model(provider, async_client=client)
    try:
        response = await model.acall("Reply with exactly: ASYNC")
    finally:
        await client.aclose()

    assert response.consume() == "ASYNC"
    request = captured[0]
    assert str(request.url) == f"{provider.base_url}/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["user-agent"] == "msgflux"
    assert request.headers["x-opencode-session"] == model._fallback_session_id
    assert _request_body(request)["model"] == "deepseek-v4.1-flash"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_stream_completion_uses_session_and_streaming_payload(
    provider, asynchronous
):
    captured = []
    body = _stream_text("streamed")

    def sync_handler(request):
        captured.append(request)
        return httpx2.Response(200, text=body)

    async def async_handler(request):
        captured.append(request)
        return httpx2.Response(200, text=body)

    if asynchronous:
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(async_handler))
        model = _model(provider, async_client=client)
        try:
            stream = await model.acall("Hello", stream=True)
            chunks = [chunk async for chunk in stream.consume()]
        finally:
            await client.aclose()
    else:
        client = httpx2.Client(transport=httpx2.MockTransport(sync_handler))
        model = _model(provider, client=client)
        try:
            stream = model("Hello", stream=True)
            chunks = [chunk async for chunk in stream.consume()]
        finally:
            client.close()

    assert chunks == ["streamed"]
    request = captured[0]
    assert str(request.url) == f"{provider.base_url}/chat/completions"
    assert request.headers["user-agent"] == "msgflux"
    assert request.headers["x-opencode-session"] == model._fallback_session_id
    assert _request_body(request)["stream"] is True


def test_tool_catalog_is_sent_and_tool_call_is_decoded(provider):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(
            200,
            json=_completion(
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_inventory",
                            "arguments": '{"sku":"1842"}',
                        },
                    }
                ],
            ),
        )

    catalog = ToolCatalog.from_function_schemas(
        schemas=[
            {
                "type": "function",
                "function": {
                    "name": "lookup_inventory",
                    "description": "Look up a SKU.",
                    "parameters": {
                        "type": "object",
                        "properties": {"sku": {"type": "string"}},
                        "required": ["sku"],
                    },
                },
            }
        ],
        choice="auto",
    )
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    model = _model(provider, client=client)
    try:
        response = model("Check SKU-1842", tool_catalog=catalog)
    finally:
        client.close()

    assert response.consume().get_calls() == [
        ("call_1", "lookup_inventory", {"sku": "1842"})
    ]
    body = _request_body(captured[0])
    assert body["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_inventory",
                "description": "Look up a SKU.",
                "parameters": {
                    "type": "object",
                    "properties": {"sku": {"type": "string"}},
                    "required": ["sku"],
                },
            },
        }
    ]
    assert body["tool_choice"] == "auto"
    assert "tool_catalog" not in body


def test_responses_completion_uses_official_endpoint_and_headers(provider):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_responses_completion("RESP"))

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    model = _model(provider, client=client, api_mode="responses")
    try:
        response = model("Reply with exactly: RESP", system_prompt="Be concise.")
    finally:
        client.close()

    assert response.consume() == "RESP"
    assert response.metadata.model["provider"] == provider.provider
    assert response.metadata.model["api_mode"] == "responses"
    request = captured[0]
    assert str(request.url) == f"{provider.base_url}/responses"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["user-agent"] == "msgflux"
    assert request.headers["x-opencode-session"] == model._fallback_session_id
    assert _request_body(request)["input"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Reply with exactly: RESP"},
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_responses_all_execution_modes_keep_session_and_identity(
    provider, asynchronous, stream
):
    captured = []

    def sync_handler(request):
        captured.append(request)
        if _request_body(request).get("stream"):
            return httpx2.Response(200, text=_responses_stream("streamed"))
        return httpx2.Response(200, json=_responses_completion("RESP"))

    async def async_handler(request):
        captured.append(request)
        if _request_body(request).get("stream"):
            return httpx2.Response(200, text=_responses_stream("streamed"))
        return httpx2.Response(200, json=_responses_completion("RESP"))

    client = httpx2.Client(transport=httpx2.MockTransport(sync_handler))
    aclient = httpx2.AsyncClient(transport=httpx2.MockTransport(async_handler))
    model = _model(provider, client=client, async_client=aclient, api_mode="responses")
    try:
        with thread_context(thread_id="responses-thread"):
            if asynchronous:
                response = await model.acall("Hello", stream=stream)
            else:
                response = model("Hello", stream=stream)
            if stream:
                chunks = [chunk async for chunk in response.consume()]
                assert chunks == ["streamed"]
            else:
                assert response.consume() == "RESP"
    finally:
        client.close()
        await aclient.aclose()

    request = captured[0]
    assert str(request.url) == f"{provider.base_url}/responses"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["user-agent"] == "msgflux"
    assert request.headers["x-opencode-session"] == "responses-thread"


def test_responses_session_precedence_and_case_insensitive_override(provider):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_responses_completion())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    model = _model(provider, client=client, api_mode="responses")
    messages = ChatMessages(
        [{"role": "user", "content": "Continue."}], thread_id="messages-thread"
    )
    try:
        with thread_context(thread_id="active-thread"):
            model(messages)
        with thread_context(thread_id=""):
            model(messages)
        model("Standalone")
        model.sampling_run_params["extra_headers"] = {
            "X-OpenCode-Session": "explicit-session",
            "user-agent": "custom-client/1.0",
        }
        with thread_context(thread_id="active-thread"):
            model(messages)
    finally:
        client.close()

    assert captured[0].headers["x-opencode-session"] == "active-thread"
    assert captured[1].headers["x-opencode-session"] == "messages-thread"
    assert captured[2].headers["x-opencode-session"] == model._fallback_session_id
    assert captured[3].headers["x-opencode-session"] == "explicit-session"
    assert captured[3].headers["user-agent"] == "custom-client/1.0"


def test_anthropic_completion_uses_gateway_bearer_and_session(provider, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-decoy")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-decoy")
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_anthropic_completion())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    model = _native_model(provider, client=client)
    try:
        response = model("Reply with exactly: OK", system_prompt="Be concise.")
    finally:
        client.close()

    assert response.consume() == "OK"
    request = captured[0]
    assert str(request.url) == f"{provider.anthropic_base}/v1/messages"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["user-agent"] == "msgflux"
    assert request.headers["x-opencode-session"] == model._fallback_session_id
    assert "x-api-key" not in request.headers
    assert "anthropic-decoy" not in request.headers.values()
    body = _request_body(request)
    assert body["model"] == "anthropic/claude-sonnet-4-6"
    assert body["messages"] == [{"role": "user", "content": "Reply with exactly: OK"}]
    assert body["system"] == "Be concise."


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_anthropic_all_execution_modes_keep_session_and_identity(
    provider, asynchronous, stream
):
    captured = []

    def sync_handler(request):
        captured.append(request)
        if _request_body(request).get("stream"):
            return httpx2.Response(200, text=_anthropic_text_stream("streamed"))
        return httpx2.Response(200, json=_anthropic_completion())

    async def async_handler(request):
        captured.append(request)
        if _request_body(request).get("stream"):
            return httpx2.Response(200, text=_anthropic_text_stream("streamed"))
        return httpx2.Response(200, json=_anthropic_completion())

    client = httpx2.Client(transport=httpx2.MockTransport(sync_handler))
    aclient = httpx2.AsyncClient(transport=httpx2.MockTransport(async_handler))
    model = _native_model(provider, client=client, async_client=aclient)
    try:
        with thread_context(thread_id="anthropic-thread"):
            if asynchronous:
                response = await model.acall("Hello", stream=stream)
            else:
                response = model("Hello", stream=stream)
            if stream:
                chunks = [chunk async for chunk in response.consume()]
                assert chunks == ["streamed"]
            else:
                assert response.consume() == "OK"
    finally:
        client.close()
        await aclient.aclose()

    request = captured[0]
    assert str(request.url) == f"{provider.anthropic_base}/v1/messages"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["user-agent"] == "msgflux"
    assert request.headers["x-opencode-session"] == "anthropic-thread"
    assert "x-api-key" not in request.headers


def test_anthropic_session_precedence_and_case_insensitive_override(provider):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_anthropic_completion())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    model = _native_model(provider, client=client)
    messages = ChatMessages(
        [{"role": "user", "content": "Continue."}], thread_id="messages-thread"
    )
    try:
        with thread_context(thread_id="active-thread"):
            model(messages)
        with thread_context(thread_id=""):
            model(messages)
        model("Standalone")
        model.sampling_run_params["extra_headers"] = {
            "X-OpenCode-Session": "explicit-session",
            "user-agent": "custom-client/1.0",
        }
        with thread_context(thread_id="active-thread"):
            model(messages)
    finally:
        client.close()

    assert captured[0].headers["x-opencode-session"] == "active-thread"
    assert captured[1].headers["x-opencode-session"] == "messages-thread"
    assert captured[2].headers["x-opencode-session"] == model._fallback_session_id
    assert captured[3].headers["x-opencode-session"] == "explicit-session"
    assert captured[3].headers["user-agent"] == "custom-client/1.0"


@pytest.mark.parametrize("inactive_thread", [None, ""])
def test_chat_runtime_thread_then_chat_messages_then_instance_fallback(
    provider, inactive_thread
):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_completion())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    model = _model(provider, client=client)
    messages = ChatMessages(
        [{"role": "user", "content": "Continue."}],
        thread_id="messages-thread",
    )
    try:
        with thread_context(thread_id="active-thread"):
            model(messages)
        with thread_context(thread_id=inactive_thread):
            model(messages)
        model("Standalone")
    finally:
        client.close()

    assert captured[0].headers["x-opencode-session"] == "active-thread"
    assert captured[1].headers["x-opencode-session"] == "messages-thread"
    assert captured[2].headers["x-opencode-session"] == model._fallback_session_id


def test_chat_explicit_headers_win_case_insensitively(provider):
    model = _model(provider)

    with thread_context(thread_id="active-thread"):
        params = model._adapt_params(
            {
                "model": "deepseek-v4.1-flash",
                "extra_headers": {
                    "X-OpenCode-Session": "explicit-session",
                    "user-agent": "custom-client/1.0",
                },
            }
        )

    headers = params["extra_headers"]
    assert headers == {
        "X-OpenCode-Session": "explicit-session",
        "user-agent": "custom-client/1.0",
    }


@pytest.mark.parametrize("active_thread", [None, "active-thread"])
def test_configured_headers_win_over_runtime_thread(provider, active_thread):
    model = _model(provider)
    model.sampling_run_params["extra_headers"] = {
        "X-OpenCode-Session": "explicit-session",
        "user-agent": "custom-client/1.0",
    }

    if active_thread is None:
        params = model._adapt_params({"model": "deepseek-v4.1-flash"})
    else:
        with thread_context(thread_id=active_thread):
            params = model._adapt_params({"model": "deepseek-v4.1-flash"})

    assert params["extra_headers"] == {
        "X-OpenCode-Session": "explicit-session",
        "user-agent": "custom-client/1.0",
    }


@pytest.mark.parametrize("explicit_session", [False, True])
def test_message_thread_preserves_configured_headers(provider, explicit_session):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_completion())

    headers = {"user-agent": "custom-client/1.0", "X-Other": "keep"}
    if explicit_session:
        headers["X-OpenCode-Session"] = "explicit-session"
    messages = ChatMessages(
        [{"role": "user", "content": "Continue."}], thread_id="messages-thread"
    )
    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model = _model(provider, client=client)
        model.sampling_run_params["extra_headers"] = headers
        model(messages)

    request = captured[0]
    assert request.headers["x-opencode-session"] == (
        "explicit-session" if explicit_session else "messages-thread"
    )
    assert request.headers["user-agent"] == "custom-client/1.0"
    assert request.headers["x-other"] == "keep"
    assert headers == {
        "user-agent": "custom-client/1.0",
        "X-Other": "keep",
        **({"X-OpenCode-Session": "explicit-session"} if explicit_session else {}),
    }


def test_fallback_session_is_stable_per_instance_and_isolated(provider):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_completion())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    first = _model(provider, client=client)
    second = _model(provider, client=client)
    try:
        first("one")
        first("two")
        second("three")
    finally:
        client.close()

    sessions = [request.headers["x-opencode-session"] for request in captured]
    assert sessions[0] == sessions[1] == first._fallback_session_id
    assert sessions[2] == second._fallback_session_id
    assert first._fallback_session_id != second._fallback_session_id


def test_api_key_rotates_at_request_time_and_is_not_serialized(provider, monkeypatch):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_completion())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    model = _model(provider, client=client)
    try:
        monkeypatch.setenv(provider.api_key_env, "first-rotated-key")
        model("first")
        monkeypatch.setenv(provider.api_key_env, "second-rotated-key")
        model("second")
    finally:
        client.close()

    assert captured[0].headers["authorization"] == "Bearer first-rotated-key"
    assert captured[1].headers["authorization"] == "Bearer second-rotated-key"
    serialized = str(model.serialize())
    assert "first-rotated-key" not in serialized
    assert "second-rotated-key" not in serialized


def test_responses_and_anthropic_rotate_keys_without_alternate_fallback(
    provider, monkeypatch
):
    other = _other_provider(provider)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-decoy")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-decoy")
    monkeypatch.setenv(other.api_key_env, "other-provider-decoy")
    captured = []

    def handler(request):
        captured.append(request)
        if "responses" in str(request.url):
            return httpx2.Response(200, json=_responses_completion())
        return httpx2.Response(200, json=_anthropic_completion())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    responses = _model(provider, client=client, api_mode="responses")
    native = _native_model(provider, client=client)
    try:
        monkeypatch.setenv(provider.api_key_env, "first-opencode-key")
        responses("first")
        native("first")
        monkeypatch.setenv(provider.api_key_env, "second-opencode-key")
        responses("second")
        native("second")
    finally:
        client.close()

    assert [request.headers["authorization"] for request in captured] == [
        "Bearer first-opencode-key",
        "Bearer first-opencode-key",
        "Bearer second-opencode-key",
        "Bearer second-opencode-key",
    ]
    anthropic_requests = [
        request for request in captured if "anthropic" in str(request.url)
    ]
    assert all("x-api-key" not in request.headers for request in anthropic_requests)
    serialized = f"{responses.serialize()} {native.serialize()}"
    assert "opencode-key" not in serialized
    assert "anthropic-decoy" not in serialized
    assert "openai-decoy" not in serialized
    assert "other-provider-decoy" not in serialized


def test_chat_fallback_session_is_excluded_and_recreated_on_restore(provider):
    from msgflux import Model

    facade, _ = _classes(provider)
    model = _model(provider)
    session = model._fallback_session_id
    serialized = model.serialize()

    assert serialized["provider"] == provider.provider
    assert "_fallback_session_id" not in serialized["state"]
    assert session not in str(serialized)

    restored = Model.from_serialized(
        serialized["provider"], serialized["model_type"], serialized["state"]
    )

    assert isinstance(restored, facade)
    assert restored.provider == provider.provider
    assert restored._fallback_session_id != session

    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_completion())

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        restored.chat_transport = HTTPChatTransport(client=client, max_retries=0)
        assert restored("Continue after restore").consume() == "OK"

    assert captured[0].headers["x-opencode-session"] == restored._fallback_session_id


def test_responses_fallback_session_is_recreated_and_used_on_restore(provider):
    from msgflux import Model

    facade, _ = _classes(provider)
    model = _model(provider, api_mode="responses")
    session = model._fallback_session_id
    serialized = model.serialize()

    assert serialized["provider"] == provider.provider
    assert serialized["state"]["api_mode"] == "responses"
    assert "_fallback_session_id" not in serialized["state"]

    restored = Model.from_serialized(
        serialized["provider"], serialized["model_type"], serialized["state"]
    )

    assert isinstance(restored, facade)
    assert restored.api_mode == "responses"
    assert restored._fallback_session_id != session

    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_responses_completion())

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        restored.chat_transport = HTTPChatTransport(client=client, max_retries=0)
        assert restored("Continue after restore").consume() == "OK"

    assert str(captured[0].url) == f"{provider.base_url}/responses"
    assert captured[0].headers["x-opencode-session"] == (restored._fallback_session_id)


def test_anthropic_fallback_session_is_recreated_and_used_on_restore(provider):
    from msgflux import Model

    _, backend = _classes(provider)
    model = _native_model(provider)
    session = model._fallback_session_id
    serialized = model.serialize()

    assert serialized["provider"] == provider.provider
    assert serialized["state"]["api_mode"] == "anthropic_messages"
    assert "_fallback_session_id" not in serialized["state"]
    assert "test-key" not in str(serialized)
    assert "x-api-key" not in str(serialized)

    restored = Model.from_serialized(
        serialized["provider"], serialized["model_type"], serialized["state"]
    )

    assert isinstance(restored, backend)
    assert restored.provider == provider.provider
    assert restored.api_mode == "anthropic_messages"
    assert restored._fallback_session_id != session

    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json=_anthropic_completion())

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        restored.http_transport = HTTPTransport(client=client, max_retries=0)
        assert restored("Continue after restore").consume() == "OK"

    request = captured[0]
    assert str(request.url) == f"{provider.anthropic_base}/v1/messages"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-opencode-session"] == (restored._fallback_session_id)
    assert "x-api-key" not in request.headers


def test_cross_provider_credentials_and_bases_stay_isolated(monkeypatch):
    monkeypatch.setenv(GO.api_key_env, "go-key")
    monkeypatch.setenv(ZEN.api_key_env, "zen-key")
    monkeypatch.setenv(GO.base_url_env, "https://go-gateway.example/openai/v1")
    monkeypatch.setenv(ZEN.base_url_env, "https://zen-gateway.example/openai/v1")
    captured = []

    def handler(request):
        captured.append(request)
        url = str(request.url)
        if url.endswith("/chat/completions"):
            return httpx2.Response(200, json=_completion())
        if url.endswith("/responses"):
            return httpx2.Response(200, json=_responses_completion())
        return httpx2.Response(200, json=_anthropic_completion())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    models = []
    try:
        for case in (GO, ZEN):
            models.extend(
                [
                    _model(case, client=client),
                    _model(case, client=client, api_mode="responses"),
                    _native_model(case, client=client),
                ]
            )
        for model in models:
            model("Isolate")
    finally:
        client.close()

    assert [str(request.url) for request in captured] == [
        "https://go-gateway.example/openai/v1/chat/completions",
        "https://go-gateway.example/openai/v1/responses",
        "https://go-gateway.example/anthropic/v1/messages",
        "https://zen-gateway.example/openai/v1/chat/completions",
        "https://zen-gateway.example/openai/v1/responses",
        "https://zen-gateway.example/anthropic/v1/messages",
    ]
    assert [request.headers["authorization"] for request in captured] == [
        "Bearer go-key",
        "Bearer go-key",
        "Bearer go-key",
        "Bearer zen-key",
        "Bearer zen-key",
        "Bearer zen-key",
    ]
    anthropic_requests = [
        request for request in captured if "anthropic" in str(request.url)
    ]
    assert all("x-api-key" not in request.headers for request in anthropic_requests)


def test_cross_provider_serialization_restores_its_own_backend(monkeypatch):
    import msgflux as mf

    monkeypatch.setenv(GO.api_key_env, "go-key")
    monkeypatch.setenv(ZEN.api_key_env, "zen-key")

    for case in (GO, ZEN):
        facade, backend = _classes(case)
        expected = {
            "chat_completions": facade,
            "responses": facade,
            "anthropic_messages": backend,
        }
        for api_mode, expected_class in expected.items():
            if api_mode == "anthropic_messages":
                model = _native_model(case)
            else:
                model = _model(case, api_mode=api_mode)
            serialized = model.serialize()

            assert serialized["provider"] == case.provider
            restored = mf.Model.from_serialized(
                serialized["provider"],
                serialized["model_type"],
                serialized["state"],
            )

            assert type(restored) is expected_class
            assert restored.provider == case.provider
            assert restored.api_mode == api_mode
            other = _other_provider(case)
            other_facade, other_backend = _classes(other)
            assert not isinstance(restored, other_facade)
            assert not isinstance(restored, other_backend)


def test_anthropic_typed_tools_signed_history_replay_and_cache_usage(provider):
    requests = []
    thinking_block = {
        "type": "thinking",
        "thinking": "Need the order.",
        "signature": "sig-opencode-nonstream",
    }
    tool_block = {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "submit_order",
        "input": {"quantity": 2, "items": [{"sku": "BOLT-17"}]},
    }

    def handler(request):
        body = _request_body(request)
        requests.append(body)
        if len(requests) == 1:
            return httpx2.Response(
                200,
                json=_anthropic_completion(
                    content=[thinking_block, tool_block],
                    stop_reason="tool_use",
                    usage={
                        "input_tokens": 17,
                        "output_tokens": 20,
                        "cache_creation_input_tokens": 1370,
                        "cache_read_input_tokens": 9,
                    },
                ),
            )
        return httpx2.Response(
            200, json=_anthropic_completion(content=[{"type": "text", "text": "DONE"}])
        )

    model = _native_model(provider, prompt_cache=True)
    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        model.http_transport = HTTPTransport(client=client, max_retries=0)
        response = model("Submit two BOLT-17 orders.")
        assert response.consume().get_calls() == [
            (
                "toolu_1",
                "submit_order",
                {"quantity": 2, "items": [{"sku": "BOLT-17"}]},
            )
        ]
        assert requests[0]["cache_control"] == {"type": "ephemeral"}

        usage = response.metadata.usage
        assert usage.input_tokens == 1396
        assert usage.input_tokens_details.cache_write_tokens == 1370
        assert usage.input_tokens_details.cached_tokens == 9
        assert usage.cache_hit_percentage == pytest.approx(9 / 1396 * 100)

        intent = next(iter(response.get_tool_intents()))
        messages = ChatMessages(
            [{"role": "user", "content": "Submit two BOLT-17 orders."}],
            thread_id="opencode-tools",
        )
        for item in response.history_items:
            messages.append(item)
        messages.append(
            {
                "type": "function_call",
                "call_id": intent.id,
                "name": intent.name,
                "arguments": intent.arguments,
            }
        )
        messages.add_tool(intent.id, "accepted")
        final = model(messages)

    assert final.consume() == "DONE"
    replay = requests[1]
    assert replay["messages"][0] == {
        "role": "user",
        "content": "Submit two BOLT-17 orders.",
    }
    assistant = replay["messages"][1]["content"]
    assert assistant[0] == thinking_block
    assert assistant[1] == tool_block
    assert replay["messages"][2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "accepted",
        }
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_anthropic_stream_tool_outcomes_replay_one_call_and_result(
    provider, asynchronous
):
    requests = []
    stream_body = _anthropic_sse(TOOL_STREAM_EVENTS)

    def handler(request):
        body = _request_body(request)
        requests.append(body)
        if body.get("stream"):
            return httpx2.Response(
                200,
                text=stream_body,
                headers={"content-type": "text/event-stream"},
            )
        return httpx2.Response(
            200, json=_anthropic_completion(content=[{"type": "text", "text": "DONE"}])
        )

    with httpx2.Client(transport=httpx2.MockTransport(handler)) as client:
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler)
        ) as aclient:
            model = _native_model(provider, client=client, async_client=aclient)
            with thread_context(thread_id="opencode-stream"):
                if asynchronous:
                    response = await model.acall(
                        "Submit two.", stream=True, tool_catalog=_tool_catalog()
                    )
                else:
                    response = model(
                        "Submit two.", stream=True, tool_catalog=_tool_catalog()
                    )
                [chunk async for chunk in response.consume()]

            snapshot = response.chat_accumulator.snapshot()
            (intent,) = list(response.get_tool_intents())
            assert intent.name == "submit_order"
            assert intent.arguments == {"quantity": 2}
            assert [item["type"] for item in snapshot] == [
                "reasoning",
                "function_call",
            ]

            outcomes = response.render_tool_outcomes(
                [ToolOutcome.completed(intent, {"status": "accepted"})]
            )
            assert [item["type"] for item in outcomes] == ["function_call_output"]
            assert not any(item.get("type") == "function_call" for item in outcomes)

            messages = ChatMessages(
                json.loads(
                    json.dumps(
                        [
                            {
                                "type": "message",
                                "role": "user",
                                "content": "Submit two.",
                            },
                            *snapshot,
                            *outcomes,
                        ]
                    )
                ),
                thread_id="opencode-replay",
            )
            replay_catalog = ToolCatalog(tools=_tool_catalog().tools, choice="none")
            if asynchronous:
                final = await model.acall(messages, tool_catalog=replay_catalog)
            else:
                final = model(messages, tool_catalog=replay_catalog)
            assert final.consume() == "DONE"

    assert len(requests) == 2
    assert requests[0]["stream"] is True
    replay = requests[1]
    assert [message["role"] for message in replay["messages"]] == [
        "user",
        "assistant",
        "user",
    ]
    assistant = replay["messages"][1]["content"]
    assert [block["type"] for block in assistant] == ["thinking", "tool_use"]
    assert assistant[0]["signature"] == TOOL_STREAM_SIGNATURE
    assert [block for block in assistant if block["type"] == "tool_use"] == [
        {
            "type": "tool_use",
            "id": "toolu_stream_1",
            "name": "submit_order",
            "input": {"quantity": 2},
        }
    ]
    tool_results = [
        block
        for block in replay["messages"][2]["content"]
        if block["type"] == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["tool_use_id"] == "toolu_stream_1"
    assert json.loads(tool_results[0]["content"]) == {"status": "accepted"}
