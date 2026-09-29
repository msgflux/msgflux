"""Tests for the experimental Codex subscription chat provider."""

import json
import os

import httpx2
import pytest


def _write_auth(path, token="access-one"):
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {"access_token": token, "account_id": "acct-test"},
                "last_refresh": "fixture",
            }
        ),
        encoding="utf-8",
    )
    return path


def _sse(*events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def _completed_response(text="Codex says hello."):
    return {
        "id": "resp_codex_test",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
    }


def _model(auth_file, *, client=None, async_client=None, **kwargs):
    from msgflux.models.providers.openai_codex import CodexChatTransport
    from msgflux.models import Model

    return Model.chat_completion(
        "openai-codex/gpt-5.6",
        auth_file=auth_file,
        chat_transport=CodexChatTransport(
            client=client,
            async_client=async_client,
            max_retries=0,
        ),
        **kwargs,
    )


def test_provider_is_registered_and_keeps_responses_identity(tmp_path):
    from msgflux.models import Model
    from msgflux.models.providers.openai_codex import (
        CodexChatTransport,
        OpenAICodexChatCompletion,
    )

    auth_file = _write_auth(tmp_path / "auth.json")
    model = Model.chat_completion("openai-codex/example-model", auth_file=auth_file)

    assert isinstance(model, OpenAICodexChatCompletion)
    assert isinstance(model.chat_transport, CodexChatTransport)
    assert model.provider == "openai-codex"
    assert model.model_id == "example-model"
    assert model.api_mode == "responses"
    assert model._uses_canonical_history is True


def test_request_context_retains_only_token_fingerprint(tmp_path):
    from msgflux.models.providers.openai_codex import _sent_token_fingerprint

    model = _model(_write_auth(tmp_path / "auth.json"))
    model.credential_resolver.resolve(model)

    fingerprint = _sent_token_fingerprint.get()
    assert isinstance(fingerprint, bytes)
    assert len(fingerprint) == 32
    assert b"access-one" not in fingerprint


def test_request_adapter_targets_codex_and_sets_required_responses_fields(tmp_path):
    from msgflux.models.providers.openai_codex import OpenAICodexChatCompletion

    model = OpenAICodexChatCompletion(
        "gpt-5.6", auth_file=_write_auth(tmp_path / "auth.json")
    )
    request = model.api_adapter.prepare_request(
        model,
        {
            "model": "gpt-5.6",
            "input": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )

    assert request.endpoint == "/codex/responses"
    assert request.json["stream"] is True
    assert request.json["store"] is False
    assert request.json["model"] == "gpt-5.6"
    assert request.json["instructions"] == "You are a helpful assistant."
    assert request.json["include"] == ["reasoning.encrypted_content"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_completion_uses_codex_sse_endpoint_and_request_time_oauth(
    tmp_path, asynchronous
):
    auth_file = _write_auth(tmp_path / "auth.json")
    captured = []
    response_event = {"type": "response.completed", "response": _completed_response()}
    body = _sse(response_event)

    def sync_handler(request):
        captured.append(request)
        return httpx2.Response(200, text=body)

    async def async_handler(request):
        captured.append(request)
        return httpx2.Response(200, text=body)

    if asynchronous:
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(async_handler))
        model = _model(auth_file, async_client=client)
        try:
            response = await model.acall("Hello")
        finally:
            await model.aclose()
    else:
        client = httpx2.Client(transport=httpx2.MockTransport(sync_handler))
        model = _model(auth_file, client=client)
        try:
            response = model("Hello")
        finally:
            model.close()

    request = captured[0]
    assert request.url.path == "/backend-api/codex/responses"
    assert request.headers["authorization"] == "Bearer access-one"
    assert request.headers["chatgpt-account-id"] == "acct-test"
    assert request.headers["accept"] == "text/event-stream"
    payload = json.loads(request.content)
    assert payload["model"] == "gpt-5.6"
    assert payload["stream"] is True
    assert payload["store"] is False
    assert response.consume() == "Codex says hello."
    assert response.metadata.usage.input_tokens == 5
    assert response.metadata.model.provider == "openai-codex"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_nonstream_completion_assembles_output_item_done_events(
    tmp_path, asynchronous
):
    auth_file = _write_auth(tmp_path / "auth.json")
    events = [
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Answer from item.done."}],
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_done_only",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9},
            },
        },
    ]
    body = _sse(*events)

    def sync_handler(_request):
        return httpx2.Response(200, text=body)

    async def async_handler(_request):
        return httpx2.Response(200, text=body)

    if asynchronous:
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(async_handler))
        model = _model(auth_file, async_client=client)
        try:
            response = await model.acall("Answer this")
        finally:
            await model.aclose()
    else:
        client = httpx2.Client(transport=httpx2.MockTransport(sync_handler))
        model = _model(auth_file, client=client)
        try:
            response = model("Answer this")
        finally:
            model.close()

    assert response.consume() == "Answer from item.done."
    assert response.metadata.response_id == "resp_done_only"
    assert response.metadata.usage.total_tokens == 9


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_streaming_decodes_text_and_usage_from_codex_events(
    tmp_path, asynchronous
):
    auth_file = _write_auth(tmp_path / "auth.json")
    captured = []
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "role": "assistant", "phase": "final_answer"},
        },
        {"type": "response.output_text.delta", "output_index": 0, "delta": "Streamed"},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_stream",
                "status": "completed",
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 2,
                    "total_tokens": 9,
                    "input_tokens_details": {"cached_tokens": 4},
                },
            },
        },
    ]
    body = _sse(*events)

    def sync_handler(request):
        captured.append(request)
        return httpx2.Response(200, text=body)

    async def async_handler(request):
        captured.append(request)
        return httpx2.Response(200, text=body)

    if asynchronous:
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(async_handler))
        model = _model(auth_file, async_client=client)
        try:
            stream = await model.acall("Hello", stream=True)
            chunks = [chunk async for chunk in stream.consume()]
        finally:
            await model.aclose()
    else:
        client = httpx2.Client(transport=httpx2.MockTransport(sync_handler))
        model = _model(auth_file, client=client)
        try:
            stream = model("Hello", stream=True)
            chunks = [chunk async for chunk in stream.consume()]
        finally:
            model.close()

    assert chunks == ["Streamed"]
    assert stream.metadata.usage.input_tokens == 7
    assert stream.metadata.usage.input_tokens_details.cached_tokens == 4
    assert stream.metadata.usage.cache_hit_percentage == pytest.approx(4 / 7 * 100)
    assert json.loads(captured[0].content)["stream"] is True


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_streaming_preserves_commentary_tool_and_encrypted_reasoning(
    tmp_path, asynchronous
):
    from msgflux.chat_messages import ChatMessages

    auth_file = _write_auth(tmp_path / "auth.json")
    captured = []
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs_codex",
                "encrypted_content": "opaque-reasoning",
                "summary": [],
            },
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs_codex",
                "encrypted_content": "opaque-reasoning",
                "summary": [],
            },
        },
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"type": "message", "role": "assistant", "phase": "commentary"},
        },
        {
            "type": "response.output_text.delta",
            "output_index": 1,
            "delta": "Looking up.",
        },
        {
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "lookup_status",
                "arguments": '{"code":"A-1"}',
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_tool",
                "status": "completed",
                "usage": {"input_tokens": 9, "output_tokens": 3, "total_tokens": 12},
            },
        },
    ]
    body = _sse(*events)

    def sync_handler(request):
        captured.append(request)
        return httpx2.Response(200, text=body)

    async def async_handler(request):
        captured.append(request)
        return httpx2.Response(200, text=body)

    if asynchronous:
        client = httpx2.AsyncClient(transport=httpx2.MockTransport(async_handler))
        model = _model(auth_file, async_client=client)
        try:
            stream = await model.acall("Check A-1", stream=True)
            async for _chunk in stream.consume():
                pass
        finally:
            await model.aclose()
    else:
        client = httpx2.Client(transport=httpx2.MockTransport(sync_handler))
        model = _model(auth_file, client=client)
        try:
            stream = model("Check A-1", stream=True)
            async for _chunk in stream.consume():
                pass
        finally:
            model.close()

    assert stream.response_type == "tool_call"
    assert stream.data.get_calls() == [("call_1", "lookup_status", {"code": "A-1"})]
    assert stream.commentary == ["Looking up."]
    assert stream.metadata.usage.input_tokens == 9
    assert stream.metadata.model.provider == "openai-codex"
    assert stream.metadata.model.model_id == "gpt-5.6"
    history = stream.chat_accumulator.snapshot()
    assert history[0]["provider_state"] == {
        "provider": "openai-codex",
        "api_mode": "responses",
        "codec": "openai_responses",
        "data": {
            "type": "reasoning",
            "id": "rs_codex",
            "encrypted_content": "opaque-reasoning",
        },
    }
    assert history[-1]["provider_state"]["provider"] == "openai-codex"
    tool_replay = ChatMessages(history).to_responses_input(
        provider="openai-codex",
        api_mode="responses",
        reasoning_codec=model.reasoning_codec,
    )
    assert tool_replay[0]["encrypted_content"] == "opaque-reasoning"
    follow_up = ChatMessages(history)
    follow_up.add_response_items(
        [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "scanner_restarted",
            },
            {"role": "user", "content": "Now report the status."},
        ]
    )
    next_params = model._build_generation_params(
        follow_up,
        system_prompt=None,
        prefilling=None,
        tool_catalog=None,
    )
    next_request = model.api_adapter.prepare_request(model, next_params)
    assert next_request.json["input"][0]["encrypted_content"] == "opaque-reasoning"
    assert next_request.json["model"] == "gpt-5.6"
    assert json.loads(captured[0].content)["stream"] is True


@pytest.mark.parametrize("token_changes", [False, True])
def test_401_retries_only_when_auth_file_has_a_new_access_token(
    tmp_path, token_changes
):
    auth_file = _write_auth(tmp_path / "auth.json")
    captured = []

    def handler(request):
        captured.append(request)
        if len(captured) == 1:
            if token_changes:
                _write_auth(auth_file, "access-two")
            return httpx2.Response(401, text="private error payload")
        return httpx2.Response(
            200,
            text=_sse(
                {"type": "response.completed", "response": _completed_response()}
            ),
        )

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    model = _model(auth_file, client=client)
    try:
        if token_changes:
            response = model("Retry with updated login")
            assert response.consume() == "Codex says hello."
            assert len(captured) == 2
            assert captured[1].headers["authorization"] == "Bearer access-two"
        else:
            with pytest.raises(RuntimeError, match="login expired or revoked") as exc:
                model("Do not retry unchanged login")
            assert "access-one" not in str(exc.value)
            assert "private error payload" not in str(exc.value)
            assert len(captured) == 1
    finally:
        model.close()


def test_changed_auth_file_is_used_by_the_next_request(tmp_path):
    auth_file = _write_auth(tmp_path / "auth.json")
    captured = []

    def handler(request):
        captured.append(request)
        if len(captured) == 1:
            _write_auth(auth_file, "access-two")
        return httpx2.Response(
            200,
            text=_sse(
                {"type": "response.completed", "response": _completed_response()}
            ),
        )

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    model = _model(auth_file, client=client)
    try:
        model("First")
        model("Second")
    finally:
        model.close()

    assert [request.headers["authorization"] for request in captured] == [
        "Bearer access-one",
        "Bearer access-two",
    ]


def test_multiturn_history_preserves_provider_cache_usage(tmp_path):
    from msgflux.chat_messages import ChatMessages

    auth_file = _write_auth(tmp_path / "auth.json")
    requests = []

    def handler(request):
        requests.append(request)
        turn = len(requests)
        usage = {
            "input_tokens": 80,
            "output_tokens": 4,
            "total_tokens": 84,
            "input_tokens_details": {"cached_tokens": 64 if turn == 2 else 0},
        }
        item = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": f"Turn {turn}"}],
        }
        return httpx2.Response(
            200,
            text=_sse(
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {
                    "type": "response.completed",
                    "response": {"status": "completed", "output": [], "usage": usage},
                },
            ),
        )

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    model = _model(auth_file, client=client)
    try:
        first = model(
            ChatMessages(
                [{"role": "user", "content": "First"}], thread_id="cache-thread"
            ),
            system_prompt="Stable instructions",
        )
        history = ChatMessages(
            [
                {"role": "user", "content": "First"},
                *first.history_items,
                {"role": "user", "content": "Second"},
            ],
            thread_id="cache-thread",
        )
        second = model(history, system_prompt="Stable instructions")
    finally:
        model.close()

    assert first.consume() == "Turn 1"
    assert second.consume() == "Turn 2"
    assert first.metadata.usage.input_tokens_details.cached_tokens == 0
    assert first.metadata.usage.cache_hit_percentage == 0.0
    assert second.metadata.usage.input_tokens_details.cached_tokens == 64
    assert second.metadata.usage.cache_hit_percentage == 80.0
    second_input = json.loads(requests[1].content)["input"]
    assert any(item.get("role") == "assistant" for item in second_input)
    for request in requests:
        assert request.headers["session-id"] == "cache-thread"
        assert request.headers["x-client-request-id"] == "cache-thread"
        assert json.loads(request.content)["prompt_cache_key"] == "cache-thread"


def test_codex_cache_key_uses_active_thread_and_clamps_all_locations(tmp_path):
    from msgflux.runtime.context import execution_context

    model = _model(_write_auth(tmp_path / "auth.json"))
    with execution_context(thread_id="thread-" + "x" * 70):
        params = model._build_generation_params(
            "Hello", system_prompt=None, prefilling=None, tool_catalog=None
        )
        request = model.api_adapter.prepare_request(model, params)

    expected = ("thread-" + "x" * 70)[:64]
    assert request.json["prompt_cache_key"] == expected
    assert request.headers["session-id"] == expected
    assert request.headers["x-client-request-id"] == expected


def test_codex_explicit_cache_key_overrides_thread_in_body_and_headers(tmp_path):
    from msgflux.chat_messages import ChatMessages

    model = _model(_write_auth(tmp_path / "auth.json"))
    params = model._build_generation_params(
        ChatMessages([{"role": "user", "content": "Hello"}], thread_id="thread-id"),
        system_prompt=None,
        prefilling=None,
        tool_catalog=None,
        extra_body={"prompt_cache_key": "shared-prefix"},
    )
    request = model.api_adapter.prepare_request(model, params)

    assert request.json["prompt_cache_key"] == "shared-prefix"
    assert request.headers["session-id"] == "shared-prefix"
    assert request.headers["x-client-request-id"] == "shared-prefix"


def test_codex_rejects_unsupported_storage_preference(tmp_path):
    from msgflux.models.providers.openai_codex import OpenAICodexChatCompletion

    model = OpenAICodexChatCompletion(
        "gpt-5.6", auth_file=_write_auth(tmp_path / "auth.json"), store=True
    )

    with pytest.raises(ValueError, match="store=False"):
        model.api_adapter.prepare_request(
            model,
            {"model": "gpt-5.6", "input": [], "store": True},
        )

    limited = OpenAICodexChatCompletion(
        "gpt-5.6", auth_file=tmp_path / "auth.json", max_tokens=128
    )
    with pytest.raises(ValueError, match="max_output_tokens"):
        limited.api_adapter.prepare_request(
            limited,
            {"model": "gpt-5.6", "input": [], "max_output_tokens": 128},
        )


@pytest.mark.skipif(
    os.getenv("MSGFLUX_CODEX_SMOKE") != "1"
    or not os.getenv("MSGFLUX_CODEX_AUTH_FILE")
    or not os.getenv("MSGFLUX_CODEX_MODEL"),
    reason=(
        "Set MSGFLUX_CODEX_SMOKE=1, MSGFLUX_CODEX_AUTH_FILE, and "
        "MSGFLUX_CODEX_MODEL to run the live Codex smoke test."
    ),
)
def test_live_codex_agent_tool_loop_smoke():
    """Opt-in paid check for two model turns and a tool call."""
    import msgflux as mf
    from msgflux.nn import Agent

    calls = []

    def lookup_status(code: str) -> str:
        """Return a fixed status for the requested fixture code."""
        calls.append(code)
        return "scanner_restarted"

    agent = Agent(
        name="codex_smoke",
        model=mf.Model.chat_completion(
            f"openai-codex/{os.environ['MSGFLUX_CODEX_MODEL']}",
            auth_file=os.environ["MSGFLUX_CODEX_AUTH_FILE"],
            reasoning_effort="low",
        ),
        tools=[lookup_status],
        system_prompt=(
            "Call lookup_status exactly once with the incident code in the user "
            "message, then report its result verbatim."
        ),
    )

    result = agent("Check incident code SCANNER-42.")

    assert calls == ["SCANNER-42"]
    assert "scanner_restarted" in str(result)
