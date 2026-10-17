"""Audit request effort separately from native in-conversation updates."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.models.chat_api import ChatTransport
from msgflux.models.providers.openai import OpenAIChatCompletion


class RecordingTransport(ChatTransport):
    def __init__(self):
        self.requests = []

    def _response(self, request):
        self.requests.append(deepcopy(request.json))
        if request.api == "responses":
            if request.params.get("stream"):
                return [
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {"type": "message", "role": "assistant"},
                    },
                    {
                        "type": "response.output_text.delta",
                        "output_index": 0,
                        "delta": "done",
                    },
                    {
                        "type": "response.completed",
                        "response": {"id": "resp_1", "status": "completed"},
                    },
                ]
            return SimpleNamespace(
                id="resp_1",
                status="completed",
                incomplete_details=None,
                usage=None,
                output=[
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "done"}],
                    }
                ],
            )
        choice = SimpleNamespace(
            finish_reason="stop",
            index=0,
            logprobs=None,
            message=SimpleNamespace(content="done", tool_calls=None, refusal=None),
            delta=SimpleNamespace(content="done", tool_calls=None),
        )
        response = SimpleNamespace(id="chat_1", usage=None, choices=[choice])
        return [response] if request.params.get("stream") else response

    def create(self, owner, request):
        result = self._response(request)
        return iter(result) if request.params.get("stream") else result

    async def acreate(self, owner, request):
        result = self._response(request)
        if not request.params.get("stream"):
            return result

        async def events():
            for item in result:
                yield item

        return events()


def _history():
    return ChatMessages(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "previous"},
            {"type": "model_configuration", "reasoning_effort": "high"},
            {"role": "user", "content": "next"},
        ]
    )


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("model_id", "api_mode", "request_effort", "conversation_effort"),
    [
        ("gpt-6-astra", "responses", "low", "high"),
        ("gpt-5.6", "responses", "high", None),
        ("gpt-6-astra", "chat_completions", "high", None),
    ],
)
@pytest.mark.asyncio
async def test_actual_request_reasoning_metadata(
    asynchronous, stream, model_id, api_mode, request_effort, conversation_effort
):
    model = OpenAIChatCompletion(
        model_id=model_id, api_mode=api_mode, reasoning_effort="low"
    )
    transport = RecordingTransport()
    model.chat_transport = transport
    try:
        response = (
            await model.acall(_history(), stream=stream)
            if asynchronous
            else model(_history(), stream=stream)
        )
        if stream:
            assert [chunk async for chunk in response.consume()] == ["done"]
        assert response.metadata.model.request_reasoning_effort == request_effort
        assert (
            response.metadata.model.conversation_reasoning_effort == conversation_effort
        )
        assert response.metadata.model.reasoning_effort == "high"
        sent = transport.requests[-1]
        assert not any(key.startswith("_") for key in sent)
        if api_mode == "responses":
            assert sent["reasoning"]["effort"] == request_effort
        else:
            assert sent["reasoning_effort"] == request_effort
        assert model.sampling_run_params["reasoning_effort"] == "low"
    finally:
        await model.aclose()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_response_cache_preserves_request_and_conversation_metadata(asynchronous):
    model = OpenAIChatCompletion(
        model_id="gpt-6-astra", reasoning_effort="low", enable_cache=True
    )
    transport = RecordingTransport()
    model.chat_transport = transport
    try:
        first = await model.acall(_history()) if asynchronous else model(_history())
        second = await model.acall(_history()) if asynchronous else model(_history())
        assert len(transport.requests) == 1
        assert second.metadata.model.request_reasoning_effort == "low"
        assert second.metadata.model.conversation_reasoning_effort == "high"
        assert second.metadata.model.reasoning_effort == "high"
        second.metadata.model.request_reasoning_effort = "changed-copy"
        assert first.metadata.model.request_reasoning_effort == "low"
    finally:
        await model.aclose()


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
@pytest.mark.asyncio
async def test_absent_request_reasoning_is_explicitly_known(api_mode):
    model = OpenAIChatCompletion(model_id="gpt-6-astra", api_mode=api_mode)
    model.chat_transport = RecordingTransport()
    try:
        response = model("work")
        assert "request_reasoning_effort" in response.metadata.model
        assert response.metadata.model.request_reasoning_effort is None
        assert response.metadata.model.conversation_reasoning_effort is None
        assert response.metadata.model.get("reasoning_effort") is None
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_extra_body_overrides_are_audited_from_the_final_wire_request():
    model = OpenAIChatCompletion(model_id="gpt-6-astra", reasoning_effort="low")
    transport = RecordingTransport()
    model.chat_transport = transport
    try:
        response = model(
            _history(),
            extra_body={"reasoning": {"effort": "medium"}, "input": []},
        )
        assert transport.requests[-1]["reasoning"] == {"effort": "medium"}
        assert transport.requests[-1]["input"] == []
        assert response.metadata.model.request_reasoning_effort == "medium"
        assert response.metadata.model.conversation_reasoning_effort is None
        assert response.metadata.model.reasoning_effort == "medium"
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_openrouter_effort_is_read_from_its_adapted_reasoning_object(monkeypatch):
    from msgflux.models.providers.openrouter import OpenRouterChatCompletion

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    model = OpenRouterChatCompletion(
        model_id="openai/gpt-6-astra", reasoning_effort="low"
    )
    transport = RecordingTransport()
    model.chat_transport = transport
    try:
        response = await model.acall(_history())
        assert transport.requests[-1]["reasoning"]["effort"] == "high"
        assert "reasoning_effort" not in transport.requests[-1]
        assert response.metadata.model.request_reasoning_effort == "high"
        assert response.metadata.model.conversation_reasoning_effort is None
        assert response.metadata.model.reasoning_effort == "high"
    finally:
        await model.aclose()


@pytest.mark.asyncio
async def test_concurrent_requests_on_one_model_keep_metadata_isolated():
    class BarrierTransport(RecordingTransport):
        def __init__(self):
            super().__init__()
            self.both_entered = asyncio.Event()

        async def acreate(self, owner, request):
            result = self._response(request)
            if len(self.requests) == 2:
                self.both_entered.set()
            await asyncio.wait_for(self.both_entered.wait(), timeout=5)
            return result

    model = OpenAIChatCompletion(model_id="gpt-6-astra", reasoning_effort="low")
    transport = BarrierTransport()
    model.chat_transport = transport
    try:
        first, second = await asyncio.gather(
            model.acall("first", extra_body={"reasoning": {"effort": "medium"}}),
            model.acall("second", extra_body={"reasoning": {"effort": "high"}}),
        )
        assert first.metadata.model.request_reasoning_effort == "medium"
        assert first.metadata.model.reasoning_effort == "medium"
        assert second.metadata.model.request_reasoning_effort == "high"
        assert second.metadata.model.reasoning_effort == "high"
        assert model.sampling_run_params["reasoning_effort"] == "low"
        assert all(
            not any(key.startswith("_") for key in body) for body in transport.requests
        )
    finally:
        await model.aclose()
