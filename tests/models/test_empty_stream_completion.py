"""Empty model streams settle as empty text without racing readiness."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from msgflux.exceptions import AbortRequestedError
from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.response import ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.modules.agent.model_runtime import AgentModelRuntimeMixin


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_completed_empty_stream_sets_empty_text_before_finalizer_and_wakes(mode):
    response = ModelStreamResponse(mode=mode)
    response.metadata = {"provider": "fake", "usage": {"total_tokens": 0}}
    final_states = []
    response.add_finalizer(final_states.append)

    response.finish(status="completed")

    assert response.response_type == "text_generation"
    assert response.data == ""
    assert response.metadata == {"provider": "fake", "usage": {"total_tokens": 0}}
    assert response.first_chunk_event.is_set()
    assert response._response_type_event.is_set()
    assert len(final_states) == 1
    assert final_states[0].status == "completed"
    assert final_states[0].response_type == "text_generation"
    assert final_states[0].output == ""
    assert final_states[0].items == [
        {"type": "message", "role": "assistant", "content": ""}
    ]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"\x00", id="binary"),
        pytest.param(ToolCallAggregator(), id="tool-aggregator"),
    ],
)
def test_untyped_binary_and_tool_payloads_wake_without_text_relabel(payload):
    response = ModelStreamResponse(mode="sync")
    response.data = payload

    response.finish(status="completed")

    assert response.first_chunk_event.is_set()
    assert response._response_type_event.is_set()
    assert response.response_type is None
    assert response.data is payload
    with pytest.raises(RuntimeError, match="ended before producing a response type"):
        AgentModelRuntimeMixin._ensure_stream_response_ready(None, response)


@pytest.mark.asyncio
async def test_async_empty_stream_wakes_waiters_on_terminal_events():
    response = ModelStreamResponse(mode="async")
    type_waiter = asyncio.create_task(response._response_type_event.wait())
    first_chunk_waiter = asyncio.create_task(response.first_chunk_event.wait())
    await asyncio.sleep(0)

    response.finish()

    await asyncio.wait_for(type_waiter, timeout=1)
    await asyncio.wait_for(first_chunk_waiter, timeout=1)
    assert response.response_type == "text_generation"
    assert response.data == ""


@pytest.mark.parametrize("status", ["failed", "interrupted"])
def test_unsuccessful_empty_stream_wakes_without_becoming_text(status):
    response = ModelStreamResponse(mode="sync")
    error = RuntimeError("provider stream failed")

    response.finish(error=error, status=status)

    assert response.first_chunk_event.is_set()
    assert response._response_type_event.is_set()
    assert response.response_type is None
    assert response.error is error
    assert response._final_status == status


def test_set_error_keeps_immediate_sync_readiness_notification():
    response = ModelStreamResponse(mode="sync")
    error = RuntimeError("producer failed")

    response.set_error(error)

    assert response.error is error
    assert response.first_chunk_event.wait(timeout=0.1)
    assert response._response_type_event.wait(timeout=0.1)


@pytest.mark.asyncio
async def test_set_error_keeps_immediate_async_readiness_notification():
    response = ModelStreamResponse(mode="async")
    first_waiter = asyncio.create_task(response.first_chunk_event.wait())
    type_waiter = asyncio.create_task(response._response_type_event.wait())
    await asyncio.sleep(0)

    response.set_error(RuntimeError("producer failed"))

    await asyncio.wait_for(asyncio.gather(first_waiter, type_waiter), timeout=1)


def test_empty_stream_finalizer_failure_wakes_with_failure_not_success():
    response = ModelStreamResponse(mode="sync")
    finalizer_error = RuntimeError("finalization failed")
    states = []

    def fail(state):
        states.append(state)
        raise finalizer_error

    response.add_finalizer(fail)
    with pytest.raises(RuntimeError, match="finalization failed") as raised:
        response.finish()

    assert raised.value is finalizer_error
    assert len(states) == 1
    assert states[0].response_type == "text_generation"
    assert states[0].output == ""
    assert response.first_chunk_event.is_set()
    assert response._response_type_event.is_set()
    assert response.response_type is None
    assert response.data is None
    assert response.error is finalizer_error
    assert response._final_status == "failed"


@pytest.mark.parametrize(
    "primary_error",
    [AbortRequestedError("abort"), asyncio.CancelledError("cancel")],
    ids=["abort", "cancel"],
)
@pytest.mark.parametrize(
    "fatal_error",
    [KeyboardInterrupt("interrupt"), SystemExit("exit")],
    ids=["keyboard-interrupt", "system-exit"],
)
def test_fatal_finalizer_error_is_not_masked_by_primary_control_error(
    primary_error, fatal_error
):
    response = ModelStreamResponse(mode="sync")

    def fail(_state):
        raise fatal_error

    response.add_finalizer(fail)

    with pytest.raises(type(fatal_error)) as raised:
        response.finish(error=primary_error, status="interrupted")

    assert raised.value is fatal_error
    assert response.error is primary_error
    assert response._final_status == "interrupted"
    assert response.first_chunk_event.is_set()
    assert response._response_type_event.is_set()


def test_empty_stream_readiness_waits_for_blocked_finalizer():
    response = ModelStreamResponse(mode="sync")
    finalizer_started = threading.Event()
    allow_finalizer = threading.Event()

    def block(_state):
        finalizer_started.set()
        assert allow_finalizer.wait(timeout=2)

    response.add_finalizer(block)
    producer = threading.Thread(target=response.finish)
    producer.start()
    try:
        assert finalizer_started.wait(timeout=1)
        assert not response._response_type_event.is_set()
        assert not response.first_chunk_event.is_set()
        assert response.response_type is None
        assert response.data is None
    finally:
        allow_finalizer.set()
        producer.join(timeout=2)

    assert not producer.is_alive()
    assert response._response_type_event.is_set()
    assert response.first_chunk_event.is_set()


def test_late_finalizer_sees_same_empty_state_while_finish_is_blocked():
    response = ModelStreamResponse(mode="sync")
    finalizer_started = threading.Event()
    allow_finalizer = threading.Event()
    late_states = []

    def block(_state):
        finalizer_started.set()
        assert allow_finalizer.wait(timeout=2)

    response.add_finalizer(block)
    producer = threading.Thread(target=response.finish)
    producer.start()
    try:
        assert finalizer_started.wait(timeout=1)
        response.add_finalizer(late_states.append)

        assert len(late_states) == 1
        assert late_states[0].status == "completed"
        assert late_states[0].response_type == "text_generation"
        assert late_states[0].output == ""
        assert late_states[0].items == [
            {"type": "message", "role": "assistant", "content": ""}
        ]
        assert response.response_type is None
        assert response.data is None
        assert not response._response_type_event.is_set()
    finally:
        allow_finalizer.set()
        producer.join(timeout=2)

    assert not producer.is_alive()
    assert response.response_type == "text_generation"
    assert response.data == ""
    assert response._response_type_event.is_set()


def test_first_output_still_publishes_type_immediately():
    response = ModelStreamResponse(mode="sync")
    response.set_response_type("text_generation")
    response.add("first token")

    assert response._response_type_event.is_set()
    assert response.first_chunk_event.is_set()


class _EmptyCompatibleModel(OpenAICompatibleChatCompletion):
    provider = "fake"
    api_key_env = "FAKE_API_KEY"


def test_openai_compatible_empty_chat_completions_settle_as_empty_text(monkeypatch):
    monkeypatch.setenv("FAKE_API_KEY", "test-only")
    model = _EmptyCompatibleModel(model_id="empty")
    model._execute_with_metadata = lambda _metadata, **_kwargs: iter(())
    response = ModelStreamResponse(mode="sync")
    response.metadata = {"history": "preserved"}

    model._stream_chat_completions_generate(stream_response=response)

    assert response.error is None
    assert response._final_status == "completed"
    assert response.response_type == "text_generation"
    assert response.data == ""
    assert response.metadata.model.provider == "fake"


def test_openai_compatible_does_not_publish_empty_readiness_during_cleanup(
    monkeypatch,
):
    monkeypatch.setenv("FAKE_API_KEY", "test-only")
    model = _EmptyCompatibleModel(model_id="empty")
    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()
    response = ModelStreamResponse(mode="sync")

    def output():
        try:
            yield SimpleNamespace(usage=None, choices=[])
        finally:
            cleanup_started.set()
            assert allow_cleanup.wait(timeout=2)

    model._execute_with_metadata = lambda _metadata, **_kwargs: output()
    producer = threading.Thread(
        target=model._stream_chat_completions_generate,
        kwargs={"stream_response": response},
    )
    producer.start()
    try:
        assert cleanup_started.wait(timeout=1)
        assert not response._response_type_event.is_set()
        assert not response.first_chunk_event.is_set()
    finally:
        allow_cleanup.set()
        producer.join(timeout=2)

    assert not producer.is_alive()
    assert response._response_type_event.is_set()
    assert response.response_type == "text_generation"
    assert response.data == ""


def test_openai_responses_empty_completion_is_finalized_as_empty_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    from msgflux.models.providers.openai import OpenAIChatCompletion

    model = OpenAIChatCompletion(model_id="empty", api_mode="responses")
    model._execute_with_metadata = lambda _metadata, **_kwargs: iter(
        [
            {"type": "response.created", "response": {"id": "resp_empty"}},
            {
                "type": "response.completed",
                "response": {"id": "resp_empty", "status": "completed"},
            },
        ]
    )
    response = ModelStreamResponse(mode="sync")
    states = []
    response.add_finalizer(states.append)

    model._stream_responses_generate(
        input=[], model="empty", stream=True, stream_response=response
    )

    assert response.response_type == "text_generation"
    assert response.data == ""
    assert response.error is None
    assert response.metadata.response_id == "resp_empty"
    assert states[0].response_type == "text_generation"
    assert states[0].output == ""


@pytest.mark.asyncio
async def test_async_openai_compatible_empty_chat_completions_settle_as_empty_text(
    monkeypatch,
):
    monkeypatch.setenv("FAKE_API_KEY", "test-only")
    model = _EmptyCompatibleModel(model_id="empty")

    async def empty_output():
        if False:
            yield None

    async def execute(_metadata, **_kwargs):
        return empty_output()

    model._aexecute_with_metadata = execute
    response = ModelStreamResponse(mode="async")

    await model._astream_chat_completions_generate(stream_response=response)

    assert response.error is None
    assert response._final_status == "completed"
    assert response.response_type == "text_generation"
    assert response.data == ""
    assert response._response_type_event.is_set()


@pytest.mark.asyncio
async def test_async_openai_compatible_waits_for_stream_cleanup_before_empty_ready(
    monkeypatch,
):
    monkeypatch.setenv("FAKE_API_KEY", "test-only")
    model = _EmptyCompatibleModel(model_id="empty")
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    response = ModelStreamResponse(mode="async")

    async def output():
        try:
            yield SimpleNamespace(usage=None, choices=[])
        finally:
            cleanup_started.set()
            await allow_cleanup.wait()

    async def execute(_metadata, **_kwargs):
        return output()

    model._aexecute_with_metadata = execute
    producer = asyncio.create_task(
        model._astream_chat_completions_generate(stream_response=response)
    )
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=1)
        assert not response._response_type_event.is_set()
        assert not response.first_chunk_event.is_set()
    finally:
        allow_cleanup.set()

    await asyncio.wait_for(producer, timeout=1)
    assert response._response_type_event.is_set()
    assert response.response_type == "text_generation"
    assert response.data == ""


class _FakeEmptyStreamProvider:
    model_type = "chat_completion"
    provider = "fake"

    def __call__(self, **_kwargs):
        response = ModelStreamResponse(mode="sync")
        response.metadata = {"provider": self.provider}
        response.finish()
        return response

    async def acall(self, **_kwargs):
        response = ModelStreamResponse(mode="async")
        response.metadata = {"provider": self.provider}
        response.finish()
        return response

    def close(self):
        pass

    async def aclose(self):
        pass


@pytest.mark.parametrize("asynchronous", [False, True])
def test_agent_accepts_completed_empty_fake_provider_stream(tmp_path, asynchronous):
    agent = Agent(
        name="empty-stream-agent",
        model=_FakeEmptyStreamProvider(),
        config={"stream": True},
        agent_dir=tmp_path / ("async" if asynchronous else "sync"),
    )

    try:
        if asynchronous:
            response = asyncio.run(agent.acall("hello"))
        else:
            response = agent("hello")

        assert isinstance(response, ModelStreamResponse)
        assert response.error is None
        assert response._final_status == "completed"
        assert response.response_type == "text_generation"
        assert response.data == ""
        assert response.metadata["provider"] == "fake"
    finally:
        asyncio.run(agent.aclose())


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_reasoning_only_empty_answer_keeps_reasoning_and_empty_text(mode):
    response = ModelStreamResponse(mode=mode)
    response.add_reasoning("private thought")
    response.reasoning = "private thought"

    response.finish()

    assert response.response_type == "text_generation"
    assert response.data == ""
    assert response.reasoning == "private thought"
    assert response.chat_accumulator.snapshot(fallback_output=response.data) == [
        {"type": "reasoning", "role": "assistant", "text": "private thought"},
        {"type": "message", "role": "assistant", "content": ""},
    ]
