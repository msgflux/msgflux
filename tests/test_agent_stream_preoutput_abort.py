"""Abort signals must wake an Agent waiting for the first stream output."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import AbortRequestedError, TaskInterruptRequestedError
from msgflux.models.response import ModelStreamResponse
from msgflux.nn import Agent
from msgflux.nn.modules.agent.model_runtime import AgentModelRuntimeMixin
from msgflux.runtime import AbortSignal
from msgflux.runtime.context import ExecutionScope, execution_context


def _agent_and_stream(name, store, mode):
    model = Mock(model_type="chat_completion")
    model.aclose = AsyncMock()
    agent = Agent(
        name=name,
        model=model,
        checkpoint_store=store,
        config={"stream": True},
    )
    stream = ModelStreamResponse(mode=mode)
    return agent, stream


def _runtime():
    return AgentModelRuntimeMixin()


def _observe_event_wait(event):
    """Signal when readiness starts waiting, keeping the real wait behavior."""
    entered = threading.Event()
    original_wait = event.wait

    def observed_wait(*args, **kwargs):
        entered.set()
        return original_wait(*args, **kwargs)

    event.wait = observed_wait
    return entered


@pytest.mark.parametrize("stream_mode", ["sync", "async"])
def test_sync_agent_pre_output_abort_interrupts_readiness_wait(stream_mode):
    store = InMemoryCheckpointStore()
    agent, stream = _agent_and_stream("sync-pre-output-abort", store, stream_mode)
    returned = threading.Event()
    entered = threading.Event()
    errors = []

    def model_call(**_kwargs):
        returned.set()
        return stream

    wait_for_type = agent._wait_stream_response_type

    def observed_wait(model_response):
        entered.set()
        return wait_for_type(model_response)

    agent.generator.forward = Mock(side_effect=model_call)
    agent._wait_stream_response_type = observed_wait
    signal = AbortSignal()
    scope = ExecutionScope(
        thread_id="sync-pre-output-abort",
        run_id="sync-pre-output-abort-run",
        abort_signal=signal,
    )

    def run():
        try:
            agent("hello", scope=scope)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        assert returned.wait(timeout=2)
        assert entered.wait(timeout=2)
        signal.abort("stop before first token")
        worker.join(timeout=2)
        assert not worker.is_alive(), "Agent did not observe pre-output abort"

        assert len(errors) == 1
        assert isinstance(errors[0], TaskInterruptRequestedError)
        assert "stop before first token" in str(errors[0])

        # A provider that ignores the abort and finishes later must not
        # overwrite the interrupted checkpoint.
        stream.finish()
        state = store.load_state(
            "sync-pre-output-abort",
            "sync-pre-output-abort",
            "sync-pre-output-abort-run",
        )
        assert state["status"] == "interrupted"
    finally:
        stream.finish()
        worker.join(timeout=2)
        asyncio.run(agent.aclose())


@pytest.mark.parametrize("stream_mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_async_agent_pre_output_abort_interrupts_readiness_wait(stream_mode):
    store = InMemoryCheckpointStore()
    agent, stream = _agent_and_stream("async-pre-output-abort", store, stream_mode)
    returned = asyncio.Event()
    entered = asyncio.Event()

    async def model_call(**_kwargs):
        returned.set()
        return stream

    wait_for_type = agent._await_stream_response_type

    async def observed_wait(model_response):
        entered.set()
        return await wait_for_type(model_response)

    agent.generator.aforward = model_call
    agent._await_stream_response_type = observed_wait
    signal = AbortSignal()
    scope = ExecutionScope(
        thread_id="async-pre-output-abort",
        run_id="async-pre-output-abort-run",
        abort_signal=signal,
    )
    task = asyncio.create_task(agent.acall("hello", scope=scope))
    try:
        await asyncio.wait_for(returned.wait(), timeout=2)
        await asyncio.wait_for(entered.wait(), timeout=2)
        signal.abort("stop before first token")
        with pytest.raises(
            TaskInterruptRequestedError, match="stop before first token"
        ):
            await asyncio.wait_for(task, timeout=2)

        # A provider that ignores the abort and finishes later must not
        # overwrite the interrupted checkpoint.
        stream.finish()
        state = store.load_state(
            "async-pre-output-abort",
            "async-pre-output-abort",
            "async-pre-output-abort-run",
        )
        assert state["status"] == "interrupted"
    finally:
        stream.finish()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.aclose()


@pytest.mark.parametrize("stream_mode", ["sync", "async"])
@pytest.mark.parametrize("call_style", ["sync", "async"])
@pytest.mark.asyncio
async def test_ready_stream_survives_pre_aborted_signal(stream_mode, call_style):
    owner = _runtime()
    stream = ModelStreamResponse(mode=stream_mode)
    stream.set_response_type("text_generation")
    signal = AbortSignal()
    signal.abort("after readiness")

    with execution_context(abort_signal=signal):
        if call_style == "sync":
            owner._wait_stream_response_type(stream)
        else:
            await owner._await_stream_response_type(stream)

    assert stream.response_type == "text_generation"
    stream.finish()


@pytest.mark.parametrize(
    ("stream_mode", "call_style"), [("sync", "sync"), ("async", "async")]
)
@pytest.mark.asyncio
async def test_readiness_without_signal_settles_empty_completion(
    stream_mode, call_style
):
    owner = _runtime()
    stream = ModelStreamResponse(mode=stream_mode)
    entered = _observe_event_wait(stream._response_type_event)
    errors = []
    worker = None

    if call_style == "sync":

        def run():
            try:
                owner._wait_stream_response_type(stream)
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()

    try:
        if call_style == "sync":
            assert await asyncio.to_thread(entered.wait, 1)
            stream.finish()
            await asyncio.to_thread(worker.join, 2)
            assert not worker.is_alive()
            assert errors == []
        else:
            task = asyncio.create_task(owner._await_stream_response_type(stream))
            assert await asyncio.to_thread(entered.wait, 1)
            stream.finish()
            await asyncio.wait_for(task, timeout=2)
    finally:
        stream.finish()
        if worker is not None:
            worker.join(timeout=2)

    assert stream.response_type == "text_generation"
    assert stream.data == ""


@pytest.mark.parametrize(
    ("stream_mode", "call_style"), [("sync", "sync"), ("async", "async")]
)
@pytest.mark.asyncio
async def test_readiness_error_wakes_without_signal(stream_mode, call_style):
    owner = _runtime()
    stream = ModelStreamResponse(mode=stream_mode)
    error = RuntimeError("provider failed before output")
    entered = _observe_event_wait(stream._response_type_event)
    errors = []
    worker = None

    if call_style == "sync":

        def run():
            try:
                owner._wait_stream_response_type(stream)
            except BaseException as caught:
                errors.append(caught)

        worker = threading.Thread(target=run, daemon=True)
        worker.start()

    try:
        if call_style == "sync":
            assert await asyncio.to_thread(entered.wait, 1)
            stream.set_error(error)
            await asyncio.to_thread(worker.join, 2)
            assert not worker.is_alive()
            assert errors == [error]
        else:
            task = asyncio.create_task(owner._await_stream_response_type(stream))
            assert await asyncio.to_thread(entered.wait, 1)
            stream.set_error(error)
            with pytest.raises(RuntimeError) as raised:
                await asyncio.wait_for(task, timeout=2)
            assert raised.value is error
    finally:
        stream.finish(error=error, status="failed")
        if worker is not None:
            worker.join(timeout=2)


@pytest.mark.asyncio
async def test_async_readiness_sync_stream_does_not_park_event_wait_thread():
    owner = _runtime()
    stream = ModelStreamResponse(mode="sync")
    event = stream._response_type_event
    entered = threading.Event()
    wait_started = threading.Event()
    release = threading.Event()
    original_is_set = event.is_set

    def observed_is_set():
        entered.set()
        return original_is_set()

    def blocking_wait(*_args, **_kwargs):
        wait_started.set()
        entered.set()
        release.wait(timeout=2)
        return False

    event.is_set = observed_is_set
    event.wait = blocking_wait
    signal = AbortSignal()

    try:
        with execution_context(abort_signal=signal):
            task = asyncio.create_task(owner._await_stream_response_type(stream))
            assert await asyncio.to_thread(entered.wait, 1)
            signal.abort("stop before first token")
            with pytest.raises(AbortRequestedError, match="stop before first token"):
                await asyncio.wait_for(task, timeout=1)
            assert not wait_started.is_set()
    finally:
        release.set()
        stream.finish()


def test_unready_sync_stream_observes_existing_abort_before_waiting():
    owner = AgentModelRuntimeMixin()
    stream = ModelStreamResponse(mode="sync")
    signal = AbortSignal()
    signal.abort("before readiness")
    waited = False

    def provider_sets_type_during_wait(timeout=None):
        nonlocal waited
        waited = True
        stream.set_response_type("text_generation")
        return True

    stream._response_type_event.wait = provider_sets_type_during_wait
    try:
        with execution_context(abort_signal=signal):
            with pytest.raises(AbortRequestedError, match="before readiness"):
                owner._wait_stream_response_type(stream)
        assert not waited
    finally:
        stream.finish()
