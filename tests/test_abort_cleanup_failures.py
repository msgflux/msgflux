"""Cancellation must not be masked by operation cleanup failures."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.exceptions import AbortRequestedError
from msgflux.nn import Agent
from msgflux.runtime import AbortSignal, ExecutionScope
from msgflux.runtime.abort import await_with_abort
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore


def _assert_cleanup_note(error, message):
    notes = getattr(error, "__notes__", ())
    assert any(message in note for note in notes), notes


@pytest.mark.asyncio
async def test_await_with_abort_preserves_caller_cancel_when_cleanup_fails():
    started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    failure = ValueError("cleanup failed after cancel")
    operation_task = None

    async def operation():
        nonlocal operation_task
        operation_task = asyncio.current_task()
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_finished.set()
            raise failure from None

    signal = AbortSignal()
    task = asyncio.create_task(await_with_abort(operation(), signal))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel("caller cancelled")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        assert cleanup_finished.is_set()
        _assert_cleanup_note(raised.value, "cleanup failed after cancel")
        assert task.cancelled()
        assert operation_task.done()
        assert not signal._waiters
    finally:
        signal.abort("test cleanup")
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_await_with_abort_preserves_abort_when_cleanup_fails():
    started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    failure = ValueError("cleanup failed after abort")
    operation_task = None

    async def operation():
        nonlocal operation_task
        operation_task = asyncio.current_task()
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_finished.set()
            raise failure from None

    signal = AbortSignal()
    task = asyncio.create_task(await_with_abort(operation(), signal))
    try:
        await asyncio.wait_for(started.wait(), 2)
        signal.abort("operator stop")
        with pytest.raises(AbortRequestedError, match="operator stop") as raised:
            await task
        assert cleanup_finished.is_set()
        _assert_cleanup_note(raised.value, "cleanup failed after abort")
        assert operation_task.done()
        assert not signal._waiters
    finally:
        signal.abort("test cleanup")
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_await_with_abort_keeps_normal_success_and_error_outcomes():
    signal = AbortSignal()

    async def success():
        return "ok"

    async def failure_operation():
        raise failure

    assert await await_with_abort(success(), signal) == "ok"
    assert not signal._waiters

    failure = RuntimeError("operation failed")
    with pytest.raises(RuntimeError) as raised:
        await await_with_abort(failure_operation(), signal)
    assert raised.value is failure
    assert not getattr(failure, "__notes__", ())
    assert not signal._waiters


@pytest.mark.asyncio
async def test_await_with_abort_pre_aborted_signal_joins_cleanup():
    started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    failure = ValueError("cleanup failed before abort")
    operation_task = None

    async def operation():
        nonlocal operation_task
        operation_task = asyncio.current_task()
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_finished.set()
            raise failure from None

    running = asyncio.create_task(operation())
    signal = AbortSignal()
    try:
        await asyncio.wait_for(started.wait(), 2)
        signal.abort("already aborted")
        with pytest.raises(AbortRequestedError, match="already aborted") as raised:
            await await_with_abort(running, signal)
        assert cleanup_finished.is_set()
        _assert_cleanup_note(raised.value, "cleanup failed before abort")
        assert operation_task.done()
        assert not signal._waiters
    finally:
        signal.abort("test cleanup")
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fatal_error", [SystemExit, KeyboardInterrupt])
async def test_await_with_abort_pre_aborted_signal_propagates_fatal_future(
    fatal_error,
):
    signal = AbortSignal()
    signal.abort("old abort")
    operation = asyncio.get_running_loop().create_future()
    operation.set_exception(fatal_error("new fatal control"))

    with pytest.raises(fatal_error, match="new fatal control"):
        await await_with_abort(operation, signal)

    assert operation.done()
    assert not signal._waiters


@pytest.mark.asyncio
async def test_await_with_abort_repeated_cancel_remains_observable_during_cleanup():
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    failure = ValueError("cleanup failed after repeated cancel")
    operation_task = None

    async def operation():
        nonlocal operation_task
        operation_task = asyncio.current_task()
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_finished.set()
            raise failure from None

    signal = AbortSignal()
    task = asyncio.create_task(await_with_abort(operation(), signal))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel("first cancellation")
        await asyncio.wait_for(cleanup_started.wait(), 2)
        task.cancel("second cancellation")
        await asyncio.sleep(0)
        assert not task.done()
        assert not cleanup_finished.is_set()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(task, 2)
        assert cleanup_finished.is_set()
        _assert_cleanup_note(raised.value, "cleanup failed after repeated cancel")
        assert operation_task.done()
        assert not signal._waiters
    finally:
        release_cleanup.set()
        signal.abort("test cleanup")
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_agent_cancellation_cleans_abort_waiter_after_provider_cleanup_error():
    started = asyncio.Event()
    waiters = []
    failure = OSError("provider cleanup failed")

    class TrackedSignal(AbortSignal):
        async def wait(self):
            waiters.append(asyncio.current_task())
            await super().wait()

    async def respond(**_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError as error:
            raise failure from error

    agent = Agent(name="abort-cleanup", model=Mock(model_type="chat_completion"))
    agent.generator.aforward = AsyncMock(side_effect=respond)
    signal = TrackedSignal()
    task = asyncio.create_task(
        agent.acall(
            "hello",
            scope=ExecutionScope(thread_id="abort-cleanup", abort_signal=signal),
        )
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel("caller cancelled")
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
        _assert_cleanup_note(raised.value, "provider cleanup failed")
        assert waiters and all(waiter.done() for waiter in waiters)
        assert not signal._waiters
    finally:
        signal.abort("test cleanup")
        task.cancel()
        await asyncio.gather(task, *waiters, return_exceptions=True)
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_error", [OSError, ValueError])
async def test_service_interrupt_settles_interrupted_when_provider_cleanup_fails(
    tmp_path, cleanup_error
):
    entered = asyncio.Event()
    cleanup_finished = asyncio.Event()
    failure = cleanup_error("provider cleanup failed after interrupt")

    async def respond(**_kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_finished.set()
            raise failure from None

    agent = Agent(name="abort-service", model=Mock(model_type="chat_completion"))
    agent.generator.aforward = AsyncMock(side_effect=respond)
    store = SQLiteServiceStore(tmp_path / "abort-service.sqlite3")
    service = AgentService(store=store)
    service.register("main", lambda _thread_id: AgentSession(agent))
    thread = await service.open_thread("main", thread_id="abort-service")
    wait_task = None
    try:
        receipt = await service.prompt(thread.thread_id, "hello", request_id="req-1")
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert await service.interrupt(thread.thread_id, receipt.run_id) is True

        wait_task = asyncio.create_task(service.wait(thread.thread_id, "req-1"))
        settled = await asyncio.wait_for(wait_task, timeout=5)
        await asyncio.sleep(0)

        assert settled.status == "interrupted"
        assert store.get(thread.thread_id, "req-1").receipt.status == "interrupted"
        assert cleanup_finished.is_set()
        assert not service._workers
        assert not service._producer_tasks
        assert service._cache._entries[thread.thread_id].pins == 0
    finally:
        if wait_task is not None and not wait_task.done():
            wait_task.cancel()
            await asyncio.gather(wait_task, return_exceptions=True)
        await service.aclose()
        store.close()
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_trigger", ["cancel", "abort"])
@pytest.mark.parametrize("cleanup_error", [False, True])
async def test_repeated_cancellation_joins_provider_cleanup(
    first_trigger, cleanup_error
):
    entered = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    provider_task = None
    provider_error = OSError("provider cleanup failed after repeated cancellation")

    async def respond(**_kwargs):
        nonlocal provider_task
        provider_task = asyncio.current_task()
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_finished.set()
            if cleanup_error:
                raise provider_error from None
            raise

    agent = Agent(name="abort-repeat-cancel", model=Mock(model_type="chat_completion"))
    agent.generator.aforward = AsyncMock(side_effect=respond)
    signal = AbortSignal()
    call = asyncio.create_task(
        agent.acall(
            "hello",
            scope=ExecutionScope(thread_id="abort-repeat-cancel", abort_signal=signal),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if first_trigger == "abort":
            signal.abort("operator interrupt")
        else:
            call.cancel("first cancellation")
        await asyncio.wait_for(cleanup_started.wait(), 2)
        call.cancel("second cancellation")
        await asyncio.sleep(0)
        assert not call.done()
        assert not provider_task.done()
        assert not cleanup_finished.is_set()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await asyncio.wait_for(call, 2)
        if cleanup_error:
            _assert_cleanup_note(
                raised.value,
                "provider cleanup failed after repeated cancellation",
            )
        assert cleanup_finished.is_set()
        assert not signal._waiters
    finally:
        release_cleanup.set()
        signal.abort("test cleanup")
        if not call.done():
            call.cancel()
        pending = [call]
        if provider_task is not None and provider_task is not call:
            pending.append(provider_task)
        await asyncio.gather(*pending, return_exceptions=True)
        await agent.aclose()
