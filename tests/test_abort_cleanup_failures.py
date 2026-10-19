"""Cancellation must release signal waiters even when provider cleanup fails."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.exceptions import TaskInterruptRequestedError
from msgflux.nn import Agent
from msgflux.runtime import AbortSignal, ExecutionScope


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
        with pytest.raises(OSError) as raised:
            await task
        assert raised.value is failure
        assert waiters and all(waiter.done() for waiter in waiters)
        assert not signal._waiters
    finally:
        signal.abort("test cleanup")
        task.cancel()
        await asyncio.gather(task, *waiters, return_exceptions=True)
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
        if cleanup_error:
            with pytest.raises(OSError) as raised:
                await asyncio.wait_for(call, 2)
            assert raised.value is provider_error
        elif first_trigger == "abort":
            with pytest.raises(TaskInterruptRequestedError, match="operator interrupt"):
                await asyncio.wait_for(call, 2)
        else:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(call, 2)
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
