"""Cancellation must release signal waiters even when provider cleanup fails."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

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
