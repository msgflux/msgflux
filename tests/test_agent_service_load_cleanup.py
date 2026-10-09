"""Failed initialization must not overlap incompletely closed bindings."""

from unittest.mock import Mock

import pytest

from msgflux.nn import Agent
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    ServiceRecoveryRequiredError,
    SQLiteServiceStore,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_failed_initialization_retries_only_after_successful_cleanup(
    cleanup_fails,
):
    calls = 0
    closed = 0
    cleanup_error = OSError("cannot close factory-owned resource")

    def factory(_thread):
        nonlocal calls
        calls += 1
        agent = Agent(name="load-cleanup", model=Mock(model_type="chat_completion"))

        def scope_factory(scope):
            if calls == 1:
                raise ValueError("workspace binding failed")
            return scope

        def close():
            nonlocal closed
            closed += 1
            if cleanup_fails:
                raise cleanup_error

        return AgentSession(agent, scope_factory=scope_factory, on_close=close)

    journal = SQLiteServiceStore()
    service = AgentService(store=journal)
    service.register("agent", factory)
    thread = await service.open_thread("agent")
    try:
        if cleanup_fails:
            for _ in range(2):
                with pytest.raises(ServiceRecoveryRequiredError) as raised:
                    await service.acquire_session(thread.thread_id)
                assert raised.value.__cause__ is cleanup_error
            assert calls == closed == 1
            assert thread.thread_id in service._sessions
            with pytest.raises(ExceptionGroup):
                await service.aclose()
            assert closed == 1
        else:
            with pytest.raises(ValueError, match="workspace binding failed"):
                await service.acquire_session(thread.thread_id)
            assert not service._sessions
            lease = await service.acquire_session(thread.thread_id)
            await lease.aclose()
            assert calls == 2
            assert closed == 1
    finally:
        if not cleanup_fails:
            await service.aclose()
        journal.close()
