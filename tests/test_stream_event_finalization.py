"""Agent stream finalization scheduler failures settle task-result markers."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.models.response import ModelStreamResponse
from msgflux.nn import Agent
from msgflux.nn.extensions import AgentExtension
from msgflux.runtime.context import ExecutionScope


def _agent_with_stream(tmp_path, *, mode):
    model = Mock(model_type="chat_completion")
    model.close = Mock()
    model.aclose = AsyncMock()
    agent = Agent(
        name="finalization-scheduler",
        model=model,
        agent_dir=tmp_path / "agent-state",
    )
    stream = ModelStreamResponse(mode=mode)
    stream.set_response_type("text_generation")
    if mode == "sync":
        agent.generator.forward = Mock(return_value=stream)
    else:

        async def return_stream(**_kwargs):
            return stream

        agent.generator.aforward = return_stream
    agent.register_extension("scheduler-pin", AgentExtension("scheduler-pin"))
    return agent, stream


def _assert_pins_released(agent, thread_id):
    owned = agent._owned_threads[thread_id]
    assert owned.active == 0
    assert owned.active_calls == 0
    assert owned.detached_finalizers == set()
    assert agent._extension_refcounts.get("scheduler-pin", 0) == 0


@pytest.mark.parametrize("scheduler_result", ["cancelled", "submit_error"])
def test_sync_agent_call_scheduler_completion_never_stays_pending(
    tmp_path, monkeypatch, scheduler_result
):
    agent, stream = _agent_with_stream(tmp_path, mode="sync")
    thread_id = "sync-finalization"
    scope = ExecutionScope(thread_id=thread_id, run_id="sync-finalization-run")

    if scheduler_result == "cancelled":
        scheduler_future = Future()
        scheduler_future.cancel()
        executor = Mock(submit=Mock(return_value=scheduler_future))
    else:
        submit_error = RuntimeError("sync finalizer submit failed")
        executor = Mock(submit=Mock(side_effect=submit_error))

    monkeypatch.setattr(
        "msgflux.nn.modules.module.Executor.get_instance",
        classmethod(lambda _cls: executor),
    )
    try:
        if scheduler_result == "submit_error":
            with pytest.raises(RuntimeError, match="sync finalizer submit failed"):
                agent("hello", scope=scope)
        else:
            assert agent("hello", scope=scope) is stream

        settled = stream._msgflux_event_finalization_future
        assert settled.done()
        if scheduler_result == "cancelled":
            assert settled.cancelled()
        else:
            assert settled.exception() is submit_error
        _assert_pins_released(agent, thread_id)
    finally:
        stream.finish()
        agent.remove_extension("scheduler-pin")
        assert "scheduler-pin" not in agent._pending_extensions
        asyncio.run(agent.aclose())


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduler_result", ["cancelled", "create_task_error"])
async def test_async_agent_call_scheduler_completion_never_stays_pending(
    tmp_path, monkeypatch, scheduler_result
):
    agent, stream = _agent_with_stream(tmp_path, mode="async")
    thread_id = "async-finalization"
    scope = ExecutionScope(thread_id=thread_id, run_id="async-finalization-run")
    create_task = asyncio.create_task
    scheduled = []

    def schedule_finalizer(coro, **kwargs):
        if getattr(getattr(coro, "cr_code", None), "co_name", None) == (
            "_afinalize_detached_event_result"
        ):
            if scheduler_result == "create_task_error":
                raise RuntimeError("async finalizer create_task failed")
            task = create_task(coro, **kwargs)
            scheduled.append(task)
            task.cancel()
            return task
        return create_task(coro, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", schedule_finalizer)
    try:
        if scheduler_result == "create_task_error":
            with pytest.raises(
                RuntimeError, match="async finalizer create_task failed"
            ):
                await agent.acall("hello", scope=scope)
        else:
            assert await agent.acall("hello", scope=scope) is stream
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert scheduled[0].cancelled()

        settled = stream._msgflux_event_finalization_future
        assert settled.done()
        if scheduler_result == "cancelled":
            assert settled.cancelled()
        else:
            assert isinstance(settled.exception(), RuntimeError)
            assert str(settled.exception()) == "async finalizer create_task failed"
        _assert_pins_released(agent, thread_id)
    finally:
        stream.finish()
        await stream._await_pending_finalizers()
        agent.remove_extension("scheduler-pin")
        assert "scheduler-pin" not in agent._pending_extensions
        await agent.aclose()
