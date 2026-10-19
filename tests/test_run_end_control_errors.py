"""Pre-commit hook control signals retain their durable outcome."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime import ExecutionScope
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore


def _case(kind):
    if kind == "abort":
        return (
            AbortRequestedError("end marker"),
            TaskInterruptRequestedError,
            "interrupted",
        )
    if kind == "interrupt":
        return (
            TaskInterruptRequestedError("task", "end marker"),
            TaskInterruptRequestedError,
            "interrupted",
        )
    if kind == "pause":
        return (
            TaskPauseRequestedError("task", "end marker"),
            TaskPauseRequestedError,
            "paused",
        )
    if kind == "cancel":
        return (
            asyncio.CancelledError("end marker"),
            asyncio.CancelledError,
            "interrupted",
        )
    return ValueError("end marker"), ValueError, "failed"


def _agent(store, error):
    before, after = [], []

    def stop(ctx):
        before.append(ctx.outcome)
        raise error

    def observe(ctx):
        after.append(ctx.outcome)

    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("model output")
    agent = Agent(
        name="end-control",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=store,
        hooks=[
            Hook(event="before_run_end", handler=stop),
            Hook(event="after_run_end", handler=observe),
        ],
    )
    agent.generator.forward = Mock(return_value=response)
    agent.generator.aforward = AsyncMock(return_value=response)
    return agent, before, after


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["abort", "interrupt", "pause", "cancel", "error"])
async def test_precommit_hook_control_is_settled_without_reentry(asynchronous, kind):
    error, expected_type, status = _case(kind)
    original_cause = RuntimeError("original hook cause")
    error.__cause__ = original_cause
    store = InMemoryCheckpointStore()
    agent, before, after = _agent(store, error)
    scope = ExecutionScope(thread_id="thread", run_id="run")
    try:
        with pytest.raises(expected_type) as raised:
            if asynchronous:
                await agent.acall("hello", scope=scope)
            else:
                agent("hello", scope=scope)
        if kind == "abort":
            assert raised.value.__cause__ is error
        else:
            assert raised.value is error
        assert error.__cause__ is original_cause
        assert before == ["completed"]
        assert after == [status]
        assert store.load_state("end-control", "thread", "run")["status"] == status
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["abort", "interrupt", "pause", "error"])
async def test_service_receipt_matches_precommit_hook_checkpoint(kind):
    error, _, status = _case(kind)
    checkpoints = InMemoryCheckpointStore()
    agent, before, after = _agent(checkpoints, error)
    journal = SQLiteServiceStore()
    service = AgentService(store=journal)
    service.register(
        "main", lambda _: AgentSession(agent, checkpoint_store=checkpoints)
    )
    thread = await service.open_thread("main", thread_id="hook-control")
    try:
        receipt = await service.prompt(thread.thread_id, "hello", request_id="request")
        settled = await asyncio.wait_for(
            service.wait(thread.thread_id, receipt.request_id), 3
        )
        state = checkpoints.load_state("end-control", thread.thread_id, receipt.run_id)
        assert settled.status == state["status"] == status
        assert before == ["completed"]
        assert after == [status]
    finally:
        await service.aclose()
        journal.close()
        await agent.aclose()


@pytest.mark.asyncio
async def test_sync_call_cancellation_records_interrupted_checkpoint():
    store = InMemoryCheckpointStore()
    agent = Agent(
        name="sync-cancel",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=store,
    )
    error = asyncio.CancelledError("model cancelled")
    agent.generator.forward = Mock(side_effect=error)
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            agent("hello", scope=ExecutionScope(thread_id="thread", run_id="run"))
        assert raised.value is error
        assert (
            store.load_state("sync-cancel", "thread", "run")["status"] == "interrupted"
        )
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_cancelling_pending_precommit_hook_settles_without_reentry():
    entered, cleaned = asyncio.Event(), asyncio.Event()
    before, after = [], []

    async def pending(ctx):
        before.append(ctx.outcome)
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    store = InMemoryCheckpointStore()
    agent = Agent(
        name="cancel-hook",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=store,
        hooks=[
            Hook(event="before_run_end", handler=pending),
            Hook(event="after_run_end", handler=lambda ctx: after.append(ctx.outcome)),
        ],
    )
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("model output")
    agent.generator.aforward = AsyncMock(return_value=response)
    call = asyncio.create_task(
        agent.acall("hello", scope=ExecutionScope(thread_id="thread", run_id="run"))
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        call.cancel("cancel precommit")
        with pytest.raises(asyncio.CancelledError, match="cancel precommit"):
            await asyncio.wait_for(call, 3)
        assert cleaned.is_set()
        assert before == ["completed"]
        assert after == ["interrupted"]
        assert (
            store.load_state("cancel-hook", "thread", "run")["status"] == "interrupted"
        )
    finally:
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)
        await agent.aclose()
