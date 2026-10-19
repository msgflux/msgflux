"""Post-commit observer cancellation cannot reopen a settled Agent run."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskInterruptRequestedError, TaskPauseRequestedError
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime import ExecutionScope
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore


def _make_agent(*, status="completed", streaming=False, observer=None):
    before, after = [], []
    store = InMemoryCheckpointStore()

    def observe(ctx):
        after.append(ctx.outcome)
        raise asyncio.CancelledError("observer cancelled")

    response = ModelStreamResponse(mode="async") if streaming else ModelResponse()
    response.set_response_type("text_generation")
    response.add("answer")
    if streaming:
        response.finish()
    agent = Agent(
        name="postcommit",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=store,
        config={"stream": streaming},
        hooks=[
            Hook(
                event="before_run_end", handler=lambda ctx: before.append(ctx.outcome)
            ),
            Hook(event="after_run_end", handler=observer or observe),
        ],
    )
    errors = {
        "failed": ValueError("primary model error"),
        "interrupted": TaskInterruptRequestedError("task", "primary interruption"),
        "paused": TaskPauseRequestedError("task", "primary pause"),
    }
    error = errors.get(status)
    agent.generator.forward = Mock(return_value=response, side_effect=error)
    agent.generator.aforward = AsyncMock(return_value=response, side_effect=error)
    return agent, store, before, after, error


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", ["completed", "failed", "interrupted", "paused"])
async def test_observer_cancellation_preserves_committed_outcome(asynchronous, status):
    agent, store, before, after, error = _make_agent(status=status)
    try:
        with pytest.warns(RuntimeWarning, match="after_run_end hook failed"):
            if error is None:
                output = (
                    await agent.acall(
                        "question",
                        scope=ExecutionScope(thread_id="thread", run_id="run"),
                    )
                    if asynchronous
                    else agent(
                        "question",
                        scope=ExecutionScope(thread_id="thread", run_id="run"),
                    )
                )
                assert output == "answer"
            else:
                with pytest.raises(type(error)) as raised:
                    if asynchronous:
                        await agent.acall(
                            "question",
                            scope=ExecutionScope(thread_id="thread", run_id="run"),
                        )
                    else:
                        agent(
                            "question",
                            scope=ExecutionScope(thread_id="thread", run_id="run"),
                        )
                assert raised.value is error
        assert store.load_state("postcommit", "thread", "run")["status"] == status
        assert before == after == [status]
        assert (
            agent.generator.aforward.call_count + agent.generator.forward.call_count
            == 1
        )
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_service_observer_cancellation_preserves_completed_receipt(streaming):
    agent, checkpoints, before, after, _ = _make_agent(streaming=streaming)
    journal = SQLiteServiceStore()
    service = AgentService(store=journal)
    service.register(
        "main", lambda _: AgentSession(agent, checkpoint_store=checkpoints)
    )
    try:
        thread = await service.open_thread("main", thread_id="thread")
        with pytest.warns(RuntimeWarning, match="after_run_end hook failed"):
            receipt = await service.prompt(
                thread.thread_id, "question", request_id="request"
            )
            settled = await asyncio.wait_for(
                service.wait(thread.thread_id, receipt.request_id), 3
            )
        state = checkpoints.load_state("postcommit", thread.thread_id, receipt.run_id)
        assert settled.status == state["status"] == "completed"
        assert before == after == ["completed"]
    finally:
        await service.aclose()
        journal.close()
        await agent.aclose()


@pytest.mark.asyncio
async def test_cancellation_during_observer_stops_hook_without_reopening_run():
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def observe(ctx):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    agent, store, before, _, _ = _make_agent(observer=observe)
    call = asyncio.create_task(
        agent.acall("question", scope=ExecutionScope(thread_id="thread", run_id="run"))
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert store.load_state("postcommit", "thread", "run")["status"] == "completed"
        with pytest.warns(RuntimeWarning, match="after_run_end hook failed"):
            call.cancel("observer cancellation")
            assert await asyncio.wait_for(call, 3) == "answer"
        assert cleaned.is_set()
        assert before == ["completed"]
        assert store.load_state("postcommit", "thread", "run")["status"] == "completed"
    finally:
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)
        await agent.aclose()
