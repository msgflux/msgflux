import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.models.response import ModelStreamResponse
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore


@pytest.mark.asyncio
async def test_async_stream_finalizer_runs_hooks_and_persists_checkpoint():
    response = ModelStreamResponse(mode="async")
    response.set_response_type("text_generation")
    response.add("answer")
    response.finish()
    model = type("Model", (), {"model_type": "chat_completion"})()
    model.acall = AsyncMock(return_value=response)
    store = InMemoryCheckpointStore()
    seen = []

    async def before(context):
        seen.append(("before", context.outcome, context.messages))
        return context

    async def after(context):
        seen.append(("after", context.outcome, context.messages))
        return context

    agent = Agent(
        name="agent",
        model=model,
        checkpoint_store=store,
        config={"stream": True},
        hooks=[
            Hook(event="before_run_end", handler=before),
            Hook(event="after_run_end", handler=after),
        ],
    )
    events = [event async for event in agent.stream_events("question")]

    assert events[-1].type == "run.end"
    assert [item[0] for item in seen] == ["before", "after"]
    state = store.load_state("agent", events[0].data["thread_id"], events[0].run_id)
    assert state["status"] == "completed"
    history = ChatMessages()
    history._hydrate_state(state["messages"])
    assert "answer" in str(history)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_hook", [False, True])
async def test_worker_finished_stream_commits_without_consumer_and_releases(fail_hook):
    response = ModelStreamResponse()
    response.set_response_type("text_generation")
    model = type("Model", (), {"model_type": "chat_completion"})()
    model.acall = AsyncMock(return_value=response)
    store = InMemoryCheckpointStore()
    messages = ChatMessages()
    before_seen = []

    async def before(context):
        before_seen.append(str(context.messages))
        if fail_hook:
            raise ValueError("bad terminal hook")
        updated = context.messages.copy()
        updated.add_assistant_response("hook annotation")
        return replace(context, messages=updated)

    agent = Agent(
        name="agent",
        model=model,
        checkpoint_store=store,
        config={"stream": True},
        hooks=[Hook(event="before_run_end", handler=before)],
    )
    await agent.acall(
        "question",
        messages=messages,
        scope=ExecutionScope(thread_id="worker-thread", run_id="worker-run"),
    )
    response.add("answer")
    await asyncio.to_thread(response.finish)
    if fail_hook:
        with pytest.raises(Exception, match="bad terminal hook"):
            await response._await_pending_finalizers()
    else:
        await response._await_pending_finalizers()
    assert "answer" in before_seen[0]
    state = store.load_state("agent", "worker-thread", "worker-run")
    assert state["status"] == ("failed" if fail_hook else "completed")
    assert "answer" in str(state["messages"])
    if not fail_hook:
        assert "hook annotation" in str(state["messages"])
    messages._ensure_stream_available()


@pytest.mark.asyncio
async def test_stream_finalizer_uses_scope_checkpoint_store():
    response = ModelStreamResponse(mode="async")
    response.set_response_type("text_generation")
    response.add("answer")
    response.finish()
    model = type("Model", (), {"model_type": "chat_completion"})()
    model.acall = AsyncMock(return_value=response)
    store = InMemoryCheckpointStore()
    agent = Agent(
        name="agent",
        model=model,
        checkpoint_store=store,
        config={"stream": True},
    )

    async for _ in agent.stream_events(
        "question",
        scope=ExecutionScope(thread_id="thread", run_id="run"),
    ):
        pass

    assert store.load_state("agent", "thread", "run")["status"] == "completed"


def _terminal_hook_case(kind):
    errors = {
        "interrupt": TaskInterruptRequestedError("task", "terminal interruption"),
        "pause": TaskPauseRequestedError("task", "terminal pause"),
        "abort": AbortRequestedError("terminal abort"),
        "cancel": asyncio.CancelledError("terminal cancellation"),
        "error": ValueError("terminal failure"),
    }
    error = errors[kind]
    status = (
        "paused" if kind == "pause" else "failed" if kind == "error" else "interrupted"
    )
    return error, status


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["interrupt", "pause", "abort", "cancel", "error"])
async def test_stream_precommit_control_preserves_checkpoint_outcome(
    asynchronous, kind
):
    error, status = _terminal_hook_case(kind)
    seen = []

    def stop(ctx):
        seen.append(ctx.outcome)
        raise error

    stream = ModelStreamResponse()
    stream.set_response_type("text_generation")
    store = InMemoryCheckpointStore()
    messages = ChatMessages()
    agent = Agent(
        name="stream-control",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=store,
        config={"stream": True},
        hooks=[Hook(event="before_run_end", handler=stop)],
    )
    agent.generator.forward = Mock(return_value=stream)
    agent.generator.aforward = AsyncMock(return_value=stream)
    scope = ExecutionScope(thread_id="thread", run_id="run")
    try:
        if asynchronous:
            await agent.acall("question", messages=messages, scope=scope)
        else:
            agent("question", messages=messages, scope=scope)
        stream.add("answer")
        if asynchronous:
            stream.finish()
            with pytest.raises(type(error)):
                async for _ in stream.consume():
                    pass
        else:
            with pytest.raises(type(error)) as raised:
                stream.finish()
            assert raised.value is error
        state = store.load_state("stream-control", "thread", "run")
        assert state["status"] == status
        assert seen == ["completed"]
        assert "answer" in str(state["messages"])
        assert messages.get_active_turn() is None
        messages._ensure_stream_available()
        assert messages._to_state() == state["messages"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["interrupt", "pause", "abort", "cancel", "error"])
async def test_service_stream_precommit_control_matches_checkpoint(kind):
    error, expected = _terminal_hook_case(kind)
    seen = []

    def stop(ctx):
        seen.append(ctx.outcome)
        raise error

    stream = ModelStreamResponse(mode="async")
    stream.set_response_type("text_generation")
    stream.add("answer")
    stream.finish()
    checkpoints = InMemoryCheckpointStore()
    agent = Agent(
        name="stream-service",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=checkpoints,
        config={"stream": True},
        hooks=[Hook(event="before_run_end", handler=stop)],
    )
    agent.generator.aforward = AsyncMock(return_value=stream)
    journal = SQLiteServiceStore()
    service = AgentService(store=journal)
    service.register(
        "main", lambda _: AgentSession(agent, checkpoint_store=checkpoints)
    )
    try:
        thread = await service.open_thread("main", thread_id="stream-thread")
        receipt = await service.prompt(
            thread.thread_id, "question", request_id="request"
        )
        settled = await asyncio.wait_for(
            service.wait(thread.thread_id, receipt.request_id), 3
        )
        state = checkpoints.load_state(
            "stream-service", thread.thread_id, receipt.run_id
        )
        assert settled.status == state["status"] == expected
        assert seen == ["completed"]
        assert "answer" in str(state["messages"])
        assert (
            state["messages"]["items"][-1]["event"]
            == {"paused": "pause", "failed": "fail", "interrupted": "interrupt"}[
                expected
            ]
        )
    finally:
        await service.aclose()
        journal.close()
        await agent.aclose()
