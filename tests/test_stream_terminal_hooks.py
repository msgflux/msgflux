from unittest.mock import AsyncMock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelStreamResponse
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.context import ExecutionScope


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
