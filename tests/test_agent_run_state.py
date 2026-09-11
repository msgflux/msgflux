"""Run state is restored before hooks and isolated between executions."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.agent_run import get_agent_run
from msgflux.runtime.context import ExecutionScope


class Counter(AgentExtension):
    def __init__(self):
        super().__init__("counter")
        self.restored = []

    def hooks(self):
        return (
            Hook(event="before_run", handler=self.increment),
            Hook(event="before_resume", handler=self.restore),
        )

    def increment(self, ctx):
        state = self.durable_state()
        state["count"] = state.get("count", 0) + 1

    def restore(self, ctx):
        self.restored.append(self.durable_state().get("count"))
        self.increment(ctx)


def answer():
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("ok")
    return response


def make_agent(store, extension):
    agent = Agent(
        name="state",
        model=Mock(model_type="chat_completion"),
        extensions=[extension],
        checkpoint_store=store,
    )
    agent.generator.forward = Mock(return_value=answer())
    agent.generator.aforward = AsyncMock(return_value=answer())
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_durable_extension_restored_before_resume(asynchronous):
    store = InMemoryCheckpointStore()
    first = make_agent(store, Counter())
    scope = ExecutionScope(
        thread_id="thread", run_id="run", parent_run_id="parent", root_run_id="root"
    )
    first.generator.forward = Mock(side_effect=RuntimeError("retry later"))
    first.generator.aforward = AsyncMock(side_effect=RuntimeError("retry later"))
    with pytest.raises(RuntimeError, match="retry later"):
        if asynchronous:
            await first.acall("start", scope=scope)
        else:
            first("start", scope=scope)
    extension = Counter()
    second = make_agent(store, extension)
    if asynchronous:
        await second.acall(scope=scope)
    else:
        second(scope=scope)
    assert extension.restored == [1]
    state = store.load_state("state", "thread", "run")
    assert state["runtime"]["extensions"]["counter"]["count"] == 2
    assert state["_checkpoint"]["extensions"]["counter"]["count"] == 2
    assert state["scope"]["parent_run_id"] == "parent"
    assert get_agent_run() is None


@pytest.mark.asyncio
async def test_concurrent_runs_have_separate_durable_state():
    store = InMemoryCheckpointStore()
    extension = Counter()
    agent = make_agent(store, extension)
    await asyncio.gather(
        *[
            agent.acall(
                "start", scope=ExecutionScope(thread_id="thread", run_id=f"run-{i}")
            )
            for i in range(3)
        ]
    )
    for i in range(3):
        state = store.load_state("state", "thread", f"run-{i}")
        assert state["runtime"]["extensions"]["counter"]["count"] == 1


def test_fork_resume_commits_only_to_target_run_and_preserves_lineage():
    store = InMemoryCheckpointStore()
    agent = make_agent(store, Counter())
    agent("start", scope=ExecutionScope(thread_id="thread", run_id="source"))
    source = store.load_state("state", "thread", "source")
    forked = store.fork_run(
        "state",
        "thread",
        "source",
        target_thread_id="fork-thread",
        target_run_id="fork",
        status="running",
    )
    assert forked["runtime"]["run_id"] == "fork"
    agent(scope=ExecutionScope(thread_id="fork-thread", run_id="fork"))
    assert store.load_state("state", "thread", "source") == source
    target = store.load_state("state", "fork-thread", "fork")
    assert target["status"] == "completed"
    assert target["_checkpoint"]["fork_of"]["run_id"] == "source"
    assert target["runtime"]["parent_run_id"] == "source"
