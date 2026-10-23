"""A run-scoped control observed at completion must not reach the next run."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.exceptions import TaskInterruptRequestedError, TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    InMemoryAgentInboxStore,
    SQLiteAgentInboxStore,
)
from msgflux.runtime.context import ExecutionScope, get_execution_context


def _response(content):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _scope(run_id):
    return ExecutionScope(namespace="late_control", thread_id="thread", run_id=run_id)


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("command", "error_type", "status"),
    [
        ("pause", TaskPauseRequestedError, "paused"),
        ("interrupt", TaskInterruptRequestedError, "interrupted"),
    ],
)
def test_control_published_during_final_model_call_stops_that_run_only(
    store_kind,
    asynchronous,
    command,
    error_type,
    status,
    tmp_path,
):
    checkpoints = SQLiteCheckpointStore(
        tmp_path / f"checkpoints-{store_kind}-{asynchronous}-{command}.sqlite"
    )
    if store_kind == "memory":
        inbox_store = InMemoryAgentInboxStore()
    else:
        inbox_store = SQLiteAgentInboxStore(
            tmp_path / f"inbox-{store_kind}-{asynchronous}-{command}.sqlite"
        )
    calls = []

    def answer(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            get_execution_context()["agent_inbox"].control(
                command, reason="control arrived during the final response"
            )
        return _response("first" if len(calls) == 1 else "second")

    async def aanswer(**kwargs):
        return answer(**kwargs)

    agent = Agent(
        name="late_control",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=checkpoints,
        agent_inbox=AgentInbox(store=inbox_store),
    )
    agent.generator.forward = Mock(side_effect=answer)
    agent.generator.aforward = AsyncMock(side_effect=aanswer)

    def invoke(run_id):
        if asynchronous:
            return asyncio.run(agent.acall("start", scope=_scope(run_id)))
        return agent("start", scope=_scope(run_id))

    try:
        with pytest.raises(error_type):
            invoke("owning-run")
        assert len(calls) == 1
        owning = checkpoints.load_state("late_control", "thread", "owning-run")
        assert owning["status"] == status
        assert (
            inbox_store.load_notifications("late_control", "thread", "owning-run") == []
        )

        assert invoke("next-run") == "second"
        assert len(calls) == 2
        following = checkpoints.load_state("late_control", "thread", "next-run")
        assert following["status"] == "completed"
    finally:
        asyncio.run(agent.aclose())
        checkpoints.close()
        if store_kind == "sqlite":
            inbox_store.close()
