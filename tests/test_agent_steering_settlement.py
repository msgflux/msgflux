"""Steering admission closes whenever an Agent settles a terminal outcome."""

import asyncio
from functools import partial
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.exceptions import TaskInterruptRequestedError, TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.agent_inbox import AgentInbox, AgentInboxClosedError
from msgflux.runtime.agent_inbox import SQLiteAgentInboxStore
from msgflux.runtime.context import ExecutionScope, get_execution_context


def _scope():
    return ExecutionScope(
        namespace="settlement_agent", thread_id="thread", run_id="run"
    )


def _response(content):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _invoke(agent, asynchronous, message):
    if asynchronous:
        return asyncio.run(agent.acall(message, scope=_scope()))
    return agent(message, scope=_scope())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", ["failed", "paused", "interrupted"])
def test_terminal_settlement_seals_steering_and_resume_reopens_validated_run(
    tmp_path, asynchronous, outcome
):
    checkpoints = SQLiteCheckpointStore(tmp_path / f"checkpoints-{asynchronous}.sqlite")
    inbox_store = SQLiteAgentInboxStore(tmp_path / f"inbox-{asynchronous}.sqlite")
    inbox = AgentInbox(store=inbox_store)
    agent = Agent(
        name="settlement_agent",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=checkpoints,
        agent_inbox=inbox,
    )
    primary = {
        "failed": OSError("model failed"),
        "paused": TaskPauseRequestedError("run", "model paused"),
        "interrupted": TaskInterruptRequestedError("run", "model interrupted"),
    }[outcome]

    def fail_first(**_kwargs):
        get_execution_context()["agent_inbox"].user_message(
            "accepted before settlement"
        )
        raise primary

    async def afail_first(**kwargs):
        return fail_first(**kwargs)

    if asynchronous:
        agent.generator.aforward = AsyncMock(side_effect=afail_first)
    else:
        agent.generator.forward = Mock(side_effect=fail_first)

    target_inbox = inbox.fork(
        namespace="settlement_agent", thread_id="thread", run_id="run"
    )

    invoke = partial(_invoke, agent, asynchronous)

    try:
        with pytest.raises(type(primary)) as caught:
            invoke("start")
        assert caught.value is primary
        assert (
            checkpoints.load_state("settlement_agent", "thread", "run")["status"]
            == outcome
        )
        with pytest.raises(AgentInboxClosedError):
            target_inbox.user_message("too late")
        assert (
            len(inbox_store.load_notifications("settlement_agent", "thread", "run"))
            == 1
        )

        if outcome == "interrupted":
            with pytest.raises(ValueError, match="already reached terminal status"):
                invoke("resume")
            return

        resumed_requests = []

        def resume(**kwargs):
            resumed_requests.append(kwargs["messages"].to_chatml())
            if len(resumed_requests) == 1:
                assert "accepted before settlement" in str(resumed_requests[0])
                get_execution_context()["agent_inbox"].user_message(
                    "accepted after resume"
                )
                return _response("answer before pending steering")
            return _response("final resumed answer")

        async def aresume(**kwargs):
            return resume(**kwargs)

        if asynchronous:
            agent.generator.aforward = AsyncMock(side_effect=aresume)
            assert invoke("resume") == "final resumed answer"
        else:
            agent.generator.forward = Mock(side_effect=resume)
            assert invoke("resume") == "final resumed answer"

        assert len(resumed_requests) == 2
        assert "accepted before settlement" in str(resumed_requests[1])
        assert "accepted after resume" in str(resumed_requests[1])
        completed = checkpoints.load_state("settlement_agent", "thread", "run")
        assert completed["status"] == "completed"
        assert (
            str(completed["messages"]["items"]).count("accepted before settlement") == 1
        )
        with pytest.raises(AgentInboxClosedError):
            target_inbox.user_message("too late after resume")
    finally:
        asyncio.run(agent.aclose())
        checkpoints.close()
        inbox_store.close()
