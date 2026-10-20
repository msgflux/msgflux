"""Stream finalization seals steering on paused terminal settlement."""

from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelStreamResponse
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import NotificationContext
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    AgentInboxClosedError,
    SQLiteAgentInboxStore,
)
from msgflux.runtime.context import ExecutionScope, get_execution_context


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_filtered_stream_steering_pause_seals_gate(asynchronous, tmp_path):
    checkpoints = SQLiteCheckpointStore(tmp_path / f"checkpoints-{asynchronous}.sqlite")
    inbox_store = SQLiteAgentInboxStore(tmp_path / f"inbox-{asynchronous}.sqlite")
    scope = ExecutionScope(namespace="stream-pause", thread_id="thread", run_id="run")
    filtered = []

    def discard_user_message(context):
        assert isinstance(context, NotificationContext)
        filtered.extend(context.notifications)
        return replace(context, notifications=())

    def make_response():
        get_execution_context()["agent_inbox"].user_message(
            "accepted but filtered from model context"
        )
        response = ModelStreamResponse(mode="async" if asynchronous else "sync")
        response.set_response_type("text_generation")
        response.add("provider answer")
        response.finish()
        return response

    async def async_response(**_kwargs):
        return make_response()

    def sync_response(**_kwargs):
        return make_response()

    agent = Agent(
        name="stream-pause",
        model=Mock(model_type="chat_completion"),
        config={"stream": True},
        checkpoint_store=checkpoints,
        agent_inbox=AgentInbox(store=inbox_store),
        hooks=[Hook(event="transform_notifications", handler=discard_user_message)],
    )
    if asynchronous:
        agent.generator.aforward = AsyncMock(side_effect=async_response)
    else:
        agent.generator.forward = Mock(side_effect=sync_response)

    try:
        with pytest.raises(TaskPauseRequestedError):
            stream = (
                await agent.acall("start", scope=scope)
                if asynchronous
                else agent("start", scope=scope)
            )
            _ = [chunk async for chunk in stream.consume()]

        state = checkpoints.load_state("stream-pause", "thread", "run")
        assert state["status"] == "paused"
        assert filtered and len(filtered) == 1
        if asynchronous:
            assert agent.generator.aforward.await_count == 1
        else:
            assert agent.generator.forward.call_count == 1
        with pytest.raises(AgentInboxClosedError):
            agent.agent_inbox.fork(
                namespace="stream-pause", thread_id="thread", run_id="run"
            ).user_message("must remain rejected while paused")
        pending = inbox_store.load_notifications("stream-pause", "thread", "run")
        assert len(pending) == 1
        assert pending[0]["metadata"]["content"] == (
            "accepted but filtered from model context"
        )
    finally:
        await agent.aclose()
        checkpoints.close()
        inbox_store.close()
