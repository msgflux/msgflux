"""Inbox cleanup errors must not hide checkpoint revision conflicts."""

from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import CheckpointConflictError, InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime import ExecutionScope
from msgflux.runtime.agent_inbox import AgentInbox, InMemoryAgentInboxStore


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_inbox_release_failure_preserves_checkpoint_conflict(asynchronous):
    primary = CheckpointConflictError("stale checkpoint revision")
    cause = RuntimeError("competing writer")
    primary.__cause__ = cause
    cleanup = OSError("inbox lease release unavailable")
    checkpoints = InMemoryCheckpointStore()
    checkpoints.commit_state = Mock(side_effect=primary)
    inbox_store = InMemoryAgentInboxStore()
    release = inbox_store.release_notifications
    inbox_store.release_notifications = Mock(side_effect=cleanup)
    inbox = AgentInbox(
        store=inbox_store, owner="agent", thread_id="thread", run_id="run"
    )
    notification = inbox.user_message("pending instruction")
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("answer")
    agent = Agent(
        name="agent",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=checkpoints,
        agent_inbox=inbox,
    )
    agent.generator.forward = Mock(return_value=response)
    agent.generator.aforward = AsyncMock(return_value=response)
    scope = ExecutionScope(thread_id="thread", run_id="run")
    try:
        with pytest.raises(CheckpointConflictError) as raised:
            if asynchronous:
                await agent.acall("question", scope=scope)
            else:
                agent("question", scope=scope)
        assert raised.value is primary
        assert primary.__cause__ is cause
        assert any(
            "inbox lease release unavailable" in note for note in primary.__notes__
        )
        checkpoints.commit_state.assert_called_once()
        inbox_store.release_notifications.assert_called_once()
        assert checkpoints.load_state("agent", "thread", "run") is None
        # A failed release must leave the message in the store, without ack.
        assert inbox.peek()[0].notification_id == notification.notification_id
        call = inbox_store.release_notifications.call_args
        inbox_store.release_notifications = release
        release(*call.args, **call.kwargs)
        assert inbox.claim()[0].notification_id == notification.notification_id
        inbox.release()
    finally:
        inbox_store.release_notifications = release
        await agent.aclose()
