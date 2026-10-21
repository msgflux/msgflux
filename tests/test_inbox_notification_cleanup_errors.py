"""Notification cleanup failures must not replace the delivery failure."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.exceptions import TaskInterruptRequestedError, TaskPauseRequestedError
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime import ExecutionScope
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure_kind", ["hook", "pause", "cancel"])
def test_notification_release_failure_preserves_primary_error(
    tmp_path, monkeypatch, asynchronous, failure_kind
):
    checkpoints = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    store = SQLiteAgentInboxStore(tmp_path / "inbox.sqlite3")
    scope = ExecutionScope(namespace="agent", thread_id="thread", run_id="run")
    inbox = AgentInbox(store=store, namespace="agent", thread_id="thread", run_id="run")
    notification = inbox.user_message("pending input")
    primary = {
        "hook": ValueError("notification transform failed"),
        "pause": TaskPauseRequestedError("run", "notification transform paused"),
        "cancel": asyncio.CancelledError("notification transform cancelled"),
    }[failure_kind]
    cleanup = OSError("cannot release inbox lease")

    def transform(_context):
        raise primary

    def fail_release(*_args, **_kwargs):
        raise cleanup

    monkeypatch.setattr(store, "release_notifications", fail_release)
    agent = Agent(
        name="agent",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=checkpoints,
        agent_inbox=inbox,
        hooks=[Hook(event="transform_notifications", handler=transform)],
    )
    agent.generator.forward = Mock(side_effect=AssertionError("must not request"))
    agent.generator.aforward = AsyncMock(side_effect=AssertionError("must not request"))
    try:
        with pytest.raises(type(primary)) as caught:
            if asynchronous:
                asyncio.run(agent.acall("start", scope=scope))
            else:
                agent("start", scope=scope)
        assert caught.value is primary
        assert any("cannot release inbox lease" in note for note in primary.__notes__)
        agent.generator.forward.assert_not_called()
        agent.generator.aforward.assert_not_called()
        state = checkpoints.load_state("agent", "thread", "run")
        assert (
            state["status"]
            == {"hook": "failed", "pause": "paused", "cancel": "interrupted"}[
                failure_kind
            ]
        )
        assert [item.notification_id for item in inbox.peek()] == [
            notification.notification_id
        ]
    finally:
        asyncio.run(agent.aclose())
        checkpoints.close()
        store.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("command", ["pause", "interrupt"])
def test_control_release_failure_preserves_requested_outcome(
    tmp_path, monkeypatch, asynchronous, command
):
    checkpoints = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    store = SQLiteAgentInboxStore(tmp_path / "inbox.sqlite3")
    scope = ExecutionScope(namespace="agent", thread_id="thread", run_id="run")
    inbox = AgentInbox(store=store, namespace="agent", thread_id="thread", run_id="run")
    pending = inbox.user_message("retain this input")
    control = inbox.control(command, reason="operator requested control")

    def fail_release(*_args, **_kwargs):
        raise OSError("cannot release the other inbox claims")

    monkeypatch.setattr(store, "release_notifications", fail_release)
    agent = Agent(
        name="agent",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=checkpoints,
        agent_inbox=inbox,
    )
    agent.generator.forward = Mock(side_effect=AssertionError("must not request"))
    agent.generator.aforward = AsyncMock(side_effect=AssertionError("must not request"))
    expected = (
        TaskPauseRequestedError if command == "pause" else TaskInterruptRequestedError
    )
    try:
        with pytest.raises(expected) as caught:
            if asynchronous:
                asyncio.run(agent.acall("start", scope=scope))
            else:
                agent("start", scope=scope)
        assert any(
            "cannot release the other inbox claims" in note
            for note in caught.value.__notes__
        )
        state = checkpoints.load_state("agent", "thread", "run")
        assert state["status"] == ("paused" if command == "pause" else "interrupted")
        assert (
            control.notification_id in state["messages"]["metadata"]["inbox_receipts"]
        )
        assert pending.notification_id not in state["messages"]["metadata"].get(
            "inbox_receipts", ()
        )
        assert [item.notification_id for item in inbox.peek()] == [
            pending.notification_id
        ]
        agent.generator.forward.assert_not_called()
        agent.generator.aforward.assert_not_called()
    finally:
        asyncio.run(agent.aclose())
        checkpoints.close()
        store.close()
