"""Inbox control receipts become durable only with their terminal checkpoint."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskInterruptRequestedError, TaskPauseRequestedError
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime import ExecutionScope
from msgflux.runtime.agent_inbox import AgentInbox, InMemoryAgentInboxStore


class FailControlCheckpointOnce(InMemoryCheckpointStore):
    def __init__(self, status):
        super().__init__()
        self.status = status
        self.failure = OSError(f"{status} checkpoint rejected")
        self.fail = True

    def commit_state(self, namespace, thread_id, run_id, state, **kwargs):
        if state.get("status") == self.status and self.fail:
            self.fail = False
            raise self.failure
        return super().commit_state(namespace, thread_id, run_id, state, **kwargs)


@pytest.mark.parametrize(
    ("command", "status", "error_type"),
    [
        ("pause", "paused", TaskPauseRequestedError),
        ("interrupt", "interrupted", TaskInterruptRequestedError),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_control_receipt_is_not_published_until_terminal_checkpoint(
    command, status, error_type, asynchronous
):
    checkpoints = FailControlCheckpointOnce(status)
    scope = ExecutionScope(namespace="agent", thread_id="thread", run_id="run")
    inbox = AgentInbox(
        store=InMemoryAgentInboxStore(),
        namespace="agent",
        thread_id="thread",
        run_id="run",
    )
    notification = inbox.control(command, reason="operator requested control")
    history = ChatMessages()
    model = Mock(model_type="chat_completion")
    agent = Agent(
        name="agent",
        model=model,
        checkpoint_store=checkpoints,
        agent_inbox=inbox,
    )

    try:
        with pytest.raises(OSError) as rejected:
            if asynchronous:
                asyncio.run(agent.acall("work", messages=history, scope=scope))
            else:
                agent("work", messages=history, scope=scope)
        assert rejected.value is checkpoints.failure
        assert checkpoints.load_state("agent", "thread", "run") is None
        assert history.get_active_turn() is not None
        assert notification.notification_id not in history.metadata.get(
            "inbox_receipts", ()
        )
        assert [item.notification_id for item in inbox.peek()] == [
            notification.notification_id
        ]
        assert model.call_count == 0

        with pytest.raises(error_type):
            if asynchronous:
                asyncio.run(agent.acall("retry", messages=history, scope=scope))
            else:
                agent("retry", messages=history, scope=scope)

        saved = checkpoints.load_state("agent", "thread", "run")
        assert saved["status"] == status
        assert [
            item.get("event")
            for item in saved["messages"]["items"]
            if item.get("type") == "turn"
        ] == ["start", "pause" if status == "paused" else "interrupt"]
        assert (
            notification.notification_id
            in saved["messages"]["metadata"]["inbox_receipts"]
        )
        assert len(history.turns) == 1
        assert history.turns[0]["status"] == status
        assert model.call_count == 0
        assert inbox.peek() == []
    finally:
        asyncio.run(agent.aclose())


@pytest.mark.asyncio
async def test_cancelled_terminal_hook_releases_pause_for_immediate_retry():
    checkpoints = InMemoryCheckpointStore()
    scope = ExecutionScope(namespace="agent", thread_id="thread", run_id="run")
    inbox = AgentInbox(
        store=InMemoryAgentInboxStore(),
        namespace="agent",
        thread_id="thread",
        run_id="run",
    )
    notification = inbox.pause(reason="operator requested pause")
    history = ChatMessages()
    entered, release = asyncio.Event(), asyncio.Event()
    hook_calls = 0

    async def wait_once(_context):
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls == 1:
            entered.set()
            await release.wait()

    agent = Agent(
        name="agent",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=checkpoints,
        agent_inbox=inbox,
        hooks=[Hook(event="before_run_end", handler=wait_once)],
    )
    agent.generator.aforward = AsyncMock(
        side_effect=AssertionError("pending pause must prevent model execution")
    )
    task = asyncio.create_task(agent.acall("work", messages=history, scope=scope))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert checkpoints.load_state("agent", "thread", "run") is None
        assert notification.notification_id not in history.metadata.get(
            "inbox_receipts", ()
        )
        assert history.get_active_turn() is not None

        with pytest.raises(TaskPauseRequestedError):
            await asyncio.wait_for(
                agent.acall("retry", messages=history, scope=scope), 3
            )
        saved = checkpoints.load_state("agent", "thread", "run")
        assert saved["status"] == "paused"
        assert (
            notification.notification_id
            in saved["messages"]["metadata"]["inbox_receipts"]
        )
        assert len(history.turns) == 1
        assert history.turns[0]["status"] == "paused"
        assert inbox.peek() == []
        agent.generator.aforward.assert_not_called()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await agent.aclose()


def test_control_hook_failure_releases_claim_without_publishing_receipt():
    checkpoints = InMemoryCheckpointStore()
    scope = ExecutionScope(namespace="agent", thread_id="thread", run_id="run")
    inbox = AgentInbox(
        store=InMemoryAgentInboxStore(),
        namespace="agent",
        thread_id="thread",
        run_id="run",
    )
    notification = inbox.pause(reason="operator requested pause")
    history = ChatMessages()
    model = Mock(model_type="chat_completion")
    hook_error = TaskInterruptRequestedError("test", "secondary hook interrupt")
    hook_calls = 0

    def interrupt_once(_context):
        nonlocal hook_calls
        hook_calls += 1
        if hook_calls == 1:
            raise hook_error

    agent = Agent(
        name="agent",
        model=model,
        checkpoint_store=checkpoints,
        agent_inbox=inbox,
        hooks=[Hook(event="before_run_end", handler=interrupt_once)],
    )

    try:
        with pytest.raises(TaskInterruptRequestedError) as first:
            agent("work", messages=history, scope=scope)
        assert first.value is hook_error
        assert checkpoints.load_state("agent", "thread", "run") is None
        assert notification.notification_id not in history.metadata.get(
            "inbox_receipts", ()
        )
        assert history.get_active_turn() is not None

        with pytest.raises(TaskPauseRequestedError):
            agent("retry", messages=history, scope=scope)

        saved = checkpoints.load_state("agent", "thread", "run")
        assert saved["status"] == "paused"
        assert [
            item.get("event")
            for item in saved["messages"]["items"]
            if item.get("type") == "turn"
        ] == ["start", "pause"]
        assert (
            notification.notification_id
            in saved["messages"]["metadata"]["inbox_receipts"]
        )
        assert len(history.turns) == 1
        assert history.turns[0]["status"] == "paused"
        assert model.call_count == 0
        assert inbox.peek() == []
    finally:
        asyncio.run(agent.aclose())
