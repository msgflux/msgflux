"""Task-addressed message delivery across run and process boundaries."""

from __future__ import annotations

import pytest

from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    InMemoryAgentInboxStore,
    SQLiteAgentInboxStore,
)
from msgflux.tasks import InMemoryTaskStore, SQLiteTaskStore, TaskHandle


def test_store_routing_ids_survive_sqlite_reopen_not_memory_replacement(tmp_path):
    memory_a = InMemoryCheckpointStore()
    memory_b = InMemoryCheckpointStore()
    assert memory_a.routing_id != memory_b.routing_id

    path = str(tmp_path / "checkpoints.sqlite")
    first = SQLiteCheckpointStore(path)
    routing_id = first.routing_id
    first.close()
    reopened = SQLiteCheckpointStore(path)
    try:
        assert reopened.routing_id == routing_id
    finally:
        reopened.close()


@pytest.mark.parametrize("durable", [False, True])
def test_task_message_replayed_to_new_run_until_acknowledged(tmp_path, durable):
    if durable:
        tasks = SQLiteTaskStore(str(tmp_path / "tasks.sqlite"))
        inbox_store = SQLiteAgentInboxStore(str(tmp_path / "inbox.sqlite"))
    else:
        tasks = InMemoryTaskStore()
        inbox_store = InMemoryAgentInboxStore()
    try:
        task = tasks.create("agent", task_id="child")
        assert tasks.enqueue_message(task.task_id, "msg-1", "Keep going")
        handle = TaskHandle(task.task_id, tasks)
        old_inbox = AgentInbox(
            owner="agent",
            store=inbox_store,
            namespace="agent",
            thread_id="thread",
            run_id="old",
        )
        new_inbox = old_inbox.fork(run_id="new")

        handle.forward_messages(old_inbox)
        handle.forward_messages(old_inbox)
        assert len(old_inbox.peek()) == 1
        assert old_inbox.peek()[0].notification_id == "msg-1"

        # An old worker may stop before checkpointing the notification. The
        # task-addressed row remains and the next run can safely replay it.
        handle.forward_messages(new_inbox)
        assert [item.notification_id for item in new_inbox.peek()] == ["msg-1"]
        handle.ack_messages(["msg-1"])
        assert tasks.pending_messages(task.task_id) == []
    finally:
        if durable:
            tasks.close()
            inbox_store.close()


def test_sqlite_task_message_survives_reopen_before_forwarding(tmp_path):
    path = str(tmp_path / "tasks.sqlite")
    first = SQLiteTaskStore(path)
    first.create("agent", task_id="child")
    assert first.enqueue_message("child", "msg-1", "After restart")
    first.close()

    reopened = SQLiteTaskStore(path)
    inbox_store = InMemoryAgentInboxStore()
    try:
        inbox = AgentInbox(
            owner="agent",
            store=inbox_store,
            namespace="agent",
            thread_id="thread",
            run_id="new",
        )
        TaskHandle("child", reopened).forward_messages(inbox)
        assert inbox.peek()[0].metadata["message"] == "After restart"
    finally:
        reopened.close()


def test_sqlite_resume_claim_is_atomic_across_store_instances(tmp_path):
    path = str(tmp_path / "tasks.sqlite")
    first = SQLiteTaskStore(path)
    second = SQLiteTaskStore(path)
    try:
        first.create(
            "agent",
            task_id="child",
            metadata={"checkpoint_run_id": "old"},
        )
        first.complete("child", "done")
        stale = second.get("child")
        claimed = first.requeue(
            "child",
            expected_status="completed",
            expected_generation=0,
            run_id="new",
        )
        assert claimed is not None
        assert claimed.metadata["checkpoint_run_id"] == "new"
        assert claimed.metadata["resume_generation"] == 1
        assert stale is not None
        assert (
            second.requeue(
                "child",
                expected_status=stale.status,
                expected_generation=stale.metadata.get("resume_generation", 0),
                run_id="racing-run",
            )
            is None
        )
        assert second.get("child").metadata["checkpoint_run_id"] == "new"
    finally:
        first.close()
        second.close()
