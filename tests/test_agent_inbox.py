"""Tests for msgflux.runtime.agent_inbox."""

from shutil import copyfile
from unittest.mock import Mock

import pytest

from msgflux.data.stores import Store
from msgflux.nn import Agent
from msgflux.runtime.agent_inbox import (
    AgentControlMessage,
    AgentInbox,
    InMemoryAgentInboxStore,
    SQLiteAgentInboxStore,
)
from msgflux.runtime.context import ExecutionScope, execution_context


def _memory_inbox(**kwargs):
    return AgentInbox(store=InMemoryAgentInboxStore(), **kwargs)


def test_agent_inbox_verbose_publish_and_drain_are_printed(capsys):
    inbox = _memory_inbox(verbose=True, owner="assistant")

    inbox.publish(
        {
            "source": "task",
            "ref": "task_123",
            "status": "completed",
            "metadata": {"tool": "worker"},
        }
    )
    inbox.drain()

    captured = capsys.readouterr()
    assert "[assistant][notification_publish]" in captured.out
    assert '<notification source="task"' in captured.out
    assert 'ref="task_123"' in captured.out
    assert 'status="completed"' in captured.out
    assert 'tool="worker"' in captured.out
    assert "[assistant][notification_drain]" in captured.out
    assert "1 notification(s)" in captured.out


def test_agent_inbox_verbose_replace_is_printed(capsys):
    inbox = _memory_inbox(verbose=True, owner="assistant")

    inbox.publish(
        {
            "source": "tool_status",
            "ref": "task_123",
            "status": "prepare",
            "dedupe_key": "progress:task_123",
        }
    )
    inbox.publish(
        {
            "source": "tool_status",
            "ref": "task_123",
            "status": "process",
            "dedupe_key": "progress:task_123",
        }
    )

    captured = capsys.readouterr()
    assert "[assistant][notification_replace]" in captured.out
    assert 'status="process"' in captured.out
    assert "dedupe_key: progress:task_123" in captured.out


def test_agent_inbox_accepts_control_messages():
    inbox = _memory_inbox()

    inbox.publish(AgentControlMessage(command="pause", reason="operator request"))

    notification = inbox.drain()[0]
    assert notification.source == "control"
    assert notification.status == "pause"
    assert notification.metadata["reason"] == "operator request"


def test_agent_inbox_requires_store():
    with pytest.raises(ValueError, match="`store` is required"):
        AgentInbox()


def test_agent_creates_default_memory_store_for_inbox():
    model = Mock()
    model.model_type = "chat_completion"

    agent = Agent(name="assistant", model=model)

    assert isinstance(agent.agent_inbox.store, InMemoryAgentInboxStore)


def test_agent_scopes_inbox_without_mutating_an_inherited_parent_view():
    model = Mock()
    model.model_type = "chat_completion"
    parent = Agent(name="parent", model=model)
    child = Agent(name="child", model=model)
    parent_scope = ExecutionScope(
        namespace="parent",
        thread_id="shared_thread",
        run_id="parent_run",
    )
    child_scope = ExecutionScope(
        namespace="child",
        thread_id="shared_thread",
        run_id="child_run",
        parent_run_id="parent_run",
        root_run_id="parent_run",
    )
    parent_inbox = parent._get_scoped_agent_inbox(parent_scope)

    with execution_context(scope=parent_scope, agent_inbox=parent_inbox):
        child_inbox = child._get_scoped_agent_inbox(child_scope)

    assert child_inbox is not parent_inbox
    assert (parent_inbox.namespace, parent_inbox.thread_id, parent_inbox.run_id) == (
        "parent",
        "shared_thread",
        "parent_run",
    )
    assert (child_inbox.namespace, child_inbox.thread_id, child_inbox.run_id) == (
        "child",
        "shared_thread",
        "child_run",
    )


def test_concurrent_agent_inbox_views_keep_their_run_identity():
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="assistant", model=model)
    first = agent._get_scoped_agent_inbox(
        ExecutionScope(thread_id="thread_a", run_id="run_a")
    )
    second = agent._get_scoped_agent_inbox(
        ExecutionScope(thread_id="thread_b", run_id="run_b")
    )

    first.publish({"source": "test", "status": "first"})
    second.publish({"source": "test", "status": "second"})

    assert [item.status for item in first.drain()] == ["first"]
    assert [item.status for item in second.drain()] == ["second"]


def test_agent_inbox_renders_incoming_user_message():
    inbox = _memory_inbox()

    inbox.user_message("Please adjust the answer.", metadata={"user_id": "u1"})
    rendered = inbox.render(inbox.drain())

    assert rendered == {
        "role": "user",
        "content": (
            "<incoming_user_message>\n"
            "Please adjust the answer.\n"
            "</incoming_user_message>"
        ),
    }


def test_agent_inbox_renders_runtime_notifications_as_system_messages():
    inbox = _memory_inbox()

    inbox.publish(
        {
            "source": "task",
            "ref": "task_123",
            "status": "completed",
            "metadata": {"tool": "worker"},
        }
    )
    rendered = inbox.render(inbox.drain())

    assert isinstance(rendered, dict)
    assert rendered == {
        "role": "system",
        "content": (
            '<notification source="task" ref="task_123" status="completed" '
            'tool="worker"/>'
        ),
    }


def test_agent_inbox_renders_multiple_notifications_without_wrapper():
    inbox = _memory_inbox()

    inbox.publish({"source": "task", "status": "started"})
    inbox.publish({"source": "task", "status": "completed"})
    rendered = inbox.render(inbox.drain())

    assert rendered == {
        "role": "system",
        "content": (
            '<notification source="task" status="started"/>\n'
            '<notification source="task" status="completed"/>'
        ),
    }


def test_agent_inbox_nests_reserved_or_invalid_metadata_attributes():
    inbox = _memory_inbox()

    inbox.publish(
        {
            "source": "task",
            "metadata": {
                "tool": "worker & reviewer",
                "source": "shadow",
                "bad key": {"step": 1},
            },
        }
    )
    rendered = inbox.render(inbox.drain())

    assert rendered["content"] == (
        '<notification source="task" tool="worker &amp; reviewer" '
        'metadata=\'{"bad key":{"step":1},"source":"shadow"}\'/>'
    )


def test_agent_inbox_separates_incoming_user_message_from_system_notifications():
    inbox = _memory_inbox()

    inbox.user_message("Please adjust the answer.")
    inbox.publish({"source": "task", "status": "completed"})
    rendered = inbox.render_messages(inbox.drain())

    assert [message["role"] for message in rendered] == ["system", "user"]
    assert rendered[0]["content"] == (
        '<notification source="task" status="completed"/>'
    )
    assert "<incoming_user_message>" in rendered[1]["content"]


def test_agent_inbox_clear_user_messages_preserves_system_notifications():
    inbox = _memory_inbox()

    inbox.user_message("Please adjust the answer.")
    inbox.publish({"source": "task", "status": "completed"})

    assert inbox.clear_user_messages() == 1
    notifications = inbox.peek()

    assert len(notifications) == 1
    assert notifications[0].source == "task"
    assert inbox.clear_user_messages() == 0


def test_agent_inbox_persists_notifications_with_memory_store():
    store = InMemoryAgentInboxStore()
    writer = AgentInbox(
        store=store,
        namespace="assistant",
        thread_id="user_1",
        run_id="run_1",
    )
    reader = AgentInbox(
        store=store,
        namespace="assistant",
        thread_id="user_1",
        run_id="run_1",
    )

    writer.user_message("Continue with the new constraint.")

    notifications = reader.peek()
    assert len(notifications) == 1
    assert notifications[0].source == "incoming_user_message"
    assert notifications[0].metadata["content"] == "Continue with the new constraint."

    drained = reader.drain()
    assert len(drained) == 1
    assert writer.peek() == []


def test_agent_inbox_clear_user_messages_updates_store():
    store = InMemoryAgentInboxStore()
    writer = AgentInbox(
        store=store,
        namespace="assistant",
        thread_id="user_1",
        run_id="run_1",
    )
    reader = AgentInbox(
        store=store,
        namespace="assistant",
        thread_id="user_1",
        run_id="run_1",
    )

    writer.user_message("Continue with the new constraint.")
    writer.publish({"source": "task", "status": "completed"})

    assert reader.clear_user_messages() == 1
    notifications = writer.peek()

    assert len(notifications) == 1
    assert notifications[0].source == "task"


def test_agent_inbox_local_drain_is_scoped_by_thread_id():
    inbox = _memory_inbox(namespace="assistant", thread_id="user_1", run_id="run_1")

    inbox.user_message("Only user 1 should see this.")
    inbox.bind(thread_id="user_2", run_id="run_2")

    assert inbox.drain() == []

    inbox.bind(thread_id="user_1", run_id="run_1")
    drained = inbox.drain()

    assert len(drained) == 1
    assert drained[0].metadata["content"] == "Only user 1 should see this."


def test_agent_inbox_memory_store_keeps_multiple_thread_queues():
    inbox = _memory_inbox(namespace="assistant", thread_id="user_1", run_id="run_1")

    inbox.user_message("User 1 notification.")
    inbox.bind(thread_id="user_2", run_id="run_2")
    inbox.user_message("User 2 notification.")
    inbox.bind(thread_id="user_1", run_id="run_3")
    user_1_notifications = inbox.drain()
    inbox.bind(thread_id="user_2", run_id="run_4")
    user_2_notifications = inbox.drain()

    assert [
        notification.metadata["content"] for notification in user_1_notifications
    ] == ["User 1 notification."]
    assert [
        notification.metadata["content"] for notification in user_2_notifications
    ] == ["User 2 notification."]


def test_agent_inbox_local_first_bind_keeps_prebound_notifications():
    inbox = _memory_inbox()

    inbox.user_message("Deliver on first scope bind.")
    inbox.bind(namespace="assistant", thread_id="user_1", run_id="run_1")

    drained = inbox.drain()

    assert len(drained) == 1
    assert drained[0].metadata["content"] == "Deliver on first scope bind."


def test_agent_inbox_local_moves_pending_notifications_between_runs_same_thread():
    inbox = _memory_inbox(namespace="assistant", thread_id="user_1", run_id="run_1")

    inbox.user_message("Deliver on next turn.")
    inbox.bind(run_id="run_2")

    drained = inbox.drain()

    assert len(drained) == 1
    assert drained[0].metadata["content"] == "Deliver on next turn."


def test_agent_inbox_memory_store_drain_is_scoped_by_thread_id():
    store = InMemoryAgentInboxStore()
    writer = AgentInbox(
        store=store,
        namespace="assistant",
        thread_id="user_1",
        run_id="run_1",
    )
    other_thread = AgentInbox(
        store=store,
        namespace="assistant",
        thread_id="user_2",
        run_id="run_2",
    )
    original_thread = AgentInbox(
        store=store,
        namespace="assistant",
        thread_id="user_1",
        run_id="run_1",
    )

    writer.user_message("Only user 1 should see this.")

    assert other_thread.drain() == []

    drained = original_thread.drain()
    assert len(drained) == 1
    assert drained[0].metadata["content"] == "Only user 1 should see this."


def test_agent_inbox_memory_store_moves_pending_notifications_between_runs_same_thread():
    store = InMemoryAgentInboxStore()
    inbox = AgentInbox(
        store=store,
        namespace="assistant",
        thread_id="user_1",
        run_id="run_1",
    )

    inbox.user_message("Deliver on next turn.")
    inbox.bind(run_id="run_2")

    drained = inbox.drain()

    assert len(drained) == 1
    assert drained[0].metadata["content"] == "Deliver on next turn."


def test_agent_inbox_sqlite_store_moves_pending_notifications_between_runs_same_thread(
    tmp_path,
):
    store = SQLiteAgentInboxStore(path=str(tmp_path / "agent-inboxes.sqlite3"))
    inbox = AgentInbox(
        store=store,
        namespace="assistant",
        thread_id="user_1",
        run_id="run_1",
    )

    inbox.user_message("Deliver on next turn.")
    inbox.bind(run_id="run_2")

    drained = inbox.drain()

    assert len(drained) == 1
    assert drained[0].metadata["content"] == "Deliver on next turn."

    store.close()


def test_store_factory_creates_agent_inbox_stores(tmp_path):
    memory_store = Store.agent_inbox("in_memory")
    sqlite_store = Store.agent_inbox(
        "sqlite",
        path=str(tmp_path / "agent-inboxes.sqlite3"),
    )

    assert isinstance(memory_store, InMemoryAgentInboxStore)
    assert isinstance(sqlite_store, SQLiteAgentInboxStore)

    sqlite_store.close()


def test_inbox_store_routing_ids_distinguish_memory_instances():
    first = InMemoryAgentInboxStore()
    second = InMemoryAgentInboxStore()

    assert first.routing_id == first.routing_id
    assert first.routing_id != second.routing_id


def test_sqlite_inbox_routing_id_survives_reopen_and_move(tmp_path):
    source = tmp_path / "source.sqlite3"
    destination = tmp_path / "destination.sqlite3"
    store = SQLiteAgentInboxStore(path=str(source))
    routing_id = store.routing_id
    store.close()

    copyfile(source, destination)
    reopened = SQLiteAgentInboxStore(path=str(destination))
    try:
        assert reopened.routing_id == routing_id
    finally:
        reopened.close()


def test_agent_inbox_persists_notifications_with_sqlite_store(tmp_path):
    path = tmp_path / "agent-inboxes.sqlite3"
    store = SQLiteAgentInboxStore(path=str(path))
    writer = AgentInbox(
        store=store,
        namespace="assistant",
        thread_id="user_1",
        run_id="run_1",
    )

    writer.pause(reason="operator needs review")
    store.close()

    reopened = SQLiteAgentInboxStore(path=str(path))
    reader = AgentInbox(
        store=reopened,
        namespace="assistant",
        thread_id="user_1",
        run_id="run_1",
    )
    notification = reader.drain()[0]

    assert notification.source == "control"
    assert notification.status == "pause"
    assert notification.metadata["reason"] == "operator needs review"
    assert reader.peek() == []

    reopened.close()


def test_agent_inbox_claim_is_recoverable_after_release():
    store = InMemoryAgentInboxStore()
    writer = AgentInbox(
        store=store, namespace="assistant", thread_id="user_1", run_id="run_1"
    )
    reader = AgentInbox(
        store=store, namespace="assistant", thread_id="user_1", run_id="run_1"
    )
    notification = writer.user_message("retry me")

    assert [item.notification_id for item in reader.claim()] == [
        notification.notification_id
    ]
    assert reader.claim() == []
    reader.release()
    retry = reader.claim()
    assert [item.notification_id for item in retry] == [notification.notification_id]
    reader.ack([notification.notification_id])
    assert reader.peek() == []


def test_agent_inbox_sqlite_competing_connections_claim_once(tmp_path):
    path = str(tmp_path / "agent-inboxes.sqlite3")
    first_store = SQLiteAgentInboxStore(path=path)
    second_store = SQLiteAgentInboxStore(path=path)
    first = AgentInbox(
        store=first_store, namespace="assistant", thread_id="user_1", run_id="run_1"
    )
    second = AgentInbox(
        store=second_store, namespace="assistant", thread_id="user_1", run_id="run_1"
    )
    notification = first.user_message("claim once")

    first_claim = first.claim()
    second_claim = second.claim()
    assert [item.notification_id for item in first_claim] == [
        notification.notification_id
    ]
    assert second_claim == []
    first.ack([notification.notification_id])
    assert second.peek() == []
    first_store.close()
    second_store.close()


def test_agent_inbox_move_is_atomic_for_store_views(tmp_path):
    store = SQLiteAgentInboxStore(path=str(tmp_path / "agent-inboxes.sqlite3"))
    inbox = AgentInbox(
        store=store, namespace="assistant", thread_id="user_1", run_id="run_1"
    )
    inbox.user_message("move once")
    inbox.bind(run_id="run_2")
    assert [item.metadata["content"] for item in inbox.claim()] == ["move once"]
    store.close()


@pytest.mark.parametrize("provider", ["memory", "sqlite"])
def test_agent_inbox_expired_lease_cannot_ack_new_owner(provider, tmp_path):
    store = (
        InMemoryAgentInboxStore()
        if provider == "memory"
        else SQLiteAgentInboxStore(path=str(tmp_path / "lease.sqlite3"))
    )
    first = AgentInbox(
        store=store, namespace="assistant", thread_id="user_1", run_id="run_1"
    )
    second = AgentInbox(
        store=store, namespace="assistant", thread_id="user_1", run_id="run_1"
    )
    notification = first.user_message("lease ownership")
    old_claim = first.claim(lease_seconds=0.01)
    import time

    time.sleep(0.03)
    new_claim = second.claim(lease_seconds=30)

    first.ack([notification.notification_id])
    assert [item.notification_id for item in second.peek()] == [
        notification.notification_id
    ]
    second.ack([item.notification_id for item in new_claim])
    assert second.peek() == []
    assert old_claim and new_claim
    if hasattr(store, "close"):
        store.close()


def test_agent_inbox_ack_only_removes_requested_ids():
    inbox = _memory_inbox(namespace="assistant", thread_id="user_1", run_id="run_1")
    first = inbox.user_message("first")
    second = inbox.user_message("second")
    claimed = inbox.claim()

    inbox.ack([first.notification_id])
    assert [item.notification_id for item in inbox.peek()] == [second.notification_id]
    inbox.release()
    assert [item.notification_id for item in inbox.claim()] == [second.notification_id]
    assert claimed


@pytest.mark.parametrize("sqlite", [False, True])
def test_partial_lease_release_keeps_delivered_notification_claimed(sqlite, tmp_path):
    store = (
        SQLiteAgentInboxStore(path=str(tmp_path / "inbox.sqlite"))
        if sqlite
        else InMemoryAgentInboxStore()
    )
    first = AgentInbox(store=store, namespace="agent", thread_id="thread", run_id="run")
    second = first.fork()
    delivered = first.user_message("delivered")
    filtered = first.user_message("filtered")
    first.claim()
    first.mark_delivered([delivered.notification_id])
    first.release(except_ids=[delivered.notification_id])
    assert [item.notification_id for item in second.claim()] == [
        filtered.notification_id
    ]
    first.ack([delivered.notification_id])
    assert [item.notification_id for item in second.peek()] == [
        filtered.notification_id
    ]
    if sqlite:
        store.close()


def test_agent_inbox_dedupe_replay_replaces_claimed_notification():
    inbox = _memory_inbox(namespace="assistant", thread_id="user_1", run_id="run_1")
    first = inbox.publish(
        {"source": "task", "status": "started", "dedupe_key": "task:1"}
    )
    inbox.claim()
    second = inbox.publish(
        {"source": "task", "status": "completed", "dedupe_key": "task:1"}
    )

    pending = inbox.peek()
    assert len(pending) == 1
    assert pending[0].notification_id == second.notification_id
    assert pending[0].status == "completed"
    assert first.notification_id != second.notification_id
