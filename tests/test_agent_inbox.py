"""Tests for msgflux.runtime.agent_inbox."""

from msgflux.runtime.agent_inbox import (
    AgentControlMessage,
    AgentInbox,
    InMemoryAgentInboxStore,
    SQLiteAgentInboxStore,
)
from msgflux.data.stores import Store


def test_agent_inbox_verbose_publish_and_drain_are_printed(capsys):
    inbox = AgentInbox(verbose=True, owner="assistant")

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
    assert "<notification>" in captured.out
    assert "source: task" in captured.out
    assert "ref: task_123" in captured.out
    assert "status: completed" in captured.out
    assert "tool: worker" in captured.out
    assert "[assistant][notification_drain]" in captured.out
    assert "1 notification(s)" in captured.out
    assert "<system_note>" in captured.out
    assert "</system_note>" in captured.out


def test_agent_inbox_verbose_replace_is_printed(capsys):
    inbox = AgentInbox(verbose=True, owner="assistant")

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
    assert "status: process" in captured.out
    assert "dedupe_key: progress:task_123" in captured.out


def test_agent_inbox_accepts_control_messages():
    inbox = AgentInbox()

    inbox.publish(AgentControlMessage(command="pause", reason="operator request"))

    notification = inbox.drain()[0]
    assert notification.source == "control"
    assert notification.status == "pause"
    assert notification.hint == "operator request"


def test_agent_inbox_renders_incoming_user_message():
    inbox = AgentInbox()

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
    inbox = AgentInbox()

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
    assert rendered["role"] == "system"
    assert "<system_note>" in rendered["content"]
    assert "<notification>" in rendered["content"]
    assert "source: task" in rendered["content"]
    assert "ref: task_123" in rendered["content"]
    assert "status: completed" in rendered["content"]


def test_agent_inbox_separates_incoming_user_message_from_system_notifications():
    inbox = AgentInbox()

    inbox.user_message("Please adjust the answer.")
    inbox.publish({"source": "task", "status": "completed"})
    rendered = inbox.render_messages(inbox.drain())

    assert [message["role"] for message in rendered] == ["system", "user"]
    assert "<system_note>" in rendered[0]["content"]
    assert "<incoming_user_message>" in rendered[1]["content"]
    assert "<system_note>" not in rendered[1]["content"]


def test_agent_inbox_clear_user_messages_preserves_system_notifications():
    inbox = AgentInbox()

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
    assert notifications[0].hint == "Continue with the new constraint."

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
    inbox = AgentInbox(namespace="assistant", thread_id="user_1", run_id="run_1")

    inbox.user_message("Only user 1 should see this.")
    inbox.bind(thread_id="user_2", run_id="run_2")

    assert inbox.drain() == []

    inbox.bind(thread_id="user_1", run_id="run_1")
    drained = inbox.drain()

    assert len(drained) == 1
    assert drained[0].hint == "Only user 1 should see this."


def test_agent_inbox_local_first_bind_keeps_prebound_notifications():
    inbox = AgentInbox()

    inbox.user_message("Deliver on first scope bind.")
    inbox.bind(namespace="assistant", thread_id="user_1", run_id="run_1")

    drained = inbox.drain()

    assert len(drained) == 1
    assert drained[0].hint == "Deliver on first scope bind."


def test_agent_inbox_local_moves_pending_notifications_between_runs_same_thread():
    inbox = AgentInbox(namespace="assistant", thread_id="user_1", run_id="run_1")

    inbox.user_message("Deliver on next turn.")
    inbox.bind(run_id="run_2")

    drained = inbox.drain()

    assert len(drained) == 1
    assert drained[0].hint == "Deliver on next turn."


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
    assert drained[0].hint == "Only user 1 should see this."


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
    assert drained[0].hint == "Deliver on next turn."


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
    assert drained[0].hint == "Deliver on next turn."

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
    assert notification.hint == "operator needs review"
    assert reader.peek() == []

    reopened.close()
