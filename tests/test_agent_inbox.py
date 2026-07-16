"""Tests for msgflux.runtime.agent_inbox."""

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
    assert "task_id=task_123" in captured.out
    assert "status=completed" in captured.out
    assert "tool=worker" in captured.out
    assert "[assistant][notification_drain]" in captured.out
    assert "1 notification(s)" in captured.out
    assert "<notifications>" in captured.out
    assert "</notifications>" in captured.out


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
    assert "status=process" in captured.out
    assert "dedupe_key: progress:task_123" in captured.out


def test_agent_inbox_accepts_control_messages():
    inbox = _memory_inbox()

    inbox.publish(AgentControlMessage(command="pause", reason="operator request"))

    notification = inbox.drain()[0]
    assert notification.source == "control"
    assert notification.role is None
    assert notification.status == "pause"
    assert notification.hint == "operator request"
    assert "reason" not in notification.metadata


def test_agent_inbox_rejects_unknown_control_commands():
    inbox = _memory_inbox()

    with pytest.raises(ValueError, match="Unsupported control command"):
        inbox.control("restart")


def test_agent_inbox_does_not_render_internal_notifications():
    inbox = _memory_inbox()

    rendered = inbox.render([AgentControlMessage(command="pause").to_notification()])

    assert rendered is None


def test_agent_inbox_supports_explicit_user_role():
    inbox = _memory_inbox()

    rendered = inbox.render(
        [{"source": "operator_message", "role": "user", "hint": "Continue."}]
    )

    assert rendered == {
        "role": "user",
        "content": "<incoming_user_message>\nContinue.\n</incoming_user_message>",
    }


def test_agent_inbox_rejects_unsupported_roles():
    inbox = _memory_inbox()

    with pytest.raises(ValueError, match=r"AgentNotification\.role"):
        inbox.publish({"source": "tool", "role": "assistant"})

    with pytest.raises(ValueError, match="Only control"):
        inbox.publish({"source": "tool", "role": None})


def test_agent_inbox_infers_role_for_legacy_payloads():
    inbox = _memory_inbox()

    rendered = inbox.render(
        [{"source": "incoming_user_message", "hint": "Legacy message."}]
    )

    assert rendered["role"] == "user"


def test_agent_inbox_requires_store():
    with pytest.raises(ValueError, match="`store` is required"):
        AgentInbox()


def test_agent_creates_default_memory_store_for_inbox():
    model = Mock()
    model.model_type = "chat_completion"

    agent = Agent(name="assistant", model=model)

    assert isinstance(agent.agent_inbox.store, InMemoryAgentInboxStore)


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
    assert rendered["role"] == "system"
    assert rendered["content"] == (
        "<notifications>\n"
        "task_id=task_123 status=completed tool=worker\n"
        "</notifications>"
    )


def test_agent_inbox_separates_incoming_user_message_from_system_notifications():
    inbox = _memory_inbox()

    inbox.user_message("Please adjust the answer.")
    inbox.publish({"source": "task", "status": "completed"})
    rendered = inbox.render_messages(inbox.drain())

    assert [message["role"] for message in rendered] == ["system", "user"]
    assert "<notifications>" in rendered[0]["content"]
    assert "<incoming_user_message>" in rendered[1]["content"]
    assert "<notifications>" not in rendered[1]["content"]


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
    inbox = _memory_inbox(namespace="assistant", thread_id="user_1", run_id="run_1")

    inbox.user_message("Only user 1 should see this.")
    inbox.bind(thread_id="user_2", run_id="run_2")

    assert inbox.drain() == []

    inbox.bind(thread_id="user_1", run_id="run_1")
    drained = inbox.drain()

    assert len(drained) == 1
    assert drained[0].hint == "Only user 1 should see this."


def test_agent_inbox_memory_store_keeps_multiple_thread_queues():
    inbox = _memory_inbox(namespace="assistant", thread_id="user_1", run_id="run_1")

    inbox.user_message("User 1 notification.")
    inbox.bind(thread_id="user_2", run_id="run_2")
    inbox.user_message("User 2 notification.")
    inbox.bind(thread_id="user_1", run_id="run_3")
    user_1_notifications = inbox.drain()
    inbox.bind(thread_id="user_2", run_id="run_4")
    user_2_notifications = inbox.drain()

    assert [notification.hint for notification in user_1_notifications] == [
        "User 1 notification."
    ]
    assert [notification.hint for notification in user_2_notifications] == [
        "User 2 notification."
    ]


def test_agent_inbox_local_first_bind_keeps_prebound_notifications():
    inbox = _memory_inbox()

    inbox.user_message("Deliver on first scope bind.")
    inbox.bind(namespace="assistant", thread_id="user_1", run_id="run_1")

    drained = inbox.drain()

    assert len(drained) == 1
    assert drained[0].hint == "Deliver on first scope bind."


def test_agent_inbox_local_moves_pending_notifications_between_runs_same_thread():
    inbox = _memory_inbox(namespace="assistant", thread_id="user_1", run_id="run_1")

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
