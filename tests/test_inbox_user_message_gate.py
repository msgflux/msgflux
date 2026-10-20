"""Conformance tests for run-scoped incoming-message acceptance."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import time

import pytest

from msgflux.runtime.agent_inbox import (
    AgentInbox,
    AgentInboxClosedError,
    InMemoryAgentInboxStore,
    SQLiteAgentInboxStore,
)


@pytest.fixture(params=["memory", "sqlite"])
def inbox_store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryAgentInboxStore()
        return
    store = SQLiteAgentInboxStore(str(tmp_path / "inbox.sqlite3"))
    try:
        yield store
    finally:
        store.close()


def _inbox(store, run_id="run"):
    return AgentInbox(
        store=store,
        namespace="agent",
        thread_id="thread",
        run_id=run_id,
    )


def _race_publish_and_close(closing, publishing):
    barrier = Barrier(3)

    def publish():
        barrier.wait(timeout=3)
        try:
            return publishing.user_message("race").notification_id
        except AgentInboxClosedError:
            return None

    def close():
        barrier.wait(timeout=3)
        return closing.close_user_messages()

    with ThreadPoolExecutor(max_workers=2) as pool:
        publish_future = pool.submit(publish)
        close_future = pool.submit(close)
        barrier.wait(timeout=3)
        notification_id = publish_future.result(timeout=5)
        sealed = close_future.result(timeout=5)
    return notification_id, sealed


@pytest.mark.parametrize("source", ["user_message", "described_message"])
def test_incoming_user_content_requires_an_open_run_gate(inbox_store, source):
    inbox = _inbox(inbox_store)

    def publish():
        if source == "user_message":
            return inbox.user_message("hello")
        return inbox.message("hello", description="A note", source="operator")

    accepted = publish()
    claimed = inbox.claim()
    assert [item.notification_id for item in claimed] == [accepted.notification_id]
    inbox.mark_delivered([accepted.notification_id])

    assert inbox.close_user_messages(consumed_ids=[accepted.notification_id]) is True
    assert [item.notification_id for item in inbox.peek()] == [accepted.notification_id]
    inbox.ack([accepted.notification_id])
    assert inbox.peek() == []
    with pytest.raises(AgentInboxClosedError):
        publish()


def test_pending_leased_user_content_prevents_conditional_close(inbox_store):
    inbox = _inbox(inbox_store)
    accepted = inbox.user_message("still needs delivery")
    claimed = inbox.claim()

    assert [item.notification_id for item in claimed] == [accepted.notification_id]
    assert inbox.close_user_messages() is False
    assert inbox.close_user_messages(consumed_ids=[accepted.notification_id]) is True


def test_reopen_and_non_user_notifications_are_not_gated(inbox_store):
    inbox = _inbox(inbox_store)
    assert inbox.close_user_messages() is True

    task_notification = inbox.publish(
        {"source": "task", "status": "completed", "ref": "task-1"}
    )
    control_notification = inbox.pause(reason="review")
    effort_notification = inbox.set_reasoning_effort("high")
    assert [item.notification_id for item in inbox.peek()] == [
        task_notification.notification_id,
        control_notification.notification_id,
        effort_notification.notification_id,
    ]

    inbox.open_user_messages()
    assert inbox.user_message("new turn")


def test_gate_lifecycle_survives_save_move_and_clear(inbox_store):
    source = _inbox(inbox_store, run_id="source")
    target = _inbox(inbox_store, run_id="target")
    assert target.close_user_messages(only_if_empty=False) is True

    inbox_store.save_notifications(
        source.namespace,
        source.thread_id,
        source.run_id,
        [{"notification_id": "task-1", "source": "task", "status": "started"}],
    )
    assert source.close_user_messages() is True
    inbox_store.move_notifications(
        (source.namespace, source.thread_id, source.run_id),
        (target.namespace, target.thread_id, target.run_id),
    )

    with pytest.raises(AgentInboxClosedError):
        target.user_message("target is still sealed")
    with pytest.raises(AgentInboxClosedError):
        source.user_message("source gate survived move")

    inbox_store.clear(source.namespace, source.thread_id, source.run_id)
    source.user_message("clear resets the removed scope")


def test_sqlite_gate_is_durable_and_shared_across_connections(tmp_path):
    path = str(tmp_path / "shared-inbox.sqlite3")
    first_store = SQLiteAgentInboxStore(path)
    second_store = SQLiteAgentInboxStore(path)
    first = _inbox(first_store)
    second = _inbox(second_store)
    try:
        assert first.close_user_messages() is True
        with pytest.raises(AgentInboxClosedError):
            second.user_message("closed from another connection")

        first_store.close()
        first_store = SQLiteAgentInboxStore(path)
        first = _inbox(first_store)
        first.open_user_messages()
        assert second.user_message("reopened durably")
    finally:
        first_store.close()
        second_store.close()


def test_sqlite_age_clear_keeps_gate_for_recent_inbox_activity(tmp_path):
    store = SQLiteAgentInboxStore(str(tmp_path / "age-clear.sqlite3"))
    inbox = _inbox(store)
    try:
        assert inbox.close_user_messages() is True
        inbox.publish({"source": "task", "status": "started"})
        store._conn.execute(
            "UPDATE agent_inbox_run_gates SET updated_at=? "
            "WHERE namespace=? AND thread_id=? AND run_id=?",
            (time.time() - 120, inbox.namespace, inbox.thread_id, inbox.run_id),
        )
        store._conn.commit()

        store.clear(inbox.namespace, inbox.thread_id, inbox.run_id, older_than=60)

        assert inbox.peek()[0].source == "task"
        with pytest.raises(AgentInboxClosedError):
            inbox.user_message("recent inbox activity keeps the gate")
    finally:
        store.close()


def test_sqlite_publish_racing_close_has_one_linearized_winner(tmp_path):
    path = str(tmp_path / "race.sqlite3")
    closing_store = SQLiteAgentInboxStore(path)
    publishing_store = SQLiteAgentInboxStore(path)
    try:
        for index in range(12):
            run_id = f"run-{index}"
            closing = _inbox(closing_store, run_id)
            publishing = _inbox(publishing_store, run_id)
            notification_id, sealed = _race_publish_and_close(closing, publishing)

            assert (notification_id is not None) != sealed
            if notification_id is not None:
                assert [item.notification_id for item in closing.peek()] == [
                    notification_id
                ]
                assert (
                    closing.close_user_messages(consumed_ids=[notification_id]) is True
                )
            else:
                assert closing.peek() == []
    finally:
        closing_store.close()
        publishing_store.close()
