"""SQLite inbox mutations remain atomic and reusable after storage failures."""

import sqlite3

import pytest

from msgflux.runtime.agent_inbox import SQLiteAgentInboxStore


def _store(tmp_path):
    return SQLiteAgentInboxStore(str(tmp_path / "agent-inbox.sqlite3"))


def _trigger(store, name, operation, table, condition=""):
    store._conn.execute(
        f"""CREATE TRIGGER {name} BEFORE {operation} ON {table} {condition}
        BEGIN SELECT RAISE(ABORT, 'injected {name} failure'); END"""
    )
    store._conn.commit()


def _drop_trigger(store, name):
    store._conn.execute(f"DROP TRIGGER {name}")
    store._conn.commit()


def _notification(notification_id, message):
    return {
        "notification_id": notification_id,
        "source": "incoming_user_message",
        "metadata": {"content": message},
    }


def test_publish_failure_rolls_back_and_store_can_be_reused(tmp_path):
    store = _store(tmp_path)
    try:
        _trigger(
            store,
            "reject_publish",
            "INSERT",
            "agent_inbox_notifications",
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected reject_publish"):
            store.publish_notification(
                "ns", "thread", "run", _notification("first", "first")
            )
        assert store._conn.in_transaction is False
        assert store.load_notifications("ns", "thread", "run") == []

        _drop_trigger(store, "reject_publish")
        store.publish_notification(
            "ns", "thread", "run", _notification("second", "second")
        )
        assert [
            item["notification_id"]
            for item in store.load_notifications("ns", "thread", "run")
        ] == ["second"]
    finally:
        store.close()


def test_claim_failure_rolls_back_lease_and_store_can_be_reused(tmp_path):
    store = _store(tmp_path)
    try:
        store.publish_notification(
            "ns", "thread", "run", _notification("message", "retry")
        )
        _trigger(
            store,
            "reject_claim",
            "UPDATE",
            "agent_inbox_notifications",
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected reject_claim"):
            store.claim_notifications(
                "ns", "thread", "run", lease_id="lease-1", lease_seconds=30
            )
        row = store._conn.execute(
            "SELECT status, lease_id FROM agent_inbox_notifications WHERE notification_id='message'"
        ).fetchone()
        assert tuple(row) == ("pending", None)
        assert store._conn.in_transaction is False

        _drop_trigger(store, "reject_claim")
        claimed = store.claim_notifications(
            "ns", "thread", "run", lease_id="lease-2", lease_seconds=30
        )
        assert [item["notification_id"] for item in claimed] == ["message"]
    finally:
        store.close()


def test_move_failure_restores_source_and_does_not_partially_fill_target(tmp_path):
    store = _store(tmp_path)
    try:
        store.publish_notification(
            "ns", "thread", "old", _notification("message", "move me")
        )
        _trigger(
            store,
            "reject_move_delete",
            "DELETE",
            "agent_inbox_notifications",
            "WHEN OLD.run_id='old'",
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected reject_move_delete"):
            store.move_notifications(("ns", "thread", "old"), ("ns", "thread", "new"))
        assert [
            item["notification_id"]
            for item in store.load_notifications("ns", "thread", "old")
        ] == ["message"]
        assert store.load_notifications("ns", "thread", "new") == []
        assert store._conn.in_transaction is False

        _drop_trigger(store, "reject_move_delete")
        store.move_notifications(("ns", "thread", "old"), ("ns", "thread", "new"))
        assert store.load_notifications("ns", "thread", "old") == []
        assert [
            item["notification_id"]
            for item in store.load_notifications("ns", "thread", "new")
        ] == ["message"]
    finally:
        store.close()


def test_ack_failure_restores_notification_and_can_retry(tmp_path):
    store = _store(tmp_path)
    try:
        store.publish_notification(
            "ns", "thread", "run", _notification("message", "ack me")
        )
        _trigger(store, "reject_ack_sync", "UPDATE", "agent_inboxes")
        with pytest.raises(sqlite3.IntegrityError, match="injected reject_ack_sync"):
            store.ack_notifications("ns", "thread", "run", ["message"])
        assert [
            item["notification_id"]
            for item in store.load_notifications("ns", "thread", "run")
        ] == ["message"]
        assert store._conn.in_transaction is False

        _drop_trigger(store, "reject_ack_sync")
        store.ack_notifications("ns", "thread", "run", ["message"])
        assert store.load_notifications("ns", "thread", "run") == []
    finally:
        store.close()


def test_release_failure_keeps_lease_and_can_retry(tmp_path):
    store = _store(tmp_path)
    try:
        store.publish_notification(
            "ns", "thread", "run", _notification("message", "release me")
        )
        store.claim_notifications(
            "ns", "thread", "run", lease_id="lease", lease_seconds=30
        )
        _trigger(
            store,
            "reject_release",
            "UPDATE",
            "agent_inbox_notifications",
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected reject_release"):
            store.release_notifications(
                "ns", "thread", "run", lease_id="lease", notification_ids=["message"]
            )
        row = store._conn.execute(
            "SELECT status, lease_id FROM agent_inbox_notifications WHERE notification_id='message'"
        ).fetchone()
        assert tuple(row) == ("leased", "lease")
        assert store._conn.in_transaction is False

        _drop_trigger(store, "reject_release")
        store.release_notifications(
            "ns", "thread", "run", lease_id="lease", notification_ids=["message"]
        )
        row = store._conn.execute(
            "SELECT status, lease_id FROM agent_inbox_notifications WHERE notification_id='message'"
        ).fetchone()
        assert tuple(row) == ("pending", None)
    finally:
        store.close()


def test_save_failure_restores_old_payload_and_can_retry(tmp_path):
    store = _store(tmp_path)
    try:
        store.publish_notification(
            "ns", "thread", "run", _notification("old", "old payload")
        )
        _trigger(
            store,
            "reject_save_insert",
            "INSERT",
            "agent_inbox_notifications",
        )
        with pytest.raises(sqlite3.IntegrityError, match="injected reject_save_insert"):
            store.save_notifications(
                "ns", "thread", "run", [_notification("new", "new payload")]
            )
        assert [
            item["notification_id"]
            for item in store.load_notifications("ns", "thread", "run")
        ] == ["old"]
        assert store._conn.in_transaction is False

        _drop_trigger(store, "reject_save_insert")
        store.save_notifications(
            "ns", "thread", "run", [_notification("new", "new payload")]
        )
        assert [
            item["notification_id"]
            for item in store.load_notifications("ns", "thread", "run")
        ] == ["new"]
    finally:
        store.close()


def test_clear_failure_restores_rows_and_can_retry(tmp_path):
    store = _store(tmp_path)
    try:
        store.publish_notification(
            "ns", "thread", "run", _notification("message", "keep me")
        )
        _trigger(store, "reject_clear_parent", "DELETE", "agent_inboxes")
        with pytest.raises(
            sqlite3.IntegrityError, match="injected reject_clear_parent"
        ):
            store.clear("ns", "thread", "run")
        assert [
            item["notification_id"]
            for item in store.load_notifications("ns", "thread", "run")
        ] == ["message"]
        assert store._conn.in_transaction is False

        _drop_trigger(store, "reject_clear_parent")
        assert store.clear("ns", "thread", "run") == 1
        assert store.load_notifications("ns", "thread", "run") == []
    finally:
        store.close()


def test_rollback_failure_is_not_allowed_to_replace_original_failure(tmp_path):
    store = _store(tmp_path)
    original_connection = store._conn

    class RollbackFailsAfterCleanup:
        def __getattr__(self, name):
            return getattr(original_connection, name)

        def rollback(self):
            original_connection.rollback()
            raise OSError("injected rollback failure")

    try:
        _trigger(
            store,
            "reject_publish",
            "INSERT",
            "agent_inbox_notifications",
        )
        store._conn = RollbackFailsAfterCleanup()
        with pytest.raises(
            sqlite3.IntegrityError, match="injected reject_publish"
        ) as raised:
            store.publish_notification(
                "ns", "thread", "run", _notification("first", "first")
            )
        assert any(
            "SQLite inbox rollback failed" in note
            for note in getattr(raised.value, "__notes__", ())
        )
        assert original_connection.in_transaction is False

        _drop_trigger(store, "reject_publish")
        store.publish_notification(
            "ns", "thread", "run", _notification("second", "second")
        )
    finally:
        store._conn = original_connection
        store.close()


def test_commit_failure_rolls_back_and_store_can_be_reused(tmp_path):
    store = _store(tmp_path)
    original_connection = store._conn

    class CommitFails:
        def __getattr__(self, name):
            return getattr(original_connection, name)

        def commit(self):
            raise OSError("injected commit failure")

    try:
        store._conn = CommitFails()
        with pytest.raises(OSError, match="injected commit failure"):
            store.publish_notification(
                "ns", "thread", "run", _notification("first", "first")
            )
        assert original_connection.in_transaction is False
        assert store.load_notifications("ns", "thread", "run") == []

        store._conn = original_connection
        store.publish_notification(
            "ns", "thread", "run", _notification("second", "second")
        )
        assert [
            item["notification_id"]
            for item in store.load_notifications("ns", "thread", "run")
        ] == ["second"]
    finally:
        store._conn = original_connection
        store.close()
