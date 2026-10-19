"""SQLite checkpoint and approval writes roll back completely on failure."""

import sqlite3

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.runtime.approvals import ApprovalBinding, SQLiteApprovalStore


def _message_state():
    return ChatMessages(
        [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ],
        thread_id="thread",
        namespace="ns",
    )._to_state()


class _CursorProxy:
    def __init__(self, cursor, owner):
        self._cursor = cursor
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def execute(self, *args, **kwargs):
        try:
            return self._cursor.execute(*args, **kwargs)
        except sqlite3.IntegrityError as error:
            self._owner.primary_error = error
            raise


class _RollbackFailsAfterCleanup:
    def __init__(self, connection):
        self._connection = connection
        self.primary_error = None

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def execute(self, *args, **kwargs):
        try:
            return self._connection.execute(*args, **kwargs)
        except sqlite3.IntegrityError as error:
            self.primary_error = error
            raise

    def cursor(self, *args, **kwargs):
        return _CursorProxy(self._connection.cursor(*args, **kwargs), self)

    def rollback(self):
        self._connection.rollback()
        raise OSError("injected secondary rollback failure")


def test_checkpoint_save_state_rolls_back_partial_message_normalization(tmp_path):
    store = SQLiteCheckpointStore(str(tmp_path / "checkpoints.sqlite3"))
    store._conn.execute(
        "CREATE TRIGGER reject_second_item BEFORE INSERT "
        "ON checkpoint_message_items WHEN (SELECT count(*) "
        "FROM checkpoint_message_items WHERE namespace='ns' "
        "AND thread_id='thread') >= 1 "
        "BEGIN SELECT RAISE(ABORT, 'second item failed'); END"
    )

    with pytest.raises(sqlite3.IntegrityError, match="second item failed"):
        store.save_state(
            "ns",
            "thread",
            "failed",
            {"status": "running", "messages": _message_state()},
        )

    assert not store._conn.in_transaction
    assert store.load_state("ns", "thread", "failed") is None
    assert (
        store._conn.execute("SELECT count(*) FROM checkpoint_message_items").fetchone()[
            0
        ]
        == 0
    )

    store._conn.execute("DROP TRIGGER reject_second_item")
    store._conn.commit()
    store.save_state("ns", "thread", "next", {"status": "running"})
    assert (
        store._conn.execute("SELECT count(*) FROM checkpoint_message_items").fetchone()[
            0
        ]
        == 0
    )
    store.close()


def test_checkpoint_commit_preserves_primary_error_when_rollback_fails(tmp_path):
    store = SQLiteCheckpointStore(str(tmp_path / "checkpoints.sqlite3"))
    store._conn.execute(
        "CREATE TRIGGER reject_commit BEFORE INSERT ON checkpoint_commits "
        "BEGIN SELECT RAISE(ABORT, 'checkpoint primary failure'); END"
    )
    original_connection = store._conn
    proxy = _RollbackFailsAfterCleanup(original_connection)
    store._conn = proxy
    try:
        with pytest.raises(
            sqlite3.IntegrityError, match="checkpoint primary failure"
        ) as raised:
            store.commit_state("ns", "thread", "run", {"status": "running"})

        assert raised.value is proxy.primary_error
        assert any(
            "SQLite checkpoint rollback also failed (OSError)" in note
            for note in getattr(raised.value, "__notes__", ())
        )
        assert not original_connection.in_transaction
        assert store.load_state("ns", "thread", "run") is None
        assert store.load_events("ns", "thread", "run") == []
        assert (
            original_connection.execute(
                "SELECT count(*) FROM checkpoint_commits"
            ).fetchone()[0]
            == 0
        )

        store._conn = original_connection
        original_connection.execute("DROP TRIGGER reject_commit")
        assert store.commit_state("ns", "thread", "run", {"status": "running"})
    finally:
        store._conn = original_connection
        store.close()


@pytest.mark.parametrize("operation", ["append_event", "delete_run", "clear"])
def test_checkpoint_legacy_mutations_roll_back_and_reuse_connection(
    tmp_path, operation
):
    store = SQLiteCheckpointStore(str(tmp_path / f"{operation}.sqlite3"))
    store.save_state(
        "ns", "thread", "run", {"status": "running", "messages": _message_state()}
    )
    store._conn.execute(
        "CREATE TRIGGER reject_checkpoint_mutation BEFORE DELETE "
        "ON checkpoint_message_items "
        "BEGIN SELECT RAISE(ABORT, 'checkpoint mutation failure'); END"
    )

    if operation == "append_event":
        store._conn.execute("DROP TRIGGER reject_checkpoint_mutation")
        store._conn.execute(
            "CREATE TRIGGER reject_checkpoint_mutation BEFORE INSERT "
            "ON checkpoint_events "
            "BEGIN SELECT RAISE(ABORT, 'checkpoint mutation failure'); END"
        )
        invoke = lambda: store.append_event(
            "ns", "thread", "run", {"event_type": "test"}
        )
    elif operation == "delete_run":
        invoke = lambda: store.delete_run("ns", "thread", "run")
    else:
        invoke = lambda: store.clear("ns", "thread")

    with pytest.raises(sqlite3.IntegrityError, match="checkpoint mutation failure"):
        invoke()
    assert not store._conn.in_transaction
    assert store.load_state("ns", "thread", "run") is not None
    assert (
        store._conn.execute("SELECT count(*) FROM checkpoint_message_items").fetchone()[
            0
        ]
        == 2
    )

    store._conn.execute("DROP TRIGGER reject_checkpoint_mutation")
    store._conn.commit()
    if operation == "append_event":
        store.append_event("ns", "thread", "run", {"event_type": "test"})
        assert store.load_events("ns", "thread", "run") == [{"event_type": "test"}]
    else:
        assert invoke()
        assert store.load_state("ns", "thread", "run") is None
    store.close()


def test_approval_decision_preserves_primary_error_when_rollback_fails(tmp_path):
    binding = ApprovalBinding.from_call(
        namespace="tools",
        thread_id="thread",
        run_id="run",
        principal="user",
        tool_call_id="call",
        tool_name="write",
        tool_revision="v1",
        policy_version="v1",
        arguments={},
        resources={},
        required_permissions=(),
    )
    store = SQLiteApprovalStore(str(tmp_path / "approvals.sqlite3"), clock=lambda: 100)
    store.request(binding, request_id="request", expires_at=200)
    store._conn.execute(
        "CREATE TRIGGER reject_audit BEFORE INSERT ON runtime_approval_events "
        "WHEN NEW.revision=2 "
        "BEGIN SELECT RAISE(ABORT, 'approval primary failure'); END"
    )
    original_connection = store._conn
    proxy = _RollbackFailsAfterCleanup(original_connection)
    store._conn = proxy
    try:
        with pytest.raises(
            sqlite3.IntegrityError, match="approval primary failure"
        ) as raised:
            store.decide("tools", "request", approved=True, decided_by="reviewer")

        assert raised.value is proxy.primary_error
        assert any(
            "SQLite approval rollback also failed (OSError)" in note
            for note in getattr(raised.value, "__notes__", ())
        )
        assert not original_connection.in_transaction
        record = store.get("tools", "request")
        events = store.events("tools", "request")
        assert (record.status, record.revision) == ("pending", 1)
        assert [(event.status, event.revision) for event in events] == [("pending", 1)]

        store._conn = original_connection
        original_connection.execute("DROP TRIGGER reject_audit")
        assert (
            store.decide(
                "tools", "request", approved=True, decided_by="reviewer"
            ).status
            == "approved"
        )
    finally:
        store._conn = original_connection
        store.close()
