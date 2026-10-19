"""TaskStore preserves transaction failures when rollback also errors."""

import sqlite3

import pytest

from msgflux.tasks import SQLiteTaskStore


@pytest.mark.parametrize(
    ("operation", "trigger", "invoke", "assert_unchanged"),
    [
        (
            "claim_worker",
            "BEFORE INSERT ON task_activity WHEN NEW.summary='Task running.'",
            lambda store: store.claim_worker("task", "owner", lease_seconds=30),
            lambda store, before: (
                store.get("task").status == "queued"
                and store.get_worker_lease("task") is None
                and store.list_activity("task") == before
            ),
        ),
        (
            "update_metadata",
            "BEFORE UPDATE ON tasks WHEN NEW.metadata != OLD.metadata",
            lambda store: store.update_metadata("task", {"new": True}),
            lambda store, before: (
                store.get("task").metadata == {"before": True}
                and store.list_activity("task") == before
            ),
        ),
        (
            "update_progress",
            "BEFORE INSERT ON task_activity WHEN NEW.kind='progress'",
            lambda store: store.update_progress("task", current=1, total=2),
            lambda store, before: (
                store.get("task").status == "queued"
                and store.get("task").progress.current is None
                and store.list_activity("task") == before
            ),
        ),
        (
            "request_interrupt",
            "BEFORE INSERT ON task_activity WHEN NEW.summary='Task interrupted.'",
            lambda store: store.request_interrupt("task"),
            lambda store, before: (
                store.get("task").metadata.get("interrupt_requested") is not True
                and store.list_activity("task") == before
            ),
        ),
        (
            "requeue",
            "BEFORE INSERT ON task_activity WHEN NEW.summary='Task re-queued.'",
            lambda store: store.requeue(
                "task", expected_status="completed", expected_generation=0
            ),
            lambda store, before: (
                store.get("task").status == "completed"
                and store.get("task").result == {"done": True}
                and store.get("task").metadata.get("resume_generation", 0) == 0
                and store.list_activity("task") == before
            ),
        ),
    ],
    ids=[
        "claim-worker",
        "update-metadata",
        "progress-cas",
        "interrupt-cas",
        "requeue-cas",
    ],
)
def test_task_store_preserves_primary_error_if_rollback_fails(
    tmp_path, operation, trigger, invoke, assert_unchanged
):
    store = SQLiteTaskStore(str(tmp_path / f"{operation}.sqlite3"))
    store.create("worker", task_id="task", metadata={"before": True})
    if operation == "requeue":
        store.complete("task", {"done": True})
    activities_before = store.list_activity("task")
    store._conn.execute(
        f"CREATE TRIGGER reject_operation {trigger} "
        "BEGIN SELECT RAISE(ABORT, 'primary operation failure'); END"
    )
    original_connection = store._conn

    class RollbackFailsAfterCleanup:
        write_error = None

        def __getattr__(self, name):
            return getattr(original_connection, name)

        def execute(self, *args, **kwargs):
            try:
                return original_connection.execute(*args, **kwargs)
            except sqlite3.IntegrityError as error:
                self.write_error = error
                raise

        def rollback(self):
            original_connection.rollback()
            raise OSError("secondary rollback failure")

    proxy = RollbackFailsAfterCleanup()
    store._conn = proxy
    try:
        with pytest.raises(
            sqlite3.IntegrityError, match="primary operation failure"
        ) as raised:
            invoke(store)

        assert raised.value is proxy.write_error
        assert any(
            "SQLite rollback also failed (OSError)" in note
            for note in getattr(raised.value, "__notes__", ())
        )
        assert not original_connection.in_transaction
        assert assert_unchanged(store, activities_before)

        store._conn = original_connection
        original_connection.execute("DROP TRIGGER reject_operation")
        assert invoke(store) is not None
    finally:
        store._conn = original_connection
        store.close()
