import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import msgspec
import pytest

from msgflux.tasks import (
    InMemoryTaskStore,
    SQLiteTaskStore,
    TaskIdCollisionError,
    TaskStore,
)
from msgflux.tools.shell import ShellCommandResult, ShellResult


@dataclass
class _DataclassResult:
    count: int


class _StructResult(msgspec.Struct):
    count: int


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_DataclassResult(7), {"count": 7}),
        (_StructResult(7), {"count": 7}),
        ({"items": [_StructResult(7)]}, {"items": [{"count": 7}]}),
        (
            ShellResult(
                results=(
                    ShellCommandResult(
                        status="exited", stderr="synthetic stderr", returncode=17
                    ),
                )
            ),
            {
                "results": [
                    {
                        "status": "exited",
                        "stdout": "",
                        "stderr": "synthetic stderr",
                        "returncode": 17,
                    }
                ]
            },
        ),
    ],
)
def test_sqlite_task_store_preserves_structured_results_after_reopen(
    tmp_path, result, expected
):
    path = tmp_path / "structured-tasks.sqlite3"
    store = SQLiteTaskStore(str(path))
    try:
        store.create("worker", task_id="structured")
        store.complete("structured", result)
        assert store.get("structured").result == expected
    finally:
        store.close()

    reopened = SQLiteTaskStore(str(path))
    try:
        assert reopened.get("structured").result == expected
        assert reopened.list(status="completed")[0].result == expected
    finally:
        reopened.close()


def test_sqlite_task_store_retains_opaque_result_string_fallback(tmp_path):
    class OpaqueResult:
        def __str__(self):
            return "opaque result"

    store = SQLiteTaskStore(str(tmp_path / "opaque-tasks.sqlite3"))
    try:
        store.create("worker", task_id="opaque")
        store.complete("opaque", {"value": OpaqueResult()})
        assert store.get("opaque").result == {"value": "opaque result"}
    finally:
        store.close()


def test_in_memory_task_store_rejects_duplicate_task_id():
    store = InMemoryTaskStore()
    store.create("worker", task_id="task_1")

    with pytest.raises(TaskIdCollisionError, match="task_1"):
        store.create("other_worker", task_id="task_1")

    task = store.get("task_1")
    assert task is not None
    assert task.tool_name == "worker"


def test_sqlite_task_store_rejects_duplicate_task_id(tmp_path):
    store = SQLiteTaskStore(path=str(tmp_path / "tasks.sqlite3"))
    store.create("worker", task_id="task_1")

    with pytest.raises(TaskIdCollisionError, match="task_1"):
        store.create("other_worker", task_id="task_1")

    task = store.get("task_1")
    assert task is not None
    assert task.tool_name == "worker"
    store.close()


def test_sqlite_task_store_create_rolls_back_when_initial_activity_fails(tmp_path):
    path = tmp_path / "tasks.sqlite3"
    store = SQLiteTaskStore(str(path))
    store._conn.execute(
        "CREATE TRIGGER reject_activity BEFORE INSERT ON task_activity "
        "BEGIN SELECT RAISE(ABORT, 'injected activity failure'); END"
    )

    with pytest.raises(sqlite3.IntegrityError, match="injected activity failure"):
        store.create("worker", task_id="orphan")

    assert not store._conn.in_transaction
    assert store.get("orphan") is None
    assert store.list_activity("orphan") == []

    store._conn.execute("DROP TRIGGER reject_activity")
    created = store.create("worker", task_id="subsequent")
    assert created.task_id == "subsequent"
    assert store.get("orphan") is None
    store.close()

    reopened = SQLiteTaskStore(str(path))
    try:
        assert reopened.get("orphan") is None
        assert reopened.get("subsequent") is not None
    finally:
        reopened.close()


def test_sqlite_task_store_activity_failure_rolls_back_and_keeps_connection_usable(
    tmp_path,
):
    store = SQLiteTaskStore(str(tmp_path / "tasks.sqlite3"))
    store.create("worker", task_id="existing")
    previous = store.list_activity("existing")
    store._conn.execute(
        "CREATE TRIGGER reject_activity BEFORE INSERT ON task_activity "
        "WHEN NEW.kind = 'rejected' "
        "BEGIN SELECT RAISE(ABORT, 'injected activity failure'); END"
    )

    with pytest.raises(sqlite3.IntegrityError, match="injected activity failure"):
        store.add_activity("existing", kind="rejected", summary="fail")

    assert not store._conn.in_transaction
    assert store.list_activity("existing") == previous
    store.add_activity("existing", kind="message", summary="ok")
    assert [activity.kind for activity in store.list_activity("existing")] == [
        "status",
        "message",
    ]
    store.close()


def test_sqlite_task_store_create_preserves_activity_error_if_rollback_fails(
    tmp_path,
):
    store = SQLiteTaskStore(str(tmp_path / "tasks.sqlite3"))
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
            raise OSError("injected rollback failure")

    try:
        original_connection.execute(
            "CREATE TRIGGER reject_activity BEFORE INSERT ON task_activity "
            "BEGIN SELECT RAISE(ABORT, 'injected activity failure'); END"
        )
        connection_proxy = RollbackFailsAfterCleanup()
        store._conn = connection_proxy

        with pytest.raises(
            sqlite3.IntegrityError, match="injected activity failure"
        ) as raised:
            store.create("worker", task_id="orphan")

        assert raised.value is connection_proxy.write_error
        assert any(
            "SQLite rollback also failed (OSError)" in note
            for note in getattr(raised.value, "__notes__", ())
        )
        assert original_connection.in_transaction is False
        assert store.get("orphan") is None
        assert store.list_activity("orphan") == []

        store._conn = original_connection
        original_connection.execute("DROP TRIGGER reject_activity")
        assert store.create("worker", task_id="subsequent").task_id == "subsequent"
    finally:
        store._conn = original_connection
        store.close()


def test_sqlite_task_store_roundtrip_and_reopen(tmp_path):
    path = tmp_path / "tasks.sqlite3"
    store = SQLiteTaskStore(path=str(path))

    task = store.create(
        "worker",
        task_id="task_1",
        metadata={"thread_id": "user_42", "checkpoint_run_id": "run_1"},
    )
    assert task.status == "queued"

    store.set_running("task_1", stage="start", message="Starting")
    store.update_progress("task_1", current=1, total=2)
    store.update_metadata("task_1", {"checkpoint_run_id": "run_2"})
    store.add_activity(
        "task_1",
        kind="message",
        summary="Root message: continue",
        metadata={"direction": "root_to_task"},
    )
    store.complete("task_1", {"answer": "done"})
    store.close()

    reopened = SQLiteTaskStore(path=str(path))
    restored = reopened.get("task_1")

    assert restored is not None
    assert restored.status == "completed"
    assert restored.result == {"answer": "done"}
    assert restored.metadata["thread_id"] == "user_42"
    assert restored.metadata["checkpoint_run_id"] == "run_2"
    assert restored.progress.current == 1
    assert restored.progress.total == 2
    assert [item.kind for item in reopened.list_activity("task_1")] == [
        "status",
        "status",
        "progress",
        "message",
        "status",
    ]
    reopened.close()


def test_task_store_sqlite_factory(tmp_path):
    store = TaskStore.sqlite(path=str(tmp_path / "tasks.sqlite3"))

    assert isinstance(store, SQLiteTaskStore)
    store.close()


def test_sqlite_task_store_serializes_concurrent_updates(tmp_path):
    store = SQLiteTaskStore(path=str(tmp_path / "tasks.sqlite3"))
    store.create("worker", task_id="task_1")

    def update(index: int) -> None:
        store.update_progress(
            "task_1",
            stage=f"step_{index}",
            current=index,
            total=20,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(update, range(1, 21)))

    task = store.get("task_1")
    activity = store.list_activity("task_1")

    assert task is not None
    assert task.status == "running"
    assert len([item for item in activity if item.kind == "progress"]) == 20
    store.close()
