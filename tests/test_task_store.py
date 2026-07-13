from concurrent.futures import ThreadPoolExecutor

import pytest

from msgflux.tasks import (
    InMemoryTaskStore,
    SQLiteTaskStore,
    TaskIdCollisionError,
    TaskStore,
)


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


def test_sqlite_task_store_holds_lock_across_read_modify_write(tmp_path):
    store = SQLiteTaskStore(path=str(tmp_path / "tasks.sqlite3"))
    store.create("worker", task_id="task_1")
    original_get = store.get

    def checked_get(task_id: str):
        assert store._lock._is_owned()
        return original_get(task_id)

    store.get = checked_get
    task = store.update_progress("task_1", stage="running", message="working")

    assert task.progress.stage == "running"
    assert task.progress.message == "working"
    store.close()
