from __future__ import annotations

from msgflux.tasks import InMemoryTaskStore, SQLiteTaskStore


def test_in_memory_summaries_do_not_deepcopy_task_result():
    class ExplodingDeepcopy:
        def __deepcopy__(self, memo):
            raise AssertionError("summary query copied result")

    store = InMemoryTaskStore()
    store.create(
        "worker",
        task_id="task",
        metadata={"thread_id": "thread-a", "root_run_id": "run-a"},
    )
    store._tasks["task"].result = ExplodingDeepcopy()

    summaries = store.list_summaries(thread_id="thread-a", run_id="run-a")

    assert [item.task_id for item in summaries] == ["task"]


def test_sqlite_summaries_do_not_deserialize_result_and_survive_reopen(tmp_path):
    path = str(tmp_path / "tasks.sqlite")
    store = SQLiteTaskStore(path)
    store.create(
        "child",
        task_id="root-child",
        metadata={"thread_id": "thread-a", "root_run_id": "run-a"},
    )
    store.create(
        "child",
        task_id="direct-child",
        metadata={"thread_id": "thread-a", "parent_run_id": "run-a"},
    )
    store.complete("root-child", "large result")
    store.fail("direct-child", "tool failed")
    store._conn.execute(
        "UPDATE tasks SET result = ? WHERE task_id = ?", ("{malformed", "root-child")
    )
    store._conn.commit()
    store.close()

    reopened = SQLiteTaskStore(path)
    try:
        summaries = reopened.list_summaries(thread_id="thread-a", run_id="run-a")
        assert {item.task_id for item in summaries} == {"root-child", "direct-child"}
        assert {item.status for item in summaries} == {"completed", "failed"}
    finally:
        reopened.close()


def test_sqlite_summary_order_uses_updated_at_then_task_id(tmp_path):
    store = SQLiteTaskStore(str(tmp_path / "tasks.sqlite"))
    try:
        for task_id in ("a", "z", "m"):
            store.create(
                "worker",
                task_id=task_id,
                metadata={"thread_id": "thread-a", "root_run_id": "run-a"},
            )
        store._conn.execute(
            "UPDATE tasks SET updated_at='2025-01-01' WHERE task_id='a'"
        )
        store._conn.execute(
            "UPDATE tasks SET updated_at='2025-01-02' WHERE task_id IN ('z','m')"
        )
        store._conn.commit()

        assert [
            item.task_id
            for item in store.list_summaries(thread_id="thread-a", run_id="run-a")
        ] == ["z", "m", "a"]
    finally:
        store.close()


def test_in_memory_unfinished_query_does_not_copy_task_result():
    class ExplodingDeepcopy:
        def __deepcopy__(self, memo):
            raise AssertionError("idle query copied task result")

    store = InMemoryTaskStore()
    store.create("worker", task_id="task", metadata={"thread_id": "thread-a"})
    store._tasks["task"].result = ExplodingDeepcopy()

    assert store.has_unfinished_for_thread(thread_id="thread-a") is True


def test_sqlite_unfinished_query_does_not_deserialize_task_result(tmp_path):
    store = SQLiteTaskStore(str(tmp_path / "tasks.sqlite"))
    try:
        store.create("worker", task_id="task", metadata={"thread_id": "thread-a"})
        store._conn.execute(
            "UPDATE tasks SET result = ? WHERE task_id = ?", ("{malformed", "task")
        )
        store._conn.commit()

        assert store.has_unfinished_for_thread(thread_id="thread-a") is True
    finally:
        store.close()
