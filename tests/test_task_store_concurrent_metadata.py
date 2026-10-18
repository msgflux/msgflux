"""Concurrent task metadata updates must not overwrite sibling adapter writes."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, get_ident

from msgflux.tasks import SQLiteTaskStore


def _assert_reopened_integrity(database_path, task_id, check):
    reopened = SQLiteTaskStore(database_path)
    try:
        assert reopened._conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        check(reopened.get(task_id), reopened)
    finally:
        reopened.close()


def test_sqlite_adapters_merge_disjoint_metadata_updates(tmp_path):
    database_path = str(tmp_path / "tasks.sqlite3")
    first = SQLiteTaskStore(database_path)
    second = SQLiteTaskStore(database_path)
    first.create("worker", task_id="shared")
    first_read = Event()
    second_write_started = Event()
    original_get = first.get
    initial_read = True

    def pause_after_first_read(task_id, *, include_result=True):
        nonlocal initial_read
        task = original_get(task_id, include_result=include_result)
        if initial_read and task_id == "shared":
            initial_read = False
            first_read.set()
            assert second_write_started.wait(timeout=5)
        return task

    def observe_second_transaction(statement):
        if statement.lstrip().upper().startswith("BEGIN"):
            second_write_started.set()

    first.get = pause_after_first_read
    second._conn.set_trace_callback(observe_second_transaction)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_update = pool.submit(first.update_metadata, "shared", {"first": 1})
            assert first_read.wait(timeout=5)
            second_update = pool.submit(second.update_metadata, "shared", {"second": 2})
            updates = (first_update, second_update)
            for update in updates:
                assert update.result(timeout=3) is not None

        task = first.get("shared")
        assert task.metadata["first"] == 1
        assert task.metadata["second"] == 2
    finally:
        first.close()
        second.close()

    def assert_merged(task, _store):
        assert task.metadata["first"] == 1
        assert task.metadata["second"] == 2

    _assert_reopened_integrity(database_path, "shared", assert_merged)


def test_metadata_update_racing_with_interrupt_preserves_latest_metadata(
    tmp_path,
):
    database_path = str(tmp_path / "interrupt-metadata.sqlite3")
    first = SQLiteTaskStore(database_path)
    second = SQLiteTaskStore(database_path)
    first.create("worker", task_id="shared")
    read_complete = Event()
    interrupt_started = Event()
    interrupt_begin_started = Event()
    interrupt_complete = Event()
    original_get = first.get
    update_thread = None

    def pause_after_update_read(task_id, *, include_result=True):
        result = original_get(task_id, include_result=include_result)
        if task_id == "shared" and get_ident() == update_thread:
            read_complete.set()
            assert interrupt_started.wait(timeout=2)
            if first._conn.in_transaction:
                assert interrupt_begin_started.wait(timeout=2)
            else:
                assert interrupt_complete.wait(timeout=2)
        return result

    first.get = pause_after_update_read

    def observe_interrupt_transaction(statement):
        if statement.lstrip().upper().startswith("BEGIN"):
            interrupt_begin_started.set()

    second._conn.set_trace_callback(observe_interrupt_transaction)

    def update_metadata():
        nonlocal update_thread
        update_thread = get_ident()
        return first.update_metadata("shared", {"external": "kept"})

    def request_interrupt():
        assert read_complete.wait(timeout=2)
        interrupt_started.set()
        result = second.request_interrupt("shared")
        interrupt_complete.set()
        return result

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            update = pool.submit(update_metadata)
            interrupt = pool.submit(request_interrupt)
            assert update.result(timeout=3) is not None
            assert interrupt.result(timeout=3).metadata["interrupt_requested"] is True

        task = first.get("shared")
        assert task.status == "queued"
        assert task.metadata["external"] == "kept"
        assert task.metadata["interrupt_requested"] is True
        assert any(
            activity.summary == "Interrupt requested."
            for activity in first.list_activity("shared")
        )
    finally:
        first.close()
        second.close()

    def assert_persisted(task, store):
        assert task.status == "queued"
        assert task.metadata["external"] == "kept"
        assert task.metadata["interrupt_requested"] is True
        assert any(
            activity.summary == "Interrupt requested."
            for activity in store.list_activity("shared")
        )

    _assert_reopened_integrity(database_path, "shared", assert_persisted)


def test_metadata_update_failure_rolls_back_and_store_remains_usable(tmp_path):
    database_path = str(tmp_path / "metadata-rollback.sqlite3")
    store = SQLiteTaskStore(database_path)
    store.create("worker", task_id="shared", metadata={"original": True})
    original_save = store._save_task

    def fail_save(_task):
        raise RuntimeError("metadata save failed")

    store._save_task = fail_save
    try:
        try:
            store.update_metadata("shared", {"partial": "must roll back"})
        except RuntimeError as error:
            assert str(error) == "metadata save failed"
        else:
            raise AssertionError("expected metadata update failure")

        store._save_task = original_save
        task = store.get("shared")
        assert task.metadata == {"original": True}
        updated = store.update_metadata("shared", {"recovered": True})
        assert updated.metadata == {"original": True, "recovered": True}
    finally:
        store.close()

    def assert_rolled_back_then_recovered(task, _store):
        assert task.metadata == {"original": True, "recovered": True}

    _assert_reopened_integrity(
        database_path, "shared", assert_rolled_back_then_recovered
    )
