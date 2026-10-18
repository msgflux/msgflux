"""SQLite worker updates across adapters preserve concurrent progress fields."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event, Lock, get_ident

from msgflux.tasks import SQLiteTaskStore


def test_sqlite_worker_progress_updates_merge_across_adapters(tmp_path):
    database_path = str(tmp_path / "worker-progress.sqlite3")
    first = SQLiteTaskStore(database_path)
    second = SQLiteTaskStore(database_path)
    first.create("worker", task_id="shared")
    lease = first.claim_worker("shared", "worker-owner", lease_seconds=30)
    assert lease is not None
    both_read = Barrier(2)

    def synchronize_initial_read(store):
        original_get = store.get
        used = False
        lock = Lock()

        def get(task_id, *, include_result=True):
            nonlocal used
            task = original_get(task_id, include_result=include_result)
            with lock:
                synchronize = not used and task_id == "shared"
                used = used or synchronize
            if synchronize:
                both_read.wait(timeout=5)
            return task

        store.get = get

    synchronize_initial_read(first)
    synchronize_initial_read(second)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            progress = (
                pool.submit(
                    first.update_progress,
                    "shared",
                    stage="download",
                    current=1,
                    total=10,
                    owner_id="worker-owner",
                ),
                pool.submit(
                    second.update_progress,
                    "shared",
                    message="receiving chunks",
                    owner_id="worker-owner",
                ),
            )
            for update in progress:
                assert update.result(timeout=3) is not None

        task = first.get("shared")
        assert task.status == "running"
        assert task.progress.stage == "download"
        assert task.progress.message == "receiving chunks"
        assert task.progress.current == 1
        assert task.progress.total == 10
        assert task.progress.percent == 10
        assert first.get_worker_lease("shared").owner_id == "worker-owner"
        assert first.list_activity("shared")[-1].metadata == task.progress.to_dict()
    finally:
        first.close()
        second.close()

    reopened = SQLiteTaskStore(database_path)
    try:
        assert reopened._conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        task = reopened.get("shared")
        assert task.progress.stage == "download"
        assert task.progress.message == "receiving chunks"
        assert task.progress.current == 1
        assert task.progress.total == 10
        assert task.progress.percent == 10
        assert reopened.get_worker_lease("shared").owner_id == "worker-owner"
        assert reopened.list_activity("shared")[-1].metadata == task.progress.to_dict()
    finally:
        reopened.close()


def test_sqlite_completion_preserves_progress_committed_after_its_read(tmp_path):
    database_path = str(tmp_path / "worker-completion.sqlite3")
    first = SQLiteTaskStore(database_path)
    second = SQLiteTaskStore(database_path)
    first.create("worker", task_id="shared")
    assert first.claim_worker("shared", "worker-owner", lease_seconds=30)
    completion_read = Event()
    progress_committed = Event()
    original_get = first.get
    completion_thread = None
    paused = False

    def pause_completion_after_read(task_id, *, include_result=True):
        nonlocal paused
        task = original_get(task_id, include_result=include_result)
        if task_id == "shared" and get_ident() == completion_thread:
            if not paused:
                paused = True
                completion_read.set()
                assert progress_committed.wait(timeout=2)
        return task

    first.get = pause_completion_after_read

    def complete():
        nonlocal completion_thread
        completion_thread = get_ident()
        return first.complete("shared", "done", owner_id="worker-owner")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            completed = pool.submit(complete)
            assert completion_read.wait(timeout=2)
            progress = second.update_progress(
                "shared",
                stage="finalizing",
                message="writing result",
                current=3,
                total=3,
                owner_id="worker-owner",
            )
            assert progress is not None
            progress_committed.set()
            assert completed.result(timeout=3).result == "done"

        task = first.get("shared")
        assert task.status == "completed"
        assert task.result == "done"
        assert task.progress.stage == "finalizing"
        assert task.progress.message == "writing result"
        assert task.progress.current == 3
        assert task.progress.total == 3
        assert first.get_worker_lease("shared") is None
    finally:
        first.close()
        second.close()

    reopened = SQLiteTaskStore(database_path)
    try:
        assert reopened._conn.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        task = reopened.get("shared")
        assert task.status == "completed"
        assert task.result == "done"
        assert task.progress.stage == "finalizing"
        assert task.progress.message == "writing result"
        assert task.progress.current == 3
        assert task.progress.total == 3
        assert reopened.get_worker_lease("shared") is None
    finally:
        reopened.close()


def test_sqlite_ownerless_late_progress_cannot_resurrect_completed_task(tmp_path):
    database_path = str(tmp_path / "ownerless-late-progress.sqlite3")
    first = SQLiteTaskStore(database_path)
    second = SQLiteTaskStore(database_path)
    first.create("worker", task_id="shared")
    first.set_running("shared")
    progress_read = Event()
    completion_committed = Event()
    original_get = first.get
    progress_thread = None
    paused = False

    def pause_progress_after_read(task_id, *, include_result=True):
        nonlocal paused
        task = original_get(task_id, include_result=include_result)
        if task_id == "shared" and get_ident() == progress_thread:
            if not paused:
                paused = True
                progress_read.set()
                assert completion_committed.wait(timeout=2)
        return task

    first.get = pause_progress_after_read

    def update_progress():
        nonlocal progress_thread
        progress_thread = get_ident()
        return first.update_progress("shared", message="late callback")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            progress = pool.submit(update_progress)
            assert progress_read.wait(timeout=2)
            assert second.complete("shared", "done") is not None
            completion_committed.set()
            assert progress.result(timeout=3) is None

        task = first.get("shared")
        assert task.status == "completed"
        assert task.result == "done"
        assert first.list_activity("shared")[-1].metadata == {"status": "completed"}
    finally:
        first.close()
        second.close()


def test_sqlite_ownerless_late_pause_cannot_overwrite_completion(tmp_path):
    database_path = str(tmp_path / "ownerless-late-pause.sqlite3")
    first = SQLiteTaskStore(database_path)
    second = SQLiteTaskStore(database_path)
    first.create("worker", task_id="shared")
    first.set_running("shared")
    pause_read = Event()
    completion_committed = Event()
    original_get = first.get
    pause_thread = None
    paused = False

    def pause_after_read(task_id, *, include_result=True):
        nonlocal paused
        task = original_get(task_id, include_result=include_result)
        if task_id == "shared" and get_ident() == pause_thread:
            if not paused:
                paused = True
                pause_read.set()
                assert completion_committed.wait(timeout=2)
        return task

    first.get = pause_after_read

    def pause():
        nonlocal pause_thread
        pause_thread = get_ident()
        return first.pause("shared", reason="late callback")

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            paused_task = pool.submit(pause)
            assert pause_read.wait(timeout=2)
            assert second.complete("shared", "done") is not None
            completion_committed.set()
            assert paused_task.result(timeout=3) is None

        task = first.get("shared")
        assert task.status == "completed"
        assert task.result == "done"
        assert first.list_activity("shared")[-1].metadata == {"status": "completed"}
    finally:
        first.close()
        second.close()


def test_sqlite_ownerless_set_running_after_completion_matches_in_memory(tmp_path):
    database_path = str(tmp_path / "set-running-terminal.sqlite3")
    store = SQLiteTaskStore(database_path)
    store.create("worker", task_id="shared")
    assert store.complete("shared", "done").status == "completed"

    try:
        task = store.set_running("shared", stage="restarted")
        assert task is not None
        assert task.status == "running"
        assert task.result == "done"
        assert task.progress.stage == "restarted"
    finally:
        store.close()


def test_sqlite_concurrent_ownerless_startup_progress_is_merged(tmp_path):
    database_path = str(tmp_path / "ownerless-startup-progress.sqlite3")
    first = SQLiteTaskStore(database_path)
    second = SQLiteTaskStore(database_path)
    first.create("worker", task_id="shared")
    both_read = Barrier(2)

    def synchronize_initial_read(store):
        original_get = store.get
        used = False
        lock = Lock()

        def get(task_id, *, include_result=True):
            nonlocal used
            task = original_get(task_id, include_result=include_result)
            with lock:
                synchronize = not used and task_id == "shared"
                used = used or synchronize
            if synchronize:
                both_read.wait(timeout=5)
            return task

        store.get = get

    synchronize_initial_read(first)
    synchronize_initial_read(second)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            updates = (
                pool.submit(
                    first.update_progress,
                    "shared",
                    stage="download",
                    current=1,
                    total=2,
                ),
                pool.submit(
                    second.update_progress,
                    "shared",
                    message="receiving chunks",
                ),
            )
            assert all(update.result(timeout=3) is not None for update in updates)

        task = first.get("shared")
        activities = [
            item for item in first.list_activity("shared") if item.kind == "progress"
        ]
        assert task.status == "running"
        assert task.progress.stage == "download"
        assert task.progress.message == "receiving chunks"
        assert task.progress.current == 1
        assert task.progress.total == 2
        assert task.progress.percent == 50
        assert len(activities) == 2
        assert activities[-1].metadata == task.progress.to_dict()
    finally:
        first.close()
        second.close()
