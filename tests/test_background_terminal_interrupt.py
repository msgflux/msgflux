"""Terminal task records should not retain pending interrupt requests."""

import threading

import msgflux as mf
import pytest
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tasks import InMemoryTaskStore, SQLiteTaskStore


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_terminal_settlement_after_interrupt_clears_pending_flag(
    store_kind, outcome, tmp_path
):
    started = threading.Event()
    release = threading.Event()

    @mf.tool_config(background=True)
    def worker() -> str:
        """Wait for the test to allow completion after an interrupt request."""
        started.set()
        assert release.wait(timeout=2)
        if outcome == "failed":
            raise RuntimeError("worker failed after interrupt request")
        return "completed despite late interrupt"

    database_path = tmp_path / "tasks.sqlite3"
    task_store = (
        InMemoryTaskStore()
        if store_kind == "memory"
        else SQLiteTaskStore(str(database_path))
    )
    library = ToolLibrary(
        name="terminal_interrupt", tools=[worker], task_store=task_store
    )
    task_id = None
    future = None
    try:
        launch = library([("launch", "worker", {})]).tool_calls[0].result
        task_id = launch.split("task_id='")[1].split("'")[0]
        future = library.get_background_dispatcher().get_task_future(task_id)
        assert started.wait(timeout=2)

        interrupt = (
            library([("interrupt", "task_interrupt", {"task_id": task_id})])
            .tool_calls[0]
            .result
        )
        assert interrupt["status"] == "interrupt_requested"
        assert task_store.get(task_id).metadata["interrupt_requested"] is True

        release.set()
        if outcome == "failed":
            with pytest.raises(RuntimeError, match="worker failed after interrupt"):
                future.result(timeout=2)
        else:
            assert future.result(timeout=2) == "completed despite late interrupt"

        task = task_store.get(task_id)
        assert task.status == outcome
        if outcome == "completed":
            assert task.result == "completed despite late interrupt"
            assert task.error is None
        else:
            assert task.result is None
            assert task.error == "worker failed after interrupt request"
        assert task.metadata["interrupt_requested"] is False
    finally:
        release.set()
        if future is not None:
            try:
                future.result(timeout=2)
            except RuntimeError as error:
                assert outcome == "failed"
                assert str(error) == "worker failed after interrupt request"
        close = getattr(task_store, "close", None)
        if close is not None:
            close()

    if store_kind == "sqlite" and task_id is not None:
        reopened = SQLiteTaskStore(str(database_path))
        try:
            task = reopened.get(task_id)
            assert task.status == outcome
            if outcome == "completed":
                assert task.result == "completed despite late interrupt"
            else:
                assert task.error == "worker failed after interrupt request"
            assert task.metadata["interrupt_requested"] is False
        finally:
            reopened.close()


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
@pytest.mark.parametrize("terminal_status", ["completed", "failed", "interrupted"])
def test_interrupt_racing_with_terminal_settlement_reports_latest_status(
    store_kind, terminal_status, tmp_path
):
    from msgflux.tools.builtin.task_tool import TaskInterruptTool

    database_path = tmp_path / "interrupt_race.sqlite3"
    task_store = (
        InMemoryTaskStore()
        if store_kind == "memory"
        else SQLiteTaskStore(str(database_path))
    )
    task_store.create("worker", task_id="race")
    original_request_interrupt = task_store.request_interrupt

    def settle_then_request(task_id):
        if terminal_status == "completed":
            task_store.complete(task_id, "settled first")
        elif terminal_status == "failed":
            task_store.fail(task_id, "settled first")
        else:
            task_store.interrupt(task_id, reason="settled first")
        return original_request_interrupt(task_id)

    task_store.request_interrupt = settle_then_request

    class Handle:
        def get_task_store(self):
            return task_store

        def get_task_future(self, task_id):
            return None

    try:
        result = TaskInterruptTool()("race", handle=Handle())
        task = task_store.get("race")
        assert result["status"] == terminal_status
        assert result["message"] == (
            "Task is interrupted."
            if terminal_status == "interrupted"
            else "Task already reached a terminal state."
        )
        assert task.status == terminal_status
        assert task.metadata["interrupt_requested"] is False
        assert not any(
            activity.summary == "Interrupt requested."
            for activity in task_store.list_activity("race")
        )
    finally:
        close = getattr(task_store, "close", None)
        if close is not None:
            close()

    if store_kind == "sqlite":
        reopened = SQLiteTaskStore(str(database_path))
        try:
            task = reopened.get("race")
            assert task.status == terminal_status
            assert task.metadata["interrupt_requested"] is False
            assert not any(
                activity.summary == "Interrupt requested."
                for activity in reopened.list_activity("race")
            )
        finally:
            reopened.close()


def test_interrupt_store_failure_does_not_cancel_real_queued_dispatch(tmp_path):
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor

    from msgflux._private.executor import Executor
    from msgflux.tools.builtin.task_tool import TaskInterruptTool
    from msgflux.tools.handles import ToolLibraryHandle

    blockers_release = threading.Event()
    blocker_started = threading.Event()
    effect_ran = threading.Event()

    @mf.tool_config(background=True)
    def queued_worker() -> str:
        """Record execution if the queued background task gets scheduled."""
        effect_ran.set()
        return "executed"

    task_store = SQLiteTaskStore(str(tmp_path / "queued-interrupt.sqlite3"))
    library = ToolLibrary(
        name="queued_interrupt", tools=[queued_worker], task_store=task_store
    )
    executor = Executor.get_instance()
    previous_pool = executor.thread_pool
    executor.thread_pool = ThreadPoolExecutor(
        max_workers=1, initializer=executor._mark_thread_pool_worker
    )
    blocker = executor.submit(
        lambda: (blocker_started.set(), blockers_release.wait(timeout=30))
    )
    future = None
    try:
        assert blocker_started.wait(timeout=5)
        launch = library.run("queued_worker", {})
        task_id = launch.split("task_id='")[1].split("'")[0]
        future = library.get_background_dispatcher().get_task_future(task_id)
        assert future is not None
        assert not future.running()

        with task_store._lock:
            task_store._conn.execute(
                "CREATE TRIGGER reject_interrupt BEFORE UPDATE ON tasks "
                "WHEN NEW.status = 'interrupted' BEGIN "
                "SELECT RAISE(ABORT, 'interrupt persistence failed'); END"
            )
            task_store._conn.commit()
        with pytest.raises(
            sqlite3.IntegrityError, match="interrupt persistence failed"
        ):
            TaskInterruptTool()(task_id, handle=ToolLibraryHandle(library))

        assert not future.cancelled()
        assert task_store.get(task_id).status == "queued"
        with task_store._lock:
            task_store._conn.execute("DROP TRIGGER reject_interrupt")
            task_store._conn.commit()

        retry = library.run("task_interrupt", {"task_id": task_id})
        assert retry["status"] == "interrupted"
        assert future.cancelled()
        assert task_store.get(task_id).status == "interrupted"
        blockers_release.set()
        blocker.result(timeout=5)
        assert not effect_ran.is_set()
    finally:
        blockers_release.set()
        blocker.result(timeout=5)
        if future is not None and not future.cancelled():
            future.exception(timeout=5)
        executor.thread_pool.shutdown(wait=True)
        executor.thread_pool = previous_pool
        task_store.close()
