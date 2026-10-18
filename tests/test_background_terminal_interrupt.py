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
        assert result["message"] == "Task already reached a terminal state."
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
