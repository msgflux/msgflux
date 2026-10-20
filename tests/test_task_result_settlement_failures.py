"""Task result persistence errors must not leave a completed worker running."""

import sqlite3
import threading

import msgflux as mf
import pytest
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.events import EventType
from msgflux.tasks import SQLiteTaskStore
import msgflux.tasks.handle as task_handle_module
from msgflux.tools.builtin import TaskOutputTool


def test_task_result_write_failure_settles_failed_and_survives_reopen(tmp_path):
    side_effects = []
    release_worker = threading.Event()

    @mf.tool_config(background=True)
    def worker() -> str:
        """Record a completed effect and return a result for persistence."""
        side_effects.append("effect completed")
        assert release_worker.wait(timeout=3)
        return "finished"

    path = tmp_path / "task-result-failure.sqlite3"
    task_store = SQLiteTaskStore(path)
    task_store._conn.execute(
        "CREATE TRIGGER reject_completed_result "
        "BEFORE UPDATE OF status ON tasks WHEN NEW.status = 'completed' "
        "BEGIN SELECT RAISE(ABORT, 'task result write rejected'); END"
    )
    task_store._conn.commit()
    library = ToolLibrary(
        name="task-result-failure",
        tools=[worker],
        task_store=task_store,
    )

    try:
        library([("dispatch", "worker", {})])
        task_id = (
            library([("list", "task_list", {})]).tool_calls[0].result[0]["task_id"]
        )
        future = library.get_background_dispatcher().get_task_future(task_id)
        assert future is not None
        release_worker.set()
        with pytest.raises(sqlite3.IntegrityError, match="task result write rejected"):
            future.result(timeout=3)

        task = task_store.get(task_id)
        assert task is not None
        assert task.status == "failed"
        assert "task result write rejected" in task.error
        assert task.result is None
        assert task_store.get_worker_lease(task_id) is None
        assert side_effects == ["effect completed"]

        failed_notices = [
            item
            for item in library.get_agent_inbox().peek()
            if item.ref == task_id and item.status == "failed"
        ]
        assert len(failed_notices) == 1
        assert not [
            item
            for item in library.get_agent_inbox().peek()
            if item.ref == task_id and item.status == "completed"
        ]

        waited = (
            library([("wait", "task_wait", {"task_id": task_id, "timeout": 0.1})])
            .tool_calls[0]
            .result
        )
        assert waited["status"] == "failed"
        assert "task result write rejected" in waited["error"]
    finally:
        release_worker.set()
        task_store.close()

    reopened = SQLiteTaskStore(path)
    try:
        reader = ToolLibrary(
            name="reopened-task-result-failure",
            tools=[TaskOutputTool()],
            task_store=reopened,
        )
        persisted = reopened.get(task_id)
        assert persisted is not None
        assert persisted.status == "failed"
        assert "task result write rejected" in persisted.error
        assert persisted.result is None
        assert reopened.get_worker_lease(task_id) is None
        output = (
            reader([("output", "task_output", {"task_id": task_id})])
            .tool_calls[0]
            .result
        )
        assert output["status"] == "failed"
        assert "task result write rejected" in output["error"]
        assert side_effects == ["effect completed"]
    finally:
        reopened.close()


def test_result_write_error_remains_primary_if_failure_settlement_fails(
    tmp_path, caplog
):
    side_effects = []
    release_worker = threading.Event()

    @mf.tool_config(background=True)
    def worker() -> str:
        """Record a completed effect and return a result for persistence."""
        side_effects.append("effect completed")
        assert release_worker.wait(timeout=3)
        return "finished"

    task_store = SQLiteTaskStore(tmp_path / "task-double-failure.sqlite3")
    task_store._conn.executescript(
        "CREATE TRIGGER reject_completed_status "
        "BEFORE UPDATE OF status ON tasks WHEN NEW.status = 'completed' "
        "BEGIN SELECT RAISE(ABORT, 'completed status write rejected'); END; "
        "CREATE TRIGGER reject_failed_status "
        "BEFORE UPDATE OF status ON tasks WHEN NEW.status = 'failed' "
        "BEGIN SELECT RAISE(ABORT, 'failed status write rejected'); END;"
    )
    task_store._conn.commit()
    library = ToolLibrary(
        name="task-double-failure",
        tools=[worker],
        task_store=task_store,
    )

    try:
        library([("dispatch", "worker", {})])
        task_id = (
            library([("list", "task_list", {})]).tool_calls[0].result[0]["task_id"]
        )
        future = library.get_background_dispatcher().get_task_future(task_id)
        assert future is not None
        release_worker.set()
        with pytest.raises(
            sqlite3.IntegrityError, match="completed status write rejected"
        ):
            future.result(timeout=3)

        task = task_store.get(task_id)
        assert task is not None
        assert task.status == "running"
        assert task.result is None
        assert task_store.get_worker_lease(task_id) is not None
        assert side_effects == ["effect completed"]
        assert "Could not persist background task failure" in caplog.text
        assert "failed status write rejected" in caplog.text
        assert not [
            item for item in library.get_agent_inbox().peek() if item.ref == task_id
        ]
    finally:
        release_worker.set()
        task_store.close()


def test_task_end_event_error_does_not_downgrade_committed_result(
    tmp_path, monkeypatch
):
    side_effects = []
    release_worker = threading.Event()
    task_end_error = RuntimeError("task end observer failed")

    @mf.tool_config(background=True)
    def worker() -> str:
        """Record a completed effect and return a durable result."""
        side_effects.append("effect completed")
        assert release_worker.wait(timeout=3)
        return "finished"

    task_store = SQLiteTaskStore(tmp_path / "task-end-event-failure.sqlite3")
    library = ToolLibrary(
        name="task-end-event-failure",
        tools=[worker],
        task_store=task_store,
    )
    original_emit_event = task_handle_module.emit_event

    def fail_task_end(event_type, payload):
        if event_type == EventType.TASK_END:
            raise task_end_error
        return original_emit_event(event_type, payload)

    monkeypatch.setattr(task_handle_module, "emit_event", fail_task_end)

    try:
        library([("dispatch", "worker", {})])
        task_id = (
            library([("list", "task_list", {})]).tool_calls[0].result[0]["task_id"]
        )
        future = library.get_background_dispatcher().get_task_future(task_id)
        assert future is not None
        release_worker.set()
        with pytest.raises(RuntimeError) as raised:
            future.result(timeout=3)
        assert raised.value is task_end_error

        task = task_store.get(task_id)
        assert task is not None
        assert task.status == "completed"
        assert task.result == "finished"
        assert task_store.get_worker_lease(task_id) is None
        assert side_effects == ["effect completed"]
        assert not [
            item
            for item in library.get_agent_inbox().peek()
            if item.ref == task_id and item.status == "failed"
        ]
    finally:
        release_worker.set()
        task_store.close()

    reopened = SQLiteTaskStore(tmp_path / "task-end-event-failure.sqlite3")
    try:
        reader = ToolLibrary(
            name="reopened-task-end-event-failure",
            tools=[TaskOutputTool()],
            task_store=reopened,
        )
        persisted = reopened.get(task_id)
        assert persisted is not None
        assert persisted.status == "completed"
        assert persisted.result == "finished"
        assert (
            reader([("output", "task_output", {"task_id": task_id})])
            .tool_calls[0]
            .result
            == "finished"
        )
        assert side_effects == ["effect completed"]
    finally:
        reopened.close()
