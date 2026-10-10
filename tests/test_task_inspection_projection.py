"""Read-path regressions for lightweight task inspection tools."""

import json

import msgflux as mf
import pytest

from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tasks import InMemoryTaskStore, SQLiteTaskStore
from msgflux.tools.builtin import TaskActivityTool


@mf.tool_config(background=True)
def _unused_background_tool(value: str) -> str:
    """Register task inspection tools for these library-level tests."""
    return value


@pytest.mark.parametrize("provider", ["memory", "sqlite"])
def test_task_inspection_tools_omit_results_and_preserve_task_details(
    provider, tmp_path
):
    if provider == "memory":
        store = InMemoryTaskStore()
    else:
        store = SQLiteTaskStore(str(tmp_path / "tasks.sqlite3"))

    try:
        store.create(
            "worker",
            task_id="completed-task",
            metadata={"owner": "team-a", "nested": {"trace": "kept"}},
        )
        store.set_running("completed-task", stage="index", message="Indexing")
        store.update_progress("completed-task", current=3, total=4)
        store.complete("completed-task", {"items": ["a", "b"]})

        store.create(
            "worker",
            task_id="failed-task",
            metadata={"task_kind": "agent", "owner": "team-b"},
        )
        store.set_running("failed-task", stage="fetch", message="Fetching")
        store.update_progress("failed-task", current=1, total=2)
        store.fail("failed-task", "upstream unavailable")
        store.add_activity("failed-task", kind="message", summary="Retry scheduled")

        library = ToolLibrary(
            name="inspect", tools=[_unused_background_tool], task_store=store
        )
        completed = (
            library([("status-complete", "task_status", {"task_id": "completed-task"})])
            .tool_calls[0]
            .result
        )
        failed = (
            library([("status-failed", "task_status", {"task_id": "failed-task"})])
            .tool_calls[0]
            .result
        )
        completed_list = (
            library([("list-completed", "task_list", {"status": "completed"})])
            .tool_calls[0]
            .result
        )
        failed_list = (
            library([("list-failed", "task_list", {"status": "failed"})])
            .tool_calls[0]
            .result
        )

        assert set(completed) == {
            "task_id",
            "tool_name",
            "status",
            "progress",
            "started_at",
            "elapsed_seconds",
        }
        assert completed["task_id"] == "completed-task"
        assert completed["tool_name"] == "worker"
        assert completed["status"] == "completed"
        assert completed["progress"]["stage"] == "index"
        assert completed["progress"]["percent"] == 75.0
        assert completed["started_at"]
        assert isinstance(completed["elapsed_seconds"], float)

        assert set(failed) == {
            "task_id",
            "tool_name",
            "status",
            "progress",
            "started_at",
            "elapsed_seconds",
            "error",
        }
        assert failed["status"] == "failed"
        assert failed["error"] == "upstream unavailable"
        assert failed["progress"]["stage"] == "fetch"
        assert failed["started_at"]
        assert isinstance(failed["elapsed_seconds"], float)

        internal_store = library.get_handle().get_task_store()
        assert (
            internal_store.get("completed-task").metadata["nested"]["trace"] == "kept"
        )
        assert internal_store.get("failed-task").metadata["owner"] == "team-b"

        assert [item["task_id"] for item in completed_list] == ["completed-task"]
        assert set(completed_list[0]) == set(completed)
        assert [item["task_id"] for item in failed_list] == ["failed-task"]
        assert set(failed_list[0]) == set(failed)
        assert library(
            [("status-missing", "task_status", {"task_id": "missing"})]
        ).tool_calls[0].result == {"task_id": "missing", "status": "not_found"}
        assert (
            library([("list-empty", "task_list", {"status": "paused"})])
            .tool_calls[0]
            .result
            == []
        )

        for call_id in range(2):
            assert library(
                [(f"output-{call_id}", "task_output", {"task_id": "completed-task"})]
            ).tool_calls[0].result == {"items": ["a", "b"]}
            assert library(
                [(f"wait-{call_id}", "task_wait", {"task_id": "completed-task"})]
            ).tool_calls[0].result == {"items": ["a", "b"]}
        assert store.get("completed-task").result == {"items": ["a", "b"]}
    finally:
        if provider == "sqlite":
            store.close()


def test_sqlite_projection_skips_malformed_result_after_reopen(tmp_path):
    path = str(tmp_path / "malformed.sqlite3")
    store = SQLiteTaskStore(path)
    store.create("worker", task_id="durable")
    store.complete("durable", {"valid": True})
    store._conn.execute(
        "UPDATE tasks SET result = ? WHERE task_id = ?", ("{malformed", "durable")
    )
    store._conn.commit()
    store.close()

    reopened = SQLiteTaskStore(path)
    try:
        one = reopened.get("durable", include_result=False)
        many = reopened.list(status="completed", include_result=False)
        assert one is not None and one.result is None
        assert [item.task_id for item in many] == ["durable"]
        assert many[0].result is None
        with pytest.raises(json.JSONDecodeError):
            reopened.get("durable")
        assert (
            reopened._conn.execute(
                "SELECT result FROM tasks WHERE task_id='durable'"
            ).fetchone()[0]
            == "{malformed"
        )
    finally:
        reopened.close()


def test_sqlite_outputs_and_wait_repeat_after_reopen(tmp_path):
    path = str(tmp_path / "repeat.sqlite3")
    store = SQLiteTaskStore(path)
    store.create("worker", task_id="repeated")
    expected = {"rows": [1, 2], "source": "durable"}
    store.complete("repeated", expected)
    store.close()

    reopened = SQLiteTaskStore(path)
    try:
        library = ToolLibrary(
            name="reopened", tools=[_unused_background_tool], task_store=reopened
        )
        for index in range(3):
            status = (
                library([(f"status-{index}", "task_status", {"task_id": "repeated"})])
                .tool_calls[0]
                .result
            )
            assert set(status) == {
                "task_id",
                "tool_name",
                "status",
                "progress",
                "started_at",
                "elapsed_seconds",
            }
            assert (
                library([(f"output-{index}", "task_output", {"task_id": "repeated"})])
                .tool_calls[0]
                .result
                == expected
            )
            assert (
                library([(f"wait-{index}", "task_wait", {"task_id": "repeated"})])
                .tool_calls[0]
                .result
                == expected
            )
    finally:
        reopened.close()


def test_status_and_list_skip_activity_reads_while_task_activity_remains_available(
    monkeypatch,
):
    class ExplodingDeepcopy:
        def __deepcopy__(self, memo):
            raise AssertionError("task_activity inspection copied task result")

    store = InMemoryTaskStore()
    store.create("worker", task_id="activity-task", metadata={"task_kind": "agent"})
    store.complete("activity-task", "kept output")
    store._tasks["activity-task"].result = ExplodingDeepcopy()
    store.add_activity("activity-task", kind="tool_call", summary="worker_tool({})")
    activity_reads = {"last": 0, "list": 0}
    result_reads = []
    original_get_last = store.get_last_activity
    original_list_activity = store.list_activity
    original_get = store.get

    def record_get(task_id, *, include_result=True):
        result_reads.append(include_result)
        return original_get(task_id, include_result=include_result)

    def record_get_last(task_id):
        activity_reads["last"] += 1
        return original_get_last(task_id)

    def record_list(task_id, *, limit=None):
        activity_reads["list"] += 1
        return original_list_activity(task_id, limit=limit)

    monkeypatch.setattr(store, "get_last_activity", record_get_last)
    monkeypatch.setattr(store, "list_activity", record_list)
    monkeypatch.setattr(store, "get", record_get)
    library = ToolLibrary(
        name="activity-inspect", tools=[_unused_background_tool], task_store=store
    )

    status = (
        library([("status", "task_status", {"task_id": "activity-task"})])
        .tool_calls[0]
        .result
    )
    listing = library([("list", "task_list", {})]).tool_calls[0].result
    assert set(status) == {
        "task_id",
        "tool_name",
        "status",
        "progress",
        "started_at",
        "elapsed_seconds",
    }
    assert all(set(item) == set(status) for item in listing)
    assert activity_reads == {"last": 0, "list": 0}

    activity = TaskActivityTool()(task_id="activity-task", handle=library.get_handle())
    assert "ToolCall: worker_tool({})" in activity
    assert activity_reads == {"last": 0, "list": 1}
    assert result_reads and all(
        include_result is False for include_result in result_reads
    )
