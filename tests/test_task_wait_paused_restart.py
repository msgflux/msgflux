"""Task waiters report settled states after restart and live-future timeouts."""

import threading

import msgflux as mf
import pytest
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tasks import SQLiteTaskStore, TaskHandle
from msgflux.tools.builtin import TaskWaitTool


def test_task_wait_reports_paused_task_without_live_worker_after_restart(tmp_path):
    path = tmp_path / "tasks.sqlite3"
    store = SQLiteTaskStore(path)
    store.create("worker", task_id="paused-task")
    worker = TaskHandle("paused-task", store)
    worker.start_worker(lease_seconds=60)
    worker.pause(reason="waiting for an operator")
    store.close()

    reopened = SQLiteTaskStore(path)
    library = ToolLibrary(
        name="reopened-task-wait",
        tools=[TaskWaitTool()],
        task_store=reopened,
    )
    try:
        output = (
            library(
                [
                    (
                        "wait-paused-task",
                        "task_wait",
                        {"task_id": "paused-task", "timeout": 0.05},
                    )
                ]
            )
            .tool_calls[0]
            .result
        )
        assert output["status"] == "paused"
        assert output["task_id"] == "paused-task"
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("settled_status", "expected_result"),
    [("paused", None), ("completed", 42)],
)
def test_task_wait_returns_state_settled_during_live_future_timeout(
    settled_status, expected_result
):
    started = threading.Event()
    settle = threading.Event()
    settled_notification = threading.Event()
    release_notification = threading.Event()
    waiter_entered_future = threading.Event()

    @mf.tool_config(background=True, runtime_inputs=["handle"])
    def work(handle: mf.Hidden) -> int:
        """Wait for the test to settle the running task."""
        started.set()
        assert settle.wait(timeout=3)
        if settled_status == "paused":
            raise TaskPauseRequestedError(handle.get_task_id(), "needs review")
        return expected_result

    library = ToolLibrary(name="task-wait-transition", tools=[work])
    library([("dispatch-work", "work", {})])
    task_id = (
        library([("list-work", "task_list", {})]).tool_calls[0].result[0]["task_id"]
    )
    assert started.wait(timeout=2)

    dispatcher = library.get_background_dispatcher()
    future = dispatcher.get_task_future(task_id)
    assert future is not None
    original_get_future = dispatcher.get_task_future
    original_publish = dispatcher.publish_task_notification

    class ObservedFuture:
        def result(self, timeout=None):
            waiter_entered_future.set()
            return future.result(timeout=timeout)

    dispatcher.get_task_future = lambda requested_id: (
        ObservedFuture()
        if requested_id == task_id
        else original_get_future(requested_id)
    )

    def hold_settled_notification(**kwargs):
        if kwargs["status"] == settled_status:
            settled_notification.set()
            assert release_notification.wait(timeout=3)
        return original_publish(**kwargs)

    dispatcher.publish_task_notification = hold_settled_notification
    wait_output = {}

    def wait_for_task():
        wait_output["result"] = (
            library(
                [
                    (
                        "wait-work",
                        "task_wait",
                        {"task_id": task_id, "timeout": 0.3},
                    )
                ]
            )
            .tool_calls[0]
            .result
        )

    waiter = threading.Thread(target=wait_for_task)
    try:
        waiter.start()
        assert waiter_entered_future.wait(timeout=2)
        settle.set()
        assert settled_notification.wait(timeout=2)
        waiter.join(timeout=2)
        assert not waiter.is_alive()
        if settled_status == "paused":
            assert wait_output["result"]["status"] == "paused"
        else:
            assert wait_output["result"] == expected_result
    finally:
        settle.set()
        release_notification.set()
        dispatcher.get_task_future = original_get_future
        dispatcher.publish_task_notification = original_publish
        if waiter.ident is not None:
            waiter.join(timeout=2)
        try:
            future.result(timeout=2)
        except TaskPauseRequestedError:
            pass
