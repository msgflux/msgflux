"""Failures while initially claiming a background task must settle it."""

from threading import Event

import msgflux as mf
import pytest

from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.events import EventType
from msgflux.runtime.task_leases import TaskLeaseHeartbeats
from msgflux.tasks import InMemoryTaskStore
import msgflux.tasks.handle as task_handle_module


@pytest.mark.parametrize("failure_point", ["claim_event", "heartbeat_start"])
def test_initial_claim_failure_settles_task_without_running_tool(
    monkeypatch, failure_point
):
    effects = []
    entered_failure = Event()
    release_failure = Event()
    observer_error = RuntimeError(f"{failure_point} failed")

    @mf.tool_config(background=True)
    def worker() -> str:
        """Record whether the background tool body began."""
        effects.append("ran")
        return "done"

    tasks = InMemoryTaskStore()
    library = ToolLibrary(
        name="initial-claim-failure", tools=[worker], task_store=tasks
    )

    if failure_point == "claim_event":
        original_emit = task_handle_module.emit_event

        def fail_running_update(event_type, data):
            if event_type == EventType.TASK_UPDATE and data.get("status") == "running":
                entered_failure.set()
                if not release_failure.wait(timeout=3):
                    raise TimeoutError("Test did not release observer")
                raise observer_error
            return original_emit(event_type, data)

        monkeypatch.setattr(task_handle_module, "emit_event", fail_running_update)
    else:

        def fail_heartbeat_start(cls, handle, *, lease_seconds):
            entered_failure.set()
            if not release_failure.wait(timeout=3):
                raise TimeoutError("Test did not release heartbeat start")
            raise observer_error

        monkeypatch.setattr(
            TaskLeaseHeartbeats, "register", classmethod(fail_heartbeat_start)
        )

    try:
        dispatch = library([("start", "worker", {})])
        task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]
        assert entered_failure.wait(timeout=3)
        future = library.get_background_dispatcher().get_task_future(task_id)
        assert future is not None
        release_failure.set()
        with pytest.raises(RuntimeError) as caught:
            future.result(timeout=3)
        assert caught.value is observer_error

        task = tasks.get(task_id)
        assert task.status == "failed"
        assert observer_error.args[0] in task.error
        assert tasks.get_worker_lease(task_id) is None
        assert effects == []
    finally:
        release_failure.set()
