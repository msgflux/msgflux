"""Failed recovery observers must not strand a claimed background task."""

import time
import sqlite3
from threading import Event
from unittest.mock import Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux._private.executor import Executor
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.events import EventType
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore
from msgflux.tasks import InMemoryTaskStore, SQLiteTaskStore
import msgflux.runtime.background as background_module
import msgflux.tasks.handle as task_handle_module


class _Model:
    model_type = "chat_completion"

    def __init__(self, release_first_call):
        self.calls = []
        self.release_first_call = release_first_call

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1 and not self.release_first_call.wait(timeout=3):
            raise TimeoutError("Initial model call was not released")
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("initial" if len(self.calls) == 1 else "recovered")
        return response


def _wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for background task")


def _running_checkpoint_task():
    checkpoints = InMemoryCheckpointStore()
    tasks = InMemoryTaskStore()
    release_first_call = Event()
    model = _Model(release_first_call)
    worker = Agent(name="worker", model=model, checkpoint_store=checkpoints)
    worker.tool_config = {"background": True}
    library = ToolLibrary(name="recovery-cleanup", tools=[worker], task_store=tasks)
    dispatch = library([("start", "worker", {"task": "Start"})])
    task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]
    dispatcher = library.get_background_dispatcher()
    first_future = dispatcher.get_task_future(task_id)
    assert first_future is not None
    release_first_call.set()
    first_future.result(timeout=3)
    _wait_until(lambda: dispatcher.get_task_future(task_id) is None)
    task = tasks.get(task_id)
    thread_id = task.metadata["checkpoint_thread_id"]
    run_id = task.metadata["checkpoint_run_id"]

    checkpoint = checkpoints.load_state("worker", thread_id, run_id)
    checkpoint["status"] = "running"
    checkpoint["messages"]["items"] = [
        item
        for item in checkpoint["messages"]["items"]
        if not (item.get("type") == "turn" and item.get("event") == "complete")
    ]
    checkpoints.save_state("worker", thread_id, run_id, checkpoint)
    assert tasks.requeue(task_id, expected_status="completed", expected_generation=0)

    clock = [100.0]
    tasks._clock = lambda: clock[0]
    assert tasks.claim_worker(task_id, "crashed", lease_seconds=10)
    clock[0] = 111.0
    library.get_background_dispatcher().lease_seconds = 12
    return checkpoints, tasks, model, library, task_id, clock, thread_id, run_id


@pytest.mark.parametrize("failure_point", ["claim_update", "recovered_start"])
def test_recovery_observer_failure_releases_lease_and_allows_retry(
    monkeypatch, failure_point
):
    checkpoints, tasks, model, library, task_id, clock, thread_id, run_id = (
        _running_checkpoint_task()
    )
    original_background_emit = background_module.emit_event
    original_handle_emit = task_handle_module.emit_event
    observer_error = RuntimeError(f"{failure_point} observer failed")

    def fail_recovered_start(event_type, data):
        if event_type == EventType.TASK_START and data.get("recovered"):
            raise observer_error
        return original_background_emit(event_type, data)

    def fail_claim_update(event_type, data):
        if event_type == EventType.TASK_UPDATE and data.get("status") == "running":
            raise observer_error
        return original_handle_emit(event_type, data)

    if failure_point == "claim_update":
        monkeypatch.setattr(task_handle_module, "emit_event", fail_claim_update)
    else:
        monkeypatch.setattr(background_module, "emit_event", fail_recovered_start)

    with pytest.raises(RuntimeError) as caught:
        library.recover_agent_task(task_id, message="Continue", worker_stopped=True)
    assert caught.value is observer_error
    record = tasks.get(task_id)
    assert record.status == "running"
    assert tasks.get_worker_lease(task_id).expires_at <= clock[0]
    assert checkpoints.load_state("worker", thread_id, run_id)["status"] == "running"
    assert len(model.calls) == 1

    monkeypatch.setattr(background_module, "emit_event", original_background_emit)
    monkeypatch.setattr(task_handle_module, "emit_event", original_handle_emit)
    assert "recovered" in library.recover_agent_task(
        task_id, message="Continue", worker_stopped=True
    )
    _wait_until(lambda: tasks.get(task_id).status == "completed")
    assert tasks.get(task_id).result == "recovered"
    assert tasks.get_worker_lease(task_id) is None
    assert len(model.calls) == 2


def test_terminal_recovery_claim_update_failure_releases_lease(monkeypatch):
    checkpoints = InMemoryCheckpointStore()
    tasks = InMemoryTaskStore()
    model = Mock(model_type="chat_completion")
    worker = Agent(name="worker", model=model, checkpoint_store=checkpoints)
    worker.tool_config = {"background": True}
    library = ToolLibrary(
        name="terminal-claim-cleanup", tools=[worker], task_store=tasks
    )
    task = tasks.create(
        "worker",
        task_id="terminal-claim-task",
        metadata={
            "task_kind": "agent",
            "checkpoint_namespace": "worker",
            "checkpoint_thread_id": "thread",
            "checkpoint_run_id": "terminal-claim-task",
            "checkpoint_store_id": checkpoints.routing_id,
            "inbox_store_id": library.get_agent_inbox().store.routing_id,
            "task_resume_params": {},
            "workspace_reference": None,
        },
    )
    clock = [0.0]
    tasks._clock = lambda: clock[0]
    assert tasks.claim_worker(task.task_id, "old", lease_seconds=1)
    checkpoints.save_state(
        "worker",
        "thread",
        task.task_id,
        {
            "status": "completed",
            "task_result": {"value": "durable"},
            "_checkpoint": {"revision": 0},
        },
    )
    clock[0] = 2.0
    original_emit = task_handle_module.emit_event
    observer_error = RuntimeError("claim update failed")

    def fail_claim_update(event_type, data):
        if event_type == EventType.TASK_UPDATE and data.get("status") == "running":
            raise observer_error
        return original_emit(event_type, data)

    monkeypatch.setattr(task_handle_module, "emit_event", fail_claim_update)
    from msgflux.runtime.context import execution_context

    with execution_context(checkpoint_store=checkpoints):
        with pytest.raises(RuntimeError) as caught:
            library.reconcile_agent_task(task.task_id)
    assert caught.value is observer_error
    assert tasks.get_worker_lease(task.task_id).expires_at <= clock[0]
    assert tasks.get(task.task_id).status == "running"
    assert tasks.get(task.task_id).result is None
    assert checkpoints.load_state("worker", "thread", task.task_id)["task_result"] == {
        "value": "durable"
    }
    assert model.call_count == 0


def test_recovery_submit_failure_settles_task_and_releases_lease(monkeypatch):
    _checkpoints, tasks, model, library, task_id, _clock, _thread_id, _run_id = (
        _running_checkpoint_task()
    )
    submit_error = RuntimeError("recovery executor unavailable")

    def reject_submit(self, function, *args, **kwargs):
        raise submit_error

    monkeypatch.setattr(Executor, "submit", reject_submit)
    with pytest.raises(RuntimeError) as caught:
        library.recover_agent_task(task_id, message="Continue", worker_stopped=True)

    assert caught.value is submit_error
    record = tasks.get(task_id)
    assert record.status == "failed"
    assert "recovery executor unavailable" in record.error
    lease = tasks.get_worker_lease(task_id)
    assert lease is None or lease.expires_at <= tasks._clock()
    assert len(model.calls) == 1


def test_recovery_submit_and_failure_settlement_errors_release_sqlite_lease(
    tmp_path, monkeypatch, caplog
):
    task_store = SQLiteTaskStore(tmp_path / "recovery-submit-failure.sqlite3")
    checkpoint_store = SQLiteCheckpointStore(tmp_path / "recovery-checkpoint.sqlite3")
    inbox_store = SQLiteAgentInboxStore(tmp_path / "recovery-inbox.sqlite3")
    model = Mock(model_type="chat_completion")
    worker = Agent(name="worker", model=model, checkpoint_store=checkpoint_store)
    worker.tool_config = {"background": True}
    library = ToolLibrary(
        name="recovery-submit-failure", tools=[worker], task_store=task_store
    )
    library.set_agent_inbox(AgentInbox(owner="root", store=inbox_store))
    task = task_store.create(
        "worker",
        task_id="recovery-submit-task",
        metadata={
            "task_kind": "agent",
            "checkpoint_namespace": "worker",
            "checkpoint_thread_id": "thread",
            "checkpoint_run_id": "recovery-submit-task",
            "checkpoint_store_id": checkpoint_store.routing_id,
            "inbox_store_id": inbox_store.routing_id,
            "task_resume_params": {},
            "workspace_reference": None,
        },
    )
    assert task_store.claim_worker(task.task_id, "old", lease_seconds=1)
    task_store._conn.execute(
        "UPDATE task_worker_leases SET expires_at = 0 WHERE task_id = ?",
        (task.task_id,),
    )
    task_store._conn.commit()
    checkpoint_store.save_state(
        "worker",
        "thread",
        task.task_id,
        {"status": "running", "_checkpoint": {"revision": 1}},
    )
    task_store._conn.execute(
        "CREATE TRIGGER reject_recovery_failure "
        "BEFORE UPDATE OF status ON tasks WHEN NEW.status = 'failed' "
        "BEGIN SELECT RAISE(ABORT, 'failure settlement rejected'); END"
    )
    task_store._conn.commit()
    submit_error = RuntimeError("executor rejected recovered worker")

    def reject_submit(self, function, *args, **kwargs):
        raise submit_error

    monkeypatch.setattr(Executor, "submit", reject_submit)
    try:
        with pytest.raises(RuntimeError) as caught:
            library.recover_agent_task(
                task.task_id, message="Continue", worker_stopped=True
            )
        assert caught.value is submit_error
        record = task_store.get(task.task_id)
        lease = task_store.get_worker_lease(task.task_id)
        assert record.status == "running"
        assert lease is None or lease.expires_at <= time.time()
        assert "Could not persist background task failure" in caplog.text
        assert "failure settlement rejected" in caplog.text
        assert (
            checkpoint_store.load_state("worker", "thread", task.task_id)["status"]
            == "running"
        )
        assert model.call_count == 0
    finally:
        checkpoint_store.close()
        task_store.close()
        inbox_store.close()
