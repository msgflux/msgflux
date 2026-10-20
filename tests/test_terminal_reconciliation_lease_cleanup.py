"""Terminal checkpoint reconciliation releases its claimed lease on failure."""

from unittest.mock import Mock
import sqlite3
import time

import pytest

from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore
from msgflux.runtime.context import execution_context
from msgflux.runtime.events import EventType
from msgflux.tasks import InMemoryTaskStore, SQLiteTaskStore
import msgflux.runtime.background as background_module


def _terminal_task():
    clock = [0.0]
    checkpoints = InMemoryCheckpointStore()
    tasks = InMemoryTaskStore()
    model = Mock(model_type="chat_completion")
    worker = Agent(name="worker", model=model, checkpoint_store=checkpoints)
    worker.tool_config = {"background": True}
    library = ToolLibrary(name="reconcile", tools=[worker], task_store=tasks)
    task = tasks.create(
        "worker",
        task_id="reconcile-task",
        metadata={
            "task_kind": "agent",
            "checkpoint_namespace": "worker",
            "checkpoint_thread_id": "thread",
            "checkpoint_run_id": "reconcile-task",
            "checkpoint_store_id": checkpoints.routing_id,
            "inbox_store_id": library.get_agent_inbox().store.routing_id,
            "task_resume_params": {},
            "workspace_reference": None,
        },
    )
    tasks._clock = lambda: clock[0]
    assert tasks.claim_worker(task.task_id, "old", lease_seconds=1)
    checkpoints.save_state(
        "worker",
        "thread",
        task.task_id,
        {
            "status": "completed",
            "task_result": {"value": "committed result"},
            "_checkpoint": {"revision": 0},
        },
    )
    clock[0] = 2.0
    library.get_background_dispatcher().lease_seconds = 10.0
    return clock, checkpoints, tasks, model, library, task


@pytest.mark.parametrize("release_error", [False, True])
def test_reconcile_start_event_failure_releases_lease_for_immediate_retry(
    monkeypatch, release_error
):
    clock, checkpoints, tasks, model, library, task = _terminal_task()
    original_emit = background_module.emit_event
    original_release = tasks.release_worker
    observer_error = RuntimeError("observer failed")

    def fail_reconciled_start(event_type, data):
        if event_type == EventType.TASK_START and data.get("reconciled"):
            raise observer_error
        return original_emit(event_type, data)

    def release_then_fail(task_id, owner_id):
        released = original_release(task_id, owner_id)
        assert released
        if release_error:
            raise RuntimeError("release observer failed")
        return released

    monkeypatch.setattr(background_module, "emit_event", fail_reconciled_start)
    monkeypatch.setattr(tasks, "release_worker", release_then_fail)
    with execution_context(checkpoint_store=checkpoints):
        with pytest.raises(RuntimeError, match="observer failed") as caught:
            library.reconcile_agent_task(task.task_id)
    assert caught.value is observer_error

    record = tasks.get(task.task_id)
    checkpoint = checkpoints.load_state("worker", "thread", task.task_id)
    lease = tasks.get_worker_lease(task.task_id)
    assert record.status == "running"
    assert record.result is None
    assert checkpoint["status"] == "completed"
    assert checkpoint["task_result"] == {"value": "committed result"}
    assert lease is None or lease.expires_at <= clock[0]
    assert model.call_count == 0

    monkeypatch.setattr(background_module, "emit_event", original_emit)
    monkeypatch.setattr(tasks, "release_worker", original_release)
    with execution_context(checkpoint_store=checkpoints):
        assert "reconciled" in library.reconcile_agent_task(task.task_id)
    assert tasks.get(task.task_id).status == "completed"
    assert tasks.get(task.task_id).result == "committed result"
    assert tasks.get_worker_lease(task.task_id) is None
    assert model.call_count == 0


def test_reconcile_complete_write_failure_retries_after_sqlite_reopen(
    tmp_path,
):
    task_path = tmp_path / "reconcile-task.sqlite3"
    checkpoint_path = tmp_path / "reconcile-checkpoint.sqlite3"
    inbox_path = tmp_path / "reconcile-inbox.sqlite3"
    checkpoints = SQLiteCheckpointStore(checkpoint_path)
    tasks = SQLiteTaskStore(task_path)
    inbox_store = SQLiteAgentInboxStore(inbox_path)
    model = Mock(model_type="chat_completion")
    worker = Agent(name="worker", model=model, checkpoint_store=checkpoints)
    worker.tool_config = {"background": True}
    library = ToolLibrary(name="reconcile-sqlite", tools=[worker], task_store=tasks)
    library.set_agent_inbox(AgentInbox(owner="root", store=inbox_store))
    task = tasks.create(
        "worker",
        task_id="sqlite-reconcile-task",
        metadata={
            "task_kind": "agent",
            "checkpoint_namespace": "worker",
            "checkpoint_thread_id": "thread",
            "checkpoint_run_id": "sqlite-reconcile-task",
            "checkpoint_store_id": checkpoints.routing_id,
            "inbox_store_id": library.get_agent_inbox().store.routing_id,
            "task_resume_params": {},
            "workspace_reference": None,
        },
    )
    assert tasks.claim_worker(task.task_id, "old", lease_seconds=0.001)
    tasks._conn.execute(
        "UPDATE task_worker_leases SET expires_at = 0 WHERE task_id = ?",
        (task.task_id,),
    )
    tasks._conn.commit()
    checkpoints.save_state(
        "worker",
        "thread",
        task.task_id,
        {
            "status": "completed",
            "task_result": {"value": "sqlite result"},
            "_checkpoint": {"revision": 0},
        },
    )
    tasks._conn.execute(
        "CREATE TRIGGER reject_reconciled_completion "
        "BEFORE UPDATE OF status ON tasks WHEN NEW.status = 'completed' "
        "BEGIN SELECT RAISE(ABORT, 'reconcile completion rejected'); END"
    )
    tasks._conn.commit()

    try:
        with execution_context(checkpoint_store=checkpoints):
            with pytest.raises(
                sqlite3.IntegrityError, match="reconcile completion rejected"
            ):
                library.reconcile_agent_task(task.task_id)

        record = tasks.get(task.task_id)
        assert record.status == "running"
        assert record.result is None
        lease = tasks.get_worker_lease(task.task_id)
        assert lease is None or lease.expires_at <= time.time()
        assert checkpoints.load_state("worker", "thread", task.task_id)[
            "task_result"
        ] == {"value": "sqlite result"}
        assert model.call_count == 0
    finally:
        checkpoints.close()
        tasks.close()
        inbox_store.close()

    checkpoints = SQLiteCheckpointStore(checkpoint_path)
    tasks = SQLiteTaskStore(task_path)
    inbox_store = SQLiteAgentInboxStore(inbox_path)
    tasks._conn.execute("DROP TRIGGER reject_reconciled_completion")
    tasks._conn.commit()
    try:
        worker = Agent(name="worker", model=model, checkpoint_store=checkpoints)
        worker.tool_config = {"background": True}
        library = ToolLibrary(
            name="reconcile-sqlite-reopened", tools=[worker], task_store=tasks
        )
        library.set_agent_inbox(AgentInbox(owner="root", store=inbox_store))
        with execution_context(checkpoint_store=checkpoints):
            assert "reconciled" in library.reconcile_agent_task(task.task_id)
        record = tasks.get(task.task_id)
        assert record.status == "completed"
        assert record.result == "sqlite result"
        assert tasks.get_worker_lease(task.task_id) is None
        assert model.call_count == 0
    finally:
        checkpoints.close()
        tasks.close()
        inbox_store.close()


def test_reconcile_notice_failure_keeps_committed_result_and_releases_lease(
    monkeypatch,
):
    _, checkpoints, tasks, model, library, task = _terminal_task()
    dispatcher = library.get_background_dispatcher()

    def fail_notice(**_kwargs):
        raise RuntimeError("notice failed")

    monkeypatch.setattr(dispatcher, "publish_task_notification", fail_notice)
    with execution_context(checkpoint_store=checkpoints):
        with pytest.raises(RuntimeError, match="notice failed"):
            library.reconcile_agent_task(task.task_id)

    record = tasks.get(task.task_id)
    assert record.status == "completed"
    assert record.result == "committed result"
    assert tasks.get_worker_lease(task.task_id) is None
    assert model.call_count == 0
