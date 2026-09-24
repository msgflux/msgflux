"""Behavioral contract shared by built-in and custom task-store providers."""

from __future__ import annotations

import threading
import multiprocessing
import os
import time

import pytest

from msgflux.runtime.task_leases import TaskLeaseHeartbeats
from msgflux.tasks import (
    InMemoryTaskStore,
    SQLiteTaskStore,
    TaskHandle,
    TaskStoreProtocol,
)


def _claim_then_exit(path: str, ready) -> None:
    store = SQLiteTaskStore(path)
    assert store.claim_worker("task-1", "crashed", lease_seconds=0.2)
    ready.set()
    os._exit(0)


@pytest.fixture(params=["memory", "sqlite"])
def task_store(request, tmp_path):
    if request.param == "memory":
        yield InMemoryTaskStore()
        return
    store = SQLiteTaskStore(str(tmp_path / "tasks.sqlite"))
    try:
        yield store
    finally:
        store.close()


def test_task_store_contract_lifecycle_and_activity(task_store):
    assert isinstance(task_store, TaskStoreProtocol)
    task = task_store.create("worker", task_id="task-1", metadata={"kind": "agent"})
    assert task.status == "queued"
    assert task_store.get("task-1").metadata["kind"] == "agent"
    assert [item.task_id for item in task_store.list(status="queued")] == ["task-1"]

    updated = task_store.update_metadata("task-1", {"scope": "child"})
    assert updated.metadata["scope"] == "child"
    activity = task_store.add_activity(
        "task-1", kind="message", summary="Accepted", metadata={"source": "root"}
    )
    assert activity.metadata["source"] == "root"
    assert task_store.get_last_activity("task-1").summary == "Accepted"
    assert task_store.list_activity("task-1", limit=1)[0].summary == "Accepted"

    running = task_store.set_running("task-1", stage="model")
    assert running.status == "running"
    assert (
        task_store.update_progress("task-1", current=1, total=2).progress.percent == 50
    )
    assert task_store.request_interrupt("task-1").metadata["interrupt_requested"]
    assert not task_store.clear_interrupt_request("task-1").metadata[
        "interrupt_requested"
    ]
    completed = task_store.complete("task-1", "done")
    assert completed.status == "completed"
    assert completed.result == "done"


def test_task_store_contract_messages_and_conditional_resume(task_store):
    task_store.create(
        "worker", task_id="task-1", metadata={"checkpoint_run_id": "initial"}
    )
    assert task_store.enqueue_message("task-1", "msg-1", "First")
    assert task_store.enqueue_message("task-1", "msg-1", "Ignored retry")
    assert task_store.enqueue_message("task-1", "msg-2", "Second")
    assert task_store.pending_messages("task-1") == [
        ("msg-1", "First"),
        ("msg-2", "Second"),
    ]
    task_store.ack_messages("task-1", ["msg-1"])
    assert task_store.pending_messages("task-1") == [("msg-2", "Second")]

    task_store.complete("task-1", "done")
    resumed = task_store.requeue(
        "task-1", expected_status="completed", expected_generation=0, run_id="next"
    )
    assert resumed.status == "queued"
    assert resumed.metadata["checkpoint_run_id"] == "next"
    assert resumed.metadata["resume_generation"] == 1
    assert (
        task_store.requeue(
            "task-1",
            expected_status="completed",
            expected_generation=0,
            run_id="racing",
        )
        is None
    )
    assert task_store.pending_messages("task-1") == [("msg-2", "Second")]
    task_store.ack_messages("task-1", ["msg-2"])
    assert task_store.pending_messages("task-1") == []


def test_task_store_contract_worker_lease_and_fencing(task_store):
    current_time = [100.0]
    task_store._clock = lambda: current_time[0]
    task_store.create("worker", task_id="task-1")

    first = task_store.claim_worker("task-1", "owner-a", lease_seconds=30)
    assert first is not None
    assert first.expires_at == 130.0
    assert task_store.get("task-1").status == "running"
    assert task_store.get_worker_lease("task-1") == first
    assert task_store.complete("task-1", "anonymous") is None
    assert task_store.request_interrupt("task-1").metadata["interrupt_requested"]
    assert task_store.clear_interrupt_request("task-1") is not None
    assert (
        task_store.claim_worker(
            "task-1", "owner-b", lease_seconds=30, recover_expired=True
        )
        is None
    )

    current_time[0] = 120.0
    assert task_store.renew_worker("task-1", "owner-a", lease_seconds=30)
    assert task_store.get_worker_lease("task-1").expires_at == 150.0
    current_time[0] = 151.0
    assert not task_store.renew_worker("task-1", "owner-a", lease_seconds=30)
    second = task_store.claim_worker(
        "task-1", "owner-b", lease_seconds=30, recover_expired=True
    )
    assert second is not None
    assert second.owner_id == "owner-b"
    assert task_store.update_progress("task-1", current=1, owner_id="owner-a") is None
    assert task_store.complete("task-1", "stale", owner_id="owner-a") is None
    assert task_store.get("task-1").status == "running"
    assert (
        task_store.complete("task-1", "current", owner_id="owner-b").result == "current"
    )
    assert task_store.get_worker_lease("task-1") is None


def test_sqlite_worker_lease_claims_are_shared_across_instances(tmp_path):
    path = str(tmp_path / "tasks.sqlite")
    first = SQLiteTaskStore(path)
    second = SQLiteTaskStore(path)
    current_time = [100.0]
    first._clock = lambda: current_time[0]
    second._clock = lambda: current_time[0]
    try:
        first.create("worker", task_id="task-1")
        assert first.claim_worker("task-1", "owner-a", lease_seconds=30)
        assert (
            second.claim_worker(
                "task-1", "owner-b", lease_seconds=30, recover_expired=True
            )
            is None
        )
        current_time[0] = 131.0
        assert second.claim_worker(
            "task-1", "owner-b", lease_seconds=30, recover_expired=True
        )
        assert not first.renew_worker("task-1", "owner-a", lease_seconds=30)
    finally:
        first.close()
        second.close()


def test_sqlite_recovers_worker_after_abrupt_process_exit(tmp_path):
    path = str(tmp_path / "tasks.sqlite")
    store = SQLiteTaskStore(path)
    store.create("worker", task_id="task-1")
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_claim_then_exit, args=(path, ready))
    try:
        process.start()
        assert ready.wait(timeout=5)
        process.join(timeout=5)
        assert process.exitcode == 0
        assert store.get("task-1").status == "running"
        assert store.get_worker_lease("task-1").owner_id == "crashed"
        deadline = time.monotonic() + 2
        while store.get_worker_lease("task-1").expires_at > time.time():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert store.claim_worker(
            "task-1", "replacement", lease_seconds=1, recover_expired=True
        )
        assert store.complete("task-1", "stale", owner_id="crashed") is None
        assert store.complete("task-1", "recovered", owner_id="replacement")
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        store.close()


def test_active_worker_renews_lease_with_shared_heartbeat():
    store = InMemoryTaskStore()
    store.create("worker", task_id="task-1")
    handle = TaskHandle("task-1", store)
    renewed = threading.Event()
    original_renew = store.renew_worker

    def observe_renew(*args, **kwargs):
        result = original_renew(*args, **kwargs)
        if result:
            renewed.set()
        return result

    store.renew_worker = observe_renew
    handle.start_worker(lease_seconds=0.3)
    try:
        TaskLeaseHeartbeats.register(handle, lease_seconds=0.3)
        assert renewed.wait(timeout=2)
        assert handle.complete("done").status == "completed"
    finally:
        TaskLeaseHeartbeats.unregister(handle)
