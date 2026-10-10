"""Behavioral contract shared by built-in and custom task-store providers."""

from __future__ import annotations

import threading
import multiprocessing
import os
import time

import pytest
import msgspec

from msgflux.runtime.task_leases import TaskLeaseHeartbeats
from msgflux.exceptions import TaskLeaseLostError
from msgflux.tasks import (
    InMemoryTaskStore,
    SQLiteTaskStore,
    TaskHandle,
    TaskSummary,
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


def test_task_store_contract_scoped_summaries(task_store):
    task_store.create(
        "child",
        task_id="root-child",
        metadata={"thread_id": "thread-a", "root_run_id": "run-a"},
    )
    task_store.create(
        "child",
        task_id="direct-child",
        metadata={"thread_id": "thread-a", "parent_run_id": "run-a"},
    )
    task_store.create(
        "child",
        task_id="wrong-thread",
        metadata={"thread_id": "thread-b", "root_run_id": "run-a"},
    )
    task_store.create(
        "child",
        task_id="wrong-run",
        metadata={"thread_id": "thread-a", "root_run_id": "run-b"},
    )
    task_store.complete("root-child", "large result")
    task_store.fail("direct-child", "tool failed")

    summaries = task_store.list_summaries(thread_id="thread-a", run_id="run-a")

    assert {item.task_id for item in summaries} == {"root-child", "direct-child"}
    by_id = {item.task_id: item for item in summaries}
    assert by_id["root-child"].status == "completed"
    assert by_id["direct-child"].status == "failed"
    assert by_id["direct-child"].error == "tool failed"
    assert all(isinstance(item, TaskSummary) for item in summaries)
    assert all(
        set(msgspec.to_builtins(item))
        == {"task_id", "tool_name", "status", "updated_at", "error"}
        for item in summaries
    )


def test_task_store_contract_detects_unfinished_thread_work_without_quiescence_guess(
    task_store,
):
    task_store.create("queued", task_id="queued", metadata={"thread_id": "thread-a"})
    task_store.create("running", task_id="running", metadata={"thread_id": "thread-a"})
    task_store.set_running("running")
    task_store.create("paused", task_id="paused", metadata={"thread_id": "thread-a"})
    task_store.set_running("paused")
    task_store.pause("paused", reason="awaiting host reconciliation")
    task_store.create("foreign", task_id="foreign", metadata={"thread_id": "thread-b"})
    task_store.set_running("foreign")

    assert task_store.has_unfinished_for_thread(thread_id="thread-a") is True
    assert task_store.has_unfinished_for_thread(thread_id="thread-b") is True
    assert task_store.has_unfinished_for_thread(thread_id="thread-c") is False

    # A stale/expired lease is not evidence that a durable running task stopped.
    lease_store = task_store
    lease_store._clock = lambda: 100.0
    lease_store.create(
        "expired-lease", task_id="expired-lease", metadata={"thread_id": "thread-c"}
    )
    lease_store.claim_worker("expired-lease", "old-owner", lease_seconds=1)
    lease_store._clock = lambda: 102.0
    assert lease_store.get_worker_lease("expired-lease").expires_at == 101.0
    assert lease_store.has_unfinished_for_thread(thread_id="thread-c") is True

    task_store.complete("queued", "done")
    task_store.complete("running", "done")
    task_store.interrupt("paused", reason="resolved")
    assert task_store.has_unfinished_for_thread(thread_id="thread-a") is False


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


def test_task_store_requeue_tracks_checkpoint_lineage_atomically(task_store):
    task_store.create(
        "worker", task_id="lineage", metadata={"checkpoint_run_id": "run-a"}
    )
    task_store.complete("lineage", "first result")
    resumed_b = task_store.requeue(
        "lineage", expected_status="completed", expected_generation=0, run_id="run-b"
    )
    assert resumed_b.metadata["checkpoint_run_id"] == "run-b"
    assert resumed_b.metadata["checkpoint_origin_run_id"] == "run-a"

    # A retry targeting the current run leaves the original lineage intact.
    task_store.complete("lineage", "second result")
    resumed_b_retry = task_store.requeue(
        "lineage", expected_status="completed", expected_generation=1, run_id="run-b"
    )
    assert resumed_b_retry.metadata["checkpoint_run_id"] == "run-b"
    assert resumed_b_retry.metadata["checkpoint_origin_run_id"] == "run-a"

    task_store.complete("lineage", "third result")
    resumed_c = task_store.requeue(
        "lineage", expected_status="completed", expected_generation=2, run_id="run-c"
    )
    assert resumed_c.metadata["checkpoint_run_id"] == "run-c"
    assert resumed_c.metadata["checkpoint_origin_run_id"] == "run-b"


def test_task_store_requeue_lineage_legacy_and_cas_failure(task_store):
    task_store.create(
        "worker", task_id="legacy", metadata={"checkpoint_run_id": "legacy-run"}
    )
    task_store.complete("legacy", "done")
    assert (
        task_store.requeue(
            "legacy", expected_status="queued", expected_generation=0, run_id="wrong"
        )
        is None
    )
    unchanged = task_store.get("legacy")
    assert unchanged.metadata["checkpoint_run_id"] == "legacy-run"
    assert "checkpoint_origin_run_id" not in unchanged.metadata

    resumed = task_store.requeue(
        "legacy", expected_status="completed", expected_generation=0, run_id="new-run"
    )
    assert resumed.metadata["checkpoint_origin_run_id"] == "legacy-run"

    task_store.create("worker", task_id="no-origin")
    task_store.complete("no-origin", "done")
    fresh = task_store.requeue(
        "no-origin",
        expected_status="completed",
        expected_generation=0,
        run_id="new-run",
    )
    assert fresh.metadata["checkpoint_run_id"] == "new-run"
    assert "checkpoint_origin_run_id" not in fresh.metadata


@pytest.mark.parametrize("previous_run_id", [None, "", 42])
def test_task_store_requeue_clears_stale_origin_for_unknown_previous_run(
    task_store, previous_run_id
):
    metadata = {"checkpoint_origin_run_id": "older-run"}
    if previous_run_id is not None:
        metadata["checkpoint_run_id"] = previous_run_id
    task_store.create("worker", task_id="stale-origin", metadata=metadata)
    task_store.complete("stale-origin", "done")

    resumed = task_store.requeue(
        "stale-origin",
        expected_status="completed",
        expected_generation=0,
        run_id="new-run",
    )

    assert resumed.metadata["checkpoint_run_id"] == "new-run"
    assert "checkpoint_origin_run_id" not in resumed.metadata


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


def test_task_store_contract_release_worker_is_owner_conditional(task_store):
    current_time = [100.0]
    task_store._clock = lambda: current_time[0]
    task_store.create("worker", task_id="task-1")
    assert task_store.claim_worker("task-1", "owner-a", lease_seconds=5)

    current_time[0] = 106.0
    assert task_store.claim_worker(
        "task-1", "owner-b", lease_seconds=30, recover_expired=True
    )
    assert not task_store.release_worker("task-1", "owner-a")
    assert task_store.get_worker_lease("task-1").owner_id == "owner-b"
    assert task_store.release_worker("task-1", "owner-b")
    released = task_store.get_worker_lease("task-1")
    assert released is not None
    assert released.owner_id == "owner-b"
    assert released.expires_at <= current_time[0]
    assert task_store.get("task-1").status == "running"


def test_released_task_handle_is_fenced_and_task_remains_reclaimable(task_store):
    current_time = [100.0]
    task_store._clock = lambda: current_time[0]
    task_store.create("worker", task_id="task-1")
    old_handle = TaskHandle("task-1", task_store)
    old_handle.start_worker(lease_seconds=30)
    assert old_handle.release_worker()

    with pytest.raises(TaskLeaseLostError):
        old_handle.complete("stale")

    replacement = TaskHandle("task-1", task_store)
    replacement.start_worker(lease_seconds=30, recover_expired=True)
    assert replacement.complete("recovered").result == "recovered"


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
