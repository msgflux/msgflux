"""Atomic interruption of queued tasks across task-store providers."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from msgflux.tasks import InMemoryTaskStore, SQLiteTaskStore
from msgflux.tasks.lease import TaskLease


def _install_queued_lease(task_store, task_id: str) -> TaskLease:
    lease = TaskLease(task_id, "recovery-owner", 0.0)
    if isinstance(task_store, InMemoryTaskStore):
        task_store._worker_leases[task_id] = lease
    else:
        with task_store._lock:
            task_store._conn.execute(
                "INSERT INTO task_worker_leases (task_id, owner_id, expires_at) "
                "VALUES (?, ?, ?)",
                (task_id, lease.owner_id, lease.expires_at),
            )
            task_store._conn.commit()
    return lease


def test_interrupt_terminalizes_queued_task_without_lease(task_store):
    task_store.create("worker", task_id="queued")

    interrupted = task_store.request_interrupt("queued")

    assert interrupted.status == "interrupted"
    assert interrupted.completed_at is not None
    assert interrupted.metadata["interrupt_requested"] is False
    assert interrupted.metadata["interrupt_reason"]
    activities = task_store.list_activity("queued")
    assert activities[-1].metadata["status"] == "interrupted"
    assert activities[-1].metadata["reason"] == interrupted.metadata["interrupt_reason"]
    assert task_store.claim_worker("queued", "worker", lease_seconds=10) is None


def test_interrupt_keeps_queued_task_with_lease_cooperative(task_store):
    task_store.create("worker", task_id="queued-with-lease")
    lease = _install_queued_lease(task_store, "queued-with-lease")

    updated = task_store.request_interrupt("queued-with-lease")

    assert updated.status == "queued"
    assert updated.metadata["interrupt_requested"] is True
    assert task_store.get_worker_lease("queued-with-lease") == lease
    assert task_store.list_activity("queued-with-lease")[-1].summary == (
        "Interrupt requested."
    )


def test_interrupt_keeps_running_and_paused_tasks_cooperative(task_store):
    task_store.create("worker", task_id="running")
    task_store.claim_worker("running", "owner", lease_seconds=10)
    running = task_store.request_interrupt("running")
    assert running.status == "running"
    assert running.metadata["interrupt_requested"] is True

    task_store.create("worker", task_id="paused")
    task_store.pause("paused", reason="waiting")
    paused = task_store.request_interrupt("paused")
    assert paused.status == "paused"
    assert paused.metadata["interrupt_requested"] is True


@pytest.mark.parametrize("iteration", range(12))
def test_claim_worker_races_atomically_with_queued_interrupt(task_store, iteration):
    task_id = f"race-{iteration}"
    task_store.create("worker", task_id=task_id)
    barrier = Barrier(2)

    def claim():
        barrier.wait(timeout=3)
        return task_store.claim_worker(task_id, "owner", lease_seconds=10)

    def interrupt():
        barrier.wait(timeout=3)
        return task_store.request_interrupt(task_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claim_future = pool.submit(claim)
        interrupt_future = pool.submit(interrupt)
        lease = claim_future.result(timeout=5)
        requested = interrupt_future.result(timeout=5)

    current = task_store.get(task_id)
    if lease is not None:
        assert current.status == "running"
        assert requested.status == "running"
        assert requested.metadata["interrupt_requested"] is True
        assert task_store.get_worker_lease(task_id) == lease
    else:
        assert current.status == "interrupted"
        assert requested.status == "interrupted"
        assert requested.metadata["interrupt_requested"] is False
        assert task_store.get_worker_lease(task_id) is None
