from __future__ import annotations

import multiprocessing
import sqlite3
from pathlib import Path

import msgspec
import pytest

from msgflux.runtime.service.records import (
    ServiceBusyError,
    ServiceConflictError,
    ServiceThread,
)
from msgflux.runtime.service.store import SQLiteServiceStore


def _claim_worker(path: str, owner_id: str, barrier, result_queue) -> None:
    store = SQLiteServiceStore(path)
    try:
        record = store.get("thread-1", "request-1")
        barrier.wait(timeout=10)
        result_queue.put(store.claim(record, owner_id))
    finally:
        store.close()


def _store(path: Path) -> SQLiteServiceStore:
    store = SQLiteServiceStore(path)
    store.bind_thread(ServiceThread("thread-1", "agent-1"))
    return store


def test_admission_deduplicates_and_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "service.sqlite3"
    store = _store(path)
    first = store.admit("thread-1", "request-1", "hello", "default")
    assert first.receipt.status == "accepted"
    assert store.claim(first, "worker-1")
    claimed = store.get("thread-1", "request-1")
    assert claimed.receipt.revision == 1
    store.close()

    reopened = SQLiteServiceStore(path)
    try:
        persisted = reopened.get("thread-1", "request-1")
        assert persisted == claimed
        duplicate = reopened.admit("thread-1", "request-1", "hello", "default")
        assert duplicate == claimed
        with pytest.raises(ServiceConflictError):
            reopened.admit("thread-1", "request-1", "changed", "default")
        with pytest.raises(ServiceBusyError):
            reopened.admit("thread-1", "request-2", "another", "default")
    finally:
        reopened.close()


def test_thread_binding_and_input_validation() -> None:
    store = SQLiteServiceStore()
    try:
        store.bind_thread(ServiceThread("thread-1", "agent-1"))
        with pytest.raises(ServiceConflictError):
            store.bind_thread(ServiceThread("thread-1", "agent-2"))
        with pytest.raises(ValueError):
            store.admit("thread-1", "request-1", 42, "default")
        with pytest.raises(msgspec.ValidationError):
            msgspec.convert(
                {"thread_id": "x", "agent_id": "a", "unexpected": 1}, type=ServiceThread
            )
    finally:
        store.close()


def test_failed_insert_rolls_back_without_acknowledging_admission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "service.sqlite3"
    store = _store(path)
    store._connection.execute(
        """CREATE TRIGGER reject_admission BEFORE INSERT ON service_admissions
        BEGIN SELECT RAISE(ABORT, 'simulated disk-side failure'); END"""
    )
    with pytest.raises(sqlite3.IntegrityError):
        store.admit("thread-1", "request-1", "hello", "default")
    store.close()

    reopened = SQLiteServiceStore(path)
    try:
        assert reopened.get("thread-1", "request-1") is None
    finally:
        reopened.close()


def test_only_one_process_claims_an_admission(tmp_path: Path) -> None:
    path = tmp_path / "service.sqlite3"
    store = _store(path)
    store.admit("thread-1", "request-1", "hello", "default")
    store.close()

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    result_queue = context.Queue()
    workers = [
        context.Process(
            target=_claim_worker, args=(str(path), f"owner-{i}", barrier, result_queue)
        )
        for i in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)
    try:
        assert all(not worker.is_alive() and worker.exitcode == 0 for worker in workers)
        assert sorted(result_queue.get(timeout=2) for _ in workers) == [False, True]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
        result_queue.close()


def test_finish_is_owner_fenced_and_resume_is_compare_and_swap(tmp_path: Path) -> None:
    store = _store(tmp_path / "service.sqlite3")
    try:
        accepted = store.admit("thread-1", "request-1", "hello", "default")
        assert accepted.receipt.revision == 0
        assert store.claim(accepted, "owner-1")
        running = store.get("thread-1", "request-1")
        first_attempt_receipt = running.receipt
        assert first_attempt_receipt.revision == 1
        with pytest.raises(ServiceConflictError):
            store.finish(running.receipt, "owner-2", "completed")
        paused = store.finish(running.receipt, "owner-1", "paused")
        assert paused.status == "paused"
        assert paused.revision == 2

        paused_record = store.get("thread-1", "request-1")
        resumed = store.prepare_resume(paused_record)
        assert resumed.receipt.status == "accepted"
        assert resumed.owner_id is None
        assert resumed.receipt.revision == 3
        with pytest.raises(ServiceConflictError):
            store.prepare_resume(paused_record)

        # The service owner is reused, but the prior claim revision is fenced.
        assert store.claim(resumed, "owner-1")
        second_attempt = store.get("thread-1", "request-1")
        assert second_attempt.receipt.revision == 4
        with pytest.raises(ServiceConflictError):
            store.finish(first_attempt_receipt, "owner-1", "completed")
        completed = store.finish(second_attempt.receipt, "owner-1", "completed")
        assert completed.status == "completed"
        assert completed.revision == 5
    finally:
        store.close()


def test_failed_admission_can_be_prepared_for_resume(tmp_path: Path) -> None:
    store = _store(tmp_path / "service.sqlite3")
    try:
        accepted = store.admit("thread-1", "request-1", "hello", "default")
        assert store.claim(accepted, "owner-1")
        running = store.get("thread-1", "request-1")
        failed = store.finish(running.receipt, "owner-1", "failed", error="boom")
        failed_record = store.get("thread-1", "request-1")
        assert failed_record.receipt == failed

        resumed = store.prepare_resume(failed_record)
        assert resumed.receipt.status == "accepted"
        assert resumed.receipt.revision == failed.revision + 1
    finally:
        store.close()
