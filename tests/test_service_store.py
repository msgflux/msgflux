from __future__ import annotations

import multiprocessing
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import msgspec
import pytest

from msgflux.runtime.service.records import (
    ServiceBusyError,
    ServiceConflictError,
    ServiceThread,
)
from msgflux.runtime.service.store import SQLiteServiceStore, _enable_wal


def _claim_worker(path: str, owner_id: str, barrier, result_queue) -> None:
    store = SQLiteServiceStore(path)
    try:
        record = store.get("thread-1", "request-1")
        barrier.wait(timeout=10)
        result_queue.put(store.claim(record, owner_id))
    finally:
        store.close()


def _adopt_worker(path: str, barrier, result_queue) -> None:
    store = SQLiteServiceStore(path)
    try:
        barrier.wait(timeout=10)
        record = store.adopt_checkpoint("thread-1", "run-1", "main")
        result_queue.put((record.receipt.request_id, record.receipt.run_id))
    except Exception as error:
        result_queue.put((type(error).__name__, str(error)))
    finally:
        store.close()


def _open_migrated_store(path: Path, barrier: threading.Barrier) -> ServiceThread:
    barrier.wait(timeout=10)
    store = SQLiteServiceStore(path)
    try:
        return store.thread("legacy-thread")
    finally:
        store.close()


class _BusyOnceConnection:
    def __init__(
        self, connection: sqlite3.Connection, error_code: int = sqlite3.SQLITE_BUSY
    ) -> None:
        self.connection = connection
        self.error_code = error_code
        self.attempts = 0

    def execute(self, sql: str):
        if sql == "PRAGMA journal_mode=WAL":
            self.attempts += 1
            if self.attempts == 1:
                error = sqlite3.OperationalError("simulated SQLite error")
                error.sqlite_errorcode = self.error_code
                raise error
        return self.connection.execute(sql)


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
        with pytest.raises(ServiceConflictError):
            store.bind_thread(ServiceThread("thread-1", "agent-1", "/workspace"))
        with pytest.raises(ValueError):
            store.bind_thread(ServiceThread("thread-2", "agent-1", "relative"))
        with pytest.raises(ValueError):
            store.admit("thread-1", "request-1", 42, "default")
        with pytest.raises(msgspec.ValidationError):
            msgspec.convert(
                {"thread_id": "x", "agent_id": "a", "unexpected": 1}, type=ServiceThread
            )
    finally:
        store.close()


def test_legacy_thread_table_migrates_and_preserves_bindings(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE service_threads (thread_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO service_threads (thread_id, agent_id) VALUES (?, ?)",
        ("legacy-thread", "agent-legacy"),
    )
    connection.commit()
    connection.close()

    store = SQLiteServiceStore(path)
    try:
        assert store.thread("legacy-thread") == ServiceThread(
            "legacy-thread", "agent-legacy", None
        )
        assert store.threads() == (ServiceThread("legacy-thread", "agent-legacy"),)
        columns = {
            row["name"]
            for row in store._connection.execute("PRAGMA table_info(service_threads)")
        }
        assert "cwd" in columns
    finally:
        store.close()


def test_concurrent_legacy_schema_migration_serializes_openers(tmp_path: Path) -> None:
    path = tmp_path / "legacy-concurrent.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE service_threads (thread_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO service_threads (thread_id, agent_id) VALUES (?, ?)",
        ("legacy-thread", "agent-legacy"),
    )
    connection.commit()
    connection.close()

    barrier = threading.Barrier(2)
    with ThreadPoolExecutor(max_workers=2) as executor:
        openers = [
            executor.submit(_open_migrated_store, path, barrier) for _ in range(2)
        ]
        assert [opener.result(timeout=15) for opener in openers] == [
            ServiceThread("legacy-thread", "agent-legacy"),
            ServiceThread("legacy-thread", "agent-legacy"),
        ]


def test_wal_setup_retries_busy_once(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "retry.sqlite3")
    busy_once = _BusyOnceConnection(connection)
    try:
        _enable_wal(busy_once)
        assert busy_once.attempts == 2
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        connection.close()


def test_wal_setup_does_not_retry_unrelated_operational_error(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "no-retry.sqlite3")
    failing = _BusyOnceConnection(connection, error_code=sqlite3.SQLITE_IOERR)
    try:
        with pytest.raises(sqlite3.OperationalError, match="simulated SQLite error"):
            _enable_wal(failing)
        assert failing.attempts == 1
    finally:
        connection.close()


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


def test_checkpoint_adoption_is_idempotent_and_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "service.sqlite3"
    store = _store(path)
    adopted = store.adopt_checkpoint("thread-1", "run-1", "main")
    assert adopted.receipt.request_id == "checkpoint:run-1"
    assert adopted.receipt.run_id == "run-1"
    assert adopted.receipt.status == "running"
    assert adopted.prompt == ""
    assert adopted.owner_id == "checkpoint"
    assert store.get_for_run("thread-1", "run-1") == adopted
    assert store.adopt_checkpoint("thread-1", "run-1", "main") == adopted
    store.close()

    reopened = SQLiteServiceStore(path)
    try:
        assert reopened.get_for_run("thread-1", "run-1") == adopted
        with pytest.raises(ServiceConflictError, match="namespace"):
            reopened.adopt_checkpoint("thread-1", "run-1", "other")
        with pytest.raises(ServiceBusyError):
            reopened.admit("thread-1", "request-2", "new prompt", "main")
    finally:
        reopened.close()


def test_checkpoint_adoption_rejects_reserved_request_collision_and_active_run(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "service.sqlite3")
    try:
        store.admit("thread-1", "checkpoint:run-1", "ordinary", "main")
        with pytest.raises(ServiceConflictError, match="reserved"):
            store.adopt_checkpoint("thread-1", "run-1", "main")
        with pytest.raises(ServiceBusyError):
            store.adopt_checkpoint("thread-1", "run-2", "main")
        with pytest.raises(KeyError):
            store.get_for_run("missing-thread", "run-1")
    finally:
        store.close()


def test_concurrent_checkpoint_adoption_returns_one_journal_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "service.sqlite3"
    _store(path).close()
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    result_queue = context.Queue()
    workers = [
        context.Process(target=_adopt_worker, args=(str(path), barrier, result_queue))
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=15)
    try:
        assert all(not worker.is_alive() and worker.exitcode == 0 for worker in workers)
        assert [result_queue.get(timeout=2) for _ in workers] == [
            ("checkpoint:run-1", "run-1"),
            ("checkpoint:run-1", "run-1"),
        ]
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
        result_queue.close()
