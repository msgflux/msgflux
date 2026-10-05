"""Small SQLite admission journal; conversation and task stores remain separate."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from threading import RLock

import msgspec

from msgflux.runtime.context import new_run_id
from msgflux.runtime.service_records import (
    AdmissionReceipt,
    AdmissionRecord,
    AdmissionStatus,
    ServiceBusyError,
    ServiceConflictError,
    ServiceThread,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS service_threads (
    thread_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS service_admissions (
    thread_id TEXT NOT NULL REFERENCES service_threads(thread_id),
    request_id TEXT NOT NULL, run_id TEXT NOT NULL UNIQUE,
    namespace TEXT NOT NULL, prompt TEXT NOT NULL, fingerprint TEXT NOT NULL,
    status TEXT NOT NULL, owner_id TEXT, error TEXT,
    revision INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (thread_id, request_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS service_active_thread
ON service_admissions(thread_id) WHERE status IN ('accepted', 'running', 'paused');
"""


def validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{label}` must be a non-empty string")


class SQLiteServiceStore:
    """Persist admissions at an explicit host path, or use ``:memory:``.

    This journal has no model history, credentials, tasks, or permission grants.
    A conditional claim fences journal writes, not arbitrary external effects.
    The host owns and closes the store, independently of AgentService.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._lock = RLock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        if str(path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def bind_thread(self, thread: ServiceThread) -> ServiceThread:
        try:
            thread = msgspec.convert(thread, type=ServiceThread, strict=True)
        except (msgspec.ValidationError, TypeError) as error:
            raise ValueError(f"Invalid service thread: {error}") from error
        validate_identifier(thread.thread_id, "thread_id")
        validate_identifier(thread.agent_id, "agent_id")
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO service_threads VALUES (?, ?)",
                (thread.thread_id, thread.agent_id),
            )
            existing = self.thread(thread.thread_id)
            if existing != thread:
                raise ServiceConflictError("The thread belongs to another agent")
        return thread

    def thread(self, thread_id: str) -> ServiceThread:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM service_threads WHERE thread_id=?", (thread_id,)
            ).fetchone()
        if row is None:
            raise KeyError(thread_id)
        return ServiceThread(row["thread_id"], row["agent_id"])

    def threads(self) -> tuple[ServiceThread, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM service_threads ORDER BY thread_id"
            ).fetchall()
        return tuple(ServiceThread(row["thread_id"], row["agent_id"]) for row in rows)

    @staticmethod
    def _record(row: sqlite3.Row) -> AdmissionRecord:
        return AdmissionRecord(
            AdmissionReceipt(
                row["thread_id"],
                row["request_id"],
                row["run_id"],
                row["status"],
                row["error"],
                1,
                row["revision"],
            ),
            row["namespace"],
            row["prompt"],
            row["owner_id"],
        )

    def get(self, thread_id: str, request_id: str) -> AdmissionRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM service_admissions WHERE thread_id=? AND request_id=?",
                (thread_id, request_id),
            ).fetchone()
        return self._record(row) if row is not None else None

    def admit(
        self, thread_id: str, request_id: str, prompt: str, namespace: str
    ) -> AdmissionRecord:
        validate_identifier(thread_id, "thread_id")
        validate_identifier(request_id, "request_id")
        validate_identifier(namespace, "namespace")
        try:
            prompt = msgspec.convert(prompt, type=str, strict=True)
        except (msgspec.ValidationError, TypeError) as error:
            raise ValueError("`prompt` must be a string") from error
        fingerprint = hashlib.sha256(
            msgspec.json.encode([namespace, prompt])
        ).hexdigest()
        with self._lock, self._connection:
            # Serialize admission across independent connections as well as callers.
            self._connection.execute("BEGIN IMMEDIATE")
            self.thread(thread_id)
            existing = self.get(thread_id, request_id)
            if existing is not None:
                if (existing.namespace, existing.prompt) != (namespace, prompt):
                    raise ServiceConflictError("request_id identifies another input")
                return existing
            try:
                self._connection.execute(
                    """INSERT INTO service_admissions
                    (thread_id, request_id, run_id, namespace, prompt,
                     fingerprint, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'accepted')""",
                    (
                        thread_id,
                        request_id,
                        new_run_id(),
                        namespace,
                        prompt,
                        fingerprint,
                    ),
                )
            except sqlite3.IntegrityError as error:
                active = self._connection.execute(
                    """SELECT 1 FROM service_admissions WHERE thread_id=?
                    AND status IN ('accepted', 'running', 'paused')""",
                    (thread_id,),
                ).fetchone()
                if active is not None:
                    raise ServiceBusyError(
                        "The thread has unfinished admitted work"
                    ) from error
                raise
            return self.get(thread_id, request_id)

    def claim(self, record: AdmissionRecord, owner_id: str) -> bool:
        validate_identifier(owner_id, "owner_id")
        record = msgspec.convert(record, type=AdmissionRecord, strict=True)
        with self._lock, self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            changed = self._connection.execute(
                """UPDATE service_admissions
                SET status='running', owner_id=?, revision=revision+1
                WHERE thread_id=? AND request_id=? AND run_id=?
                AND status='accepted' AND revision=?""",
                (
                    owner_id,
                    record.receipt.thread_id,
                    record.receipt.request_id,
                    record.receipt.run_id,
                    record.receipt.revision,
                ),
            )
            return changed.rowcount == 1

    def finish(
        self,
        receipt: AdmissionReceipt,
        owner_id: str,
        status: AdmissionStatus,
        error: str | None = None,
    ) -> AdmissionReceipt:
        if status in {"accepted", "running"}:
            raise ValueError("Finishing an attempt requires a settled status")
        if status not in {"completed", "paused", "interrupted", "failed"}:
            raise ValueError("Unknown admission status")
        validate_identifier(owner_id, "owner_id")
        receipt = msgspec.convert(receipt, type=AdmissionReceipt, strict=True)
        with self._lock, self._connection:
            changed = self._connection.execute(
                """UPDATE service_admissions SET status=?, error=?,
                revision=revision+1
                WHERE thread_id=? AND request_id=? AND run_id=?
                AND status='running' AND owner_id=? AND revision=?""",
                (
                    status,
                    error,
                    receipt.thread_id,
                    receipt.request_id,
                    receipt.run_id,
                    owner_id,
                    receipt.revision,
                ),
            )
            if changed.rowcount != 1:
                raise ServiceConflictError(
                    "The admission is no longer owned by this worker"
                )
            return self.get(receipt.thread_id, receipt.request_id).receipt

    def prepare_resume(self, record: AdmissionRecord) -> AdmissionRecord:
        """CAS after the trusted host has established old-worker quiescence."""
        record = msgspec.convert(record, type=AdmissionRecord, strict=True)
        if (
            record.receipt.status not in {"running", "paused", "failed"}
            or not record.owner_id
        ):
            raise ServiceConflictError(
                "Only an owned interrupted attempt can be resumed"
            )
        with self._lock, self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            changed = self._connection.execute(
                """UPDATE service_admissions
                SET status='accepted', owner_id=NULL, error=NULL,
                revision=revision+1
                WHERE thread_id=? AND request_id=? AND run_id=? AND status=?
                AND owner_id IS ? AND revision=?""",
                (
                    record.receipt.thread_id,
                    record.receipt.request_id,
                    record.receipt.run_id,
                    record.receipt.status,
                    record.owner_id,
                    record.receipt.revision,
                ),
            )
            if changed.rowcount != 1:
                raise ServiceConflictError("Admission changed while preparing recovery")
            return self.get(record.receipt.thread_id, record.receipt.request_id)
