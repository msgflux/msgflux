"""Small SQLite admission journal; conversation and task stores remain separate."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from threading import RLock
from time import monotonic, sleep

import msgspec

from msgflux.runtime.context import new_run_id
from msgflux.runtime.permissions import ResourcePermission
from msgflux.runtime.service.records import (
    AdmissionReceipt,
    AdmissionRecord,
    AdmissionStatus,
    ServiceBusyError,
    ServiceConflictError,
    ServiceThread,
)
from msgflux.runtime.workspace.policy import WorkspacePolicy
from msgflux.utils.time import utc_now_isoformat


def validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{label}` must be a non-empty string")


def _enable_wal(connection: sqlite3.Connection) -> None:
    deadline = monotonic() + 30
    while True:
        try:
            result = connection.execute("PRAGMA journal_mode=WAL").fetchone()
            if result is None or str(result[0]).lower() != "wal":
                raise sqlite3.OperationalError("SQLite did not enable WAL journal mode")
            return
        except sqlite3.OperationalError as error:
            error_code = getattr(error, "sqlite_errorcode", None)
            primary_code = error_code & 0xFF if isinstance(error_code, int) else None
            if (
                primary_code not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                or monotonic() >= deadline
            ):
                raise
            sleep(0.01)


class SQLiteServiceStore:
    """Persist admissions at an explicit host path, or use ``:memory:``.

    This journal has no model history, credentials, tasks, or permission grants.
    A conditional claim fences journal writes, not arbitrary external effects.
    The host owns and closes the store, independently of AgentService.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._lock = RLock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        try:
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys=ON")
            # Serialize schema inspection and migration across independent processes.
            # SQLite DDL is transactional, so a second opener sees either the old
            # schema or the fully migrated one.
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(
                    """CREATE TABLE IF NOT EXISTS service_threads (
                    thread_id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, cwd TEXT
                    )"""
                )
                columns = {
                    row["name"]
                    for row in self._connection.execute(
                        "PRAGMA table_info(service_threads)"
                    )
                }
                if "cwd" not in columns:
                    self._connection.execute(
                        "ALTER TABLE service_threads ADD COLUMN cwd TEXT"
                    )
                self._connection.execute(
                    """CREATE TABLE IF NOT EXISTS service_admissions (
                    thread_id TEXT NOT NULL REFERENCES service_threads(thread_id),
                    request_id TEXT NOT NULL, run_id TEXT NOT NULL UNIQUE,
                    namespace TEXT NOT NULL, prompt TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL, owner_id TEXT, error TEXT,
                    revision INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (thread_id, request_id)
                    )"""
                )
                self._connection.execute(
                    """CREATE UNIQUE INDEX IF NOT EXISTS service_active_thread
                    ON service_admissions(thread_id)
                    WHERE status IN ('accepted', 'running', 'paused')"""
                )
                self._connection.execute(
                    """CREATE TABLE IF NOT EXISTS service_workspace_policies (
                    thread_id TEXT PRIMARY KEY
                        REFERENCES service_threads(thread_id) ON DELETE CASCADE,
                    permissions TEXT NOT NULL,
                    resources TEXT NOT NULL,
                    approval_policy TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision > 0),
                    updated_at TEXT NOT NULL
                    )"""
                )
                self._connection.execute(
                    """CREATE TABLE IF NOT EXISTS service_workspace_policy_history (
                    thread_id TEXT NOT NULL
                        REFERENCES service_threads(thread_id) ON DELETE CASCADE,
                    permissions TEXT NOT NULL,
                    resources TEXT NOT NULL,
                    approval_policy TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision > 0),
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (thread_id, revision)
                    )"""
                )
                self._connection.execute(
                    """CREATE TRIGGER IF NOT EXISTS
                    immutable_workspace_policy_history_update
                    BEFORE UPDATE ON service_workspace_policy_history
                    BEGIN
                        SELECT RAISE(ABORT, 'workspace policy history is append-only');
                    END"""
                )
                self._connection.execute(
                    """CREATE TRIGGER IF NOT EXISTS
                    immutable_workspace_policy_history_delete
                    BEFORE DELETE ON service_workspace_policy_history
                    BEGIN
                        SELECT RAISE(ABORT, 'workspace policy history is append-only');
                    END"""
                )
                self._connection.commit()
                if str(path) != ":memory:":
                    _enable_wal(self._connection)
        except BaseException:
            try:
                self._connection.rollback()
            except sqlite3.Error:
                pass
            finally:
                self._connection.close()
            raise

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
        if thread.cwd is not None:
            validate_identifier(thread.cwd, "cwd")
            if not Path(thread.cwd).is_absolute():
                raise ValueError("`cwd` must be an absolute path")
        with self._lock, self._connection:
            self._connection.execute(
                """INSERT OR IGNORE INTO service_threads
                (thread_id, agent_id, cwd) VALUES (?, ?, ?)""",
                (thread.thread_id, thread.agent_id, thread.cwd),
            )
            existing = self.thread(thread.thread_id)
            if existing != thread:
                raise ServiceConflictError(
                    "The thread binding conflicts with stored state"
                )
        return thread

    def thread(self, thread_id: str) -> ServiceThread:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM service_threads WHERE thread_id=?", (thread_id,)
            ).fetchone()
        if row is None:
            raise KeyError(thread_id)
        return ServiceThread(row["thread_id"], row["agent_id"], row["cwd"])

    def threads(self) -> tuple[ServiceThread, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM service_threads ORDER BY thread_id"
            ).fetchall()
        return tuple(
            ServiceThread(row["thread_id"], row["agent_id"], row["cwd"]) for row in rows
        )

    @staticmethod
    def _policy_values(policy: WorkspacePolicy) -> tuple[str, str, str]:
        policy.permission_set()
        permissions = msgspec.json.encode(policy.permissions).decode("utf-8")
        resources = msgspec.json.encode(
            [
                {"resource": item.resource, "action": item.action}
                for item in policy.resources
            ]
        ).decode("utf-8")
        return permissions, resources, policy.approval_policy

    @staticmethod
    def _workspace_policy(row: sqlite3.Row) -> WorkspacePolicy:
        permissions = msgspec.json.decode(row["permissions"], type=tuple[str, ...])
        resource_values = msgspec.json.decode(
            row["resources"], type=list[dict[str, str]]
        )
        resources = tuple(ResourcePermission(**item) for item in resource_values)
        return WorkspacePolicy(
            thread_id=row["thread_id"],
            permissions=permissions,
            resources=resources,
            approval_policy=row["approval_policy"],
            revision=row["revision"],
            updated_at=row["updated_at"],
        )

    def workspace_policy(self, thread_id: str) -> WorkspacePolicy | None:
        """Read the current policy for a known thread without creating a row."""
        validate_identifier(thread_id, "thread_id")
        self.thread(thread_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM service_workspace_policies WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
        return self._workspace_policy(row) if row is not None else None

    def update_workspace_policy(
        self,
        policy: WorkspacePolicy,
        *,
        expected_revision: int | None = None,
    ) -> WorkspacePolicy:
        """Persist a policy revision and its audit row using one SQLite CAS."""
        if not isinstance(policy, WorkspacePolicy):
            raise TypeError("policy must be a WorkspacePolicy")
        validate_identifier(policy.thread_id, "thread_id")
        if expected_revision is not None and (
            type(expected_revision) is not int or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer or None")
        permissions, resources, approval_policy = self._policy_values(policy)
        with self._lock, self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            self.thread(policy.thread_id)
            current = self._connection.execute(
                "SELECT revision FROM service_workspace_policies WHERE thread_id=?",
                (policy.thread_id,),
            ).fetchone()
            current_revision = current["revision"] if current is not None else 0
            if expected_revision is not None and expected_revision != current_revision:
                raise ServiceConflictError("Workspace policy changed")
            revision = current_revision + 1
            updated_at = utc_now_isoformat()
            values = (
                permissions,
                resources,
                approval_policy,
                revision,
                updated_at,
                policy.thread_id,
            )
            self._connection.execute(
                """INSERT INTO service_workspace_policies
                (permissions, resources, approval_policy, revision, updated_at,
                 thread_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    permissions=excluded.permissions,
                    resources=excluded.resources,
                    approval_policy=excluded.approval_policy,
                    revision=excluded.revision,
                    updated_at=excluded.updated_at""",
                values,
            )
            self._connection.execute(
                """INSERT INTO service_workspace_policy_history
                (thread_id, permissions, resources, approval_policy, revision,
                 updated_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    policy.thread_id,
                    permissions,
                    resources,
                    approval_policy,
                    revision,
                    updated_at,
                ),
            )
            return self._workspace_policy(
                self._connection.execute(
                    "SELECT * FROM service_workspace_policies WHERE thread_id=?",
                    (policy.thread_id,),
                ).fetchone()
            )

    def workspace_policy_history(self, thread_id: str) -> tuple[WorkspacePolicy, ...]:
        """Return every recorded policy revision in revision order."""
        validate_identifier(thread_id, "thread_id")
        self.thread(thread_id)
        with self._lock:
            rows = self._connection.execute(
                """SELECT * FROM service_workspace_policy_history
                WHERE thread_id=? ORDER BY revision""",
                (thread_id,),
            ).fetchall()
        return tuple(self._workspace_policy(row) for row in rows)

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

    def get_for_run(self, thread_id: str, run_id: str) -> AdmissionRecord | None:
        """Find the journal entry for a checkpoint run identity."""
        validate_identifier(thread_id, "thread_id")
        validate_identifier(run_id, "run_id")
        self.thread(thread_id)
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM service_admissions WHERE thread_id=? AND run_id=?",
                (thread_id, run_id),
            ).fetchone()
        return self._record(row) if row is not None else None

    def adopt_checkpoint(
        self, thread_id: str, run_id: str, namespace: str
    ) -> AdmissionRecord:
        """Journal an existing checkpoint identity without scheduling work.

        The reserved owner marks the row as belonging to an older, untracked
        worker. A trusted service must establish quiescence before preparing it
        for resume.
        """
        validate_identifier(thread_id, "thread_id")
        validate_identifier(run_id, "run_id")
        validate_identifier(namespace, "namespace")
        request_id = f"checkpoint:{run_id}"
        fingerprint = hashlib.sha256(msgspec.json.encode([namespace, ""])).hexdigest()
        with self._lock, self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            self.thread(thread_id)
            existing_run = self._connection.execute(
                "SELECT * FROM service_admissions WHERE thread_id=? AND run_id=?",
                (thread_id, run_id),
            ).fetchone()
            if existing_run is not None:
                existing = self._record(existing_run)
                if existing.namespace != namespace:
                    raise ServiceConflictError(
                        "The checkpoint run belongs to another namespace"
                    )
                return existing

            existing_request = self._connection.execute(
                "SELECT * FROM service_admissions WHERE thread_id=? AND request_id=?",
                (thread_id, request_id),
            ).fetchone()
            if existing_request is not None:
                raise ServiceConflictError(
                    "The reserved checkpoint request_id identifies another run"
                )

            active = self._connection.execute(
                """SELECT 1 FROM service_admissions WHERE thread_id=?
                AND status IN ('accepted', 'running', 'paused') LIMIT 1""",
                (thread_id,),
            ).fetchone()
            if active is not None:
                raise ServiceBusyError("The thread has unfinished admitted work")

            try:
                self._connection.execute(
                    """INSERT INTO service_admissions
                    (thread_id, request_id, run_id, namespace, prompt,
                     fingerprint, status, owner_id, revision)
                    VALUES (?, ?, ?, ?, '', ?, 'running', 'checkpoint', 1)""",
                    (thread_id, request_id, run_id, namespace, fingerprint),
                )
            except sqlite3.IntegrityError as error:
                active = self._connection.execute(
                    """SELECT 1 FROM service_admissions WHERE thread_id=?
                    AND status IN ('accepted', 'running', 'paused') LIMIT 1""",
                    (thread_id,),
                ).fetchone()
                if active is not None:
                    raise ServiceBusyError(
                        "The thread has unfinished admitted work"
                    ) from error
                raise ServiceConflictError(
                    "The checkpoint identity conflicts with an existing admission"
                ) from error
            row = self._connection.execute(
                "SELECT * FROM service_admissions WHERE thread_id=? AND run_id=?",
                (thread_id, run_id),
            ).fetchone()
            return self._record(row)

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
