from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Callable

from msgflux.data.stores.registry import register_store
from msgflux.runtime.approvals.base import ApprovalStore, UpdateApproval
from msgflux.runtime.approvals.records import (
    ApprovalBinding,
    ApprovalEvent,
    ApprovalRecord,
    require_name,
)

_CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS runtime_approvals (
    namespace TEXT NOT NULL,
    request_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (namespace, request_id)
);
CREATE INDEX IF NOT EXISTS idx_runtime_approvals_run
    ON runtime_approvals(namespace, thread_id, run_id);
CREATE TABLE IF NOT EXISTS runtime_approval_events (
    namespace TEXT NOT NULL,
    request_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (namespace, request_id, revision),
    FOREIGN KEY (namespace, request_id)
        REFERENCES runtime_approvals(namespace, request_id)
);
"""


@register_store()
class SQLiteApprovalStore(ApprovalStore):
    """Transactional approval journal, including across independent connections."""

    provider = "sqlite"

    def __init__(
        self,
        path: str = ".msgflux/approvals.sqlite3",
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(clock=clock)
        self.path = path
        self._lock = RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_CREATE_TABLES)
        self._conn.commit()

    @staticmethod
    def _decode(payload: str) -> ApprovalRecord:
        values = json.loads(payload)
        values["binding"] = ApprovalBinding(**values["binding"])
        return ApprovalRecord(**values)

    def _update(
        self, namespace: str, request_id: str, update: UpdateApproval
    ) -> ApprovalRecord | None:
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT payload FROM runtime_approvals "
                "WHERE namespace=? AND request_id=?",
                (namespace, request_id),
            ).fetchone()
            previous = self._decode(row[0]) if row is not None else None
            record = update(previous)
            if record is not None and record != previous:
                self._conn.execute(
                    """INSERT INTO runtime_approvals
                    (namespace, request_id, thread_id, run_id, payload)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, request_id) DO UPDATE SET
                        payload=excluded.payload""",
                    (
                        namespace,
                        request_id,
                        record.binding.thread_id,
                        record.binding.run_id,
                        json.dumps(asdict(record), allow_nan=False),
                    ),
                )
                self._conn.execute(
                    """INSERT INTO runtime_approval_events
                    (namespace, request_id, revision, payload) VALUES (?, ?, ?, ?)""",
                    (
                        namespace,
                        request_id,
                        record.revision,
                        json.dumps(
                            asdict(ApprovalEvent.from_record(record)), allow_nan=False
                        ),
                    ),
                )
            return record

    def _records(
        self, namespace: str, thread_id: str, run_id: str
    ) -> list[ApprovalRecord]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT payload FROM runtime_approvals
                WHERE namespace=? AND thread_id=? AND run_id=? ORDER BY rowid""",
                (namespace, thread_id, run_id),
            ).fetchall()
        return [self._decode(row[0]) for row in rows]

    def events(self, namespace: str, request_id: str) -> list[ApprovalEvent]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT payload FROM runtime_approval_events
                WHERE namespace=? AND request_id=? ORDER BY revision""",
                (require_name(namespace), require_name(request_id)),
            ).fetchall()
        return [ApprovalEvent(**json.loads(row[0])) for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
