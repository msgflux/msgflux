from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

from msgflux.data.stores.base import CheckpointStore
from msgflux.data.stores.registry import register_store

_UPSERT_STATE = """\
INSERT INTO checkpoints
    (namespace, session_id, run_id, status, state, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(namespace, session_id, run_id) DO UPDATE SET
    status = excluded.status,
    state = excluded.state,
    updated_at = excluded.updated_at
"""

_INSERT_EVENT = """\
INSERT INTO checkpoint_events
    (namespace, session_id, run_id, event_type, timestamp, data)
VALUES (?, ?, ?, ?, ?, ?)
"""

_SELECT_STATE = (
    "SELECT state FROM checkpoints WHERE namespace=? AND session_id=? AND run_id=?"
)

_DELETE_RUN = "DELETE FROM checkpoints WHERE namespace=? AND session_id=? AND run_id=?"

_CREATE_TABLES = """\
CREATE TABLE IF NOT EXISTS checkpoints (
    namespace   TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    state       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (namespace, session_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_session
    ON checkpoints(namespace, session_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_checkpoints_status
    ON checkpoints(namespace, session_id, status);

CREATE TABLE IF NOT EXISTS checkpoint_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace   TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    data        TEXT,
    FOREIGN KEY (namespace, session_id, run_id)
        REFERENCES checkpoints(namespace, session_id, run_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_run
    ON checkpoint_events(namespace, session_id, run_id);
"""


@register_store("checkpoint", "sqlite")
class SQLiteCheckpointStore(CheckpointStore):
    """SQLite-backed checkpoint store."""

    def __init__(self, path: str = ".msgflux/checkpoints.sqlite3") -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_CREATE_TABLES)
        self._conn.commit()

    @staticmethod
    def _serialize(obj: Mapping[str, Any]) -> str:
        return json.dumps(obj, ensure_ascii=False, default=str)

    @staticmethod
    def _deserialize(text: str) -> Dict[str, Any]:
        return json.loads(text)

    def save_state(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        state: Mapping[str, Any],
    ) -> None:
        now = time.time()
        payload = self._serialize(state)
        status = state.get("status", "running")
        self._conn.execute(
            _UPSERT_STATE,
            (namespace, session_id, run_id, status, payload, now, now),
        )
        self._conn.commit()

    def load_state(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> Mapping[str, Any] | None:
        row = self._conn.execute(
            _SELECT_STATE,
            (namespace, session_id, run_id),
        ).fetchone()
        if row is None:
            return None
        return self._deserialize(row[0])

    def append_event(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        event: Mapping[str, Any],
    ) -> None:
        now = time.time()
        event_type = event.get("event_type", "unknown")
        data = self._serialize(event)
        self._conn.execute(
            _INSERT_EVENT,
            (namespace, session_id, run_id, event_type, now, data),
        )
        self._conn.commit()

    def load_events(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT data FROM checkpoint_events
            WHERE namespace=? AND session_id=? AND run_id=?
            ORDER BY id ASC
            """,
            (namespace, session_id, run_id),
        ).fetchall()
        return [self._deserialize(r[0]) for r in rows if r[0]]

    def save_with_event(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
        state: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        now = time.time()
        payload = self._serialize(state)
        status = state.get("status", "running")
        event_type = event.get("event_type", "unknown")
        event_data = self._serialize(event)

        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN")
            cur.execute(
                _UPSERT_STATE,
                (namespace, session_id, run_id, status, payload, now, now),
            )
            cur.execute(
                _INSERT_EVENT,
                (namespace, session_id, run_id, event_type, now, event_data),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def list_runs(
        self,
        namespace: str,
        session_id: str,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> List[Mapping[str, Any]]:
        query = (
            "SELECT run_id, status, updated_at FROM checkpoints "
            "WHERE namespace=? AND session_id=?"
        )
        params: List[Any] = [namespace, session_id]
        if status is not None:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY updated_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(query, tuple(params)).fetchall()
        return [{"run_id": r[0], "status": r[1], "updated_at": r[2]} for r in rows]

    def delete_run(
        self,
        namespace: str,
        session_id: str,
        run_id: str,
    ) -> bool:
        deleted = self._conn.execute(
            _DELETE_RUN,
            (namespace, session_id, run_id),
        ).rowcount
        self._conn.commit()
        return bool(deleted)

    def clear(
        self,
        namespace: str | None = None,
        session_id: str | None = None,
        *,
        older_than: float | None = None,
    ) -> int:
        query = "DELETE FROM checkpoints WHERE 1=1"
        params: List[Any] = []
        if namespace is not None:
            query += " AND namespace=?"
            params.append(namespace)
        if session_id is not None:
            query += " AND session_id=?"
            params.append(session_id)
        if older_than is not None:
            cutoff = time.time() - older_than
            query += " AND updated_at < ?"
            params.append(cutoff)

        deleted = self._conn.execute(query, tuple(params)).rowcount
        self._conn.commit()
        return deleted or 0

    def close(self) -> None:
        self._conn.close()
