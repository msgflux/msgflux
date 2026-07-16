from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, List, Mapping

from msgflux.data.stores.registry import register_store
from msgflux.runtime.agent_inbox.base import AgentInboxStore

_CREATE_INBOX_TABLES = """\
CREATE TABLE IF NOT EXISTS agent_inboxes (
    namespace      TEXT NOT NULL,
    thread_id     TEXT NOT NULL,
    run_id         TEXT NOT NULL,
    notifications  TEXT NOT NULL,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL,
    PRIMARY KEY (namespace, thread_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_agent_inboxes_thread
    ON agent_inboxes(namespace, thread_id, updated_at DESC);
"""

_UPSERT_INBOX = """\
INSERT INTO agent_inboxes
    (namespace, thread_id, run_id, notifications, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(namespace, thread_id, run_id) DO UPDATE SET
    notifications = excluded.notifications,
    updated_at = excluded.updated_at
"""


@register_store()
class SQLiteAgentInboxStore(AgentInboxStore):
    """SQLite-backed inbox store."""

    provider = "sqlite"

    def __init__(self, path: str = ".msgflux/agent-inboxes.sqlite3") -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_CREATE_INBOX_TABLES)
        self._conn.commit()

    @staticmethod
    def _serialize(notifications: Iterable[Mapping[str, Any]]) -> str:
        return json.dumps(list(notifications), ensure_ascii=False, default=str)

    @staticmethod
    def _deserialize(text: str) -> List[Mapping[str, Any]]:
        data = json.loads(text)
        if not isinstance(data, list):
            return []
        return [item for item in data if isinstance(item, Mapping)]

    def load_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT notifications FROM agent_inboxes
                WHERE namespace=? AND thread_id=? AND run_id=?
                """,
                (namespace, thread_id, run_id),
            ).fetchone()
        if row is None:
            return []
        return self._deserialize(row[0])

    def save_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        notifications: Iterable[Mapping[str, Any]],
    ) -> None:
        with self._lock:
            now = time.time()
            created_at = self._conn.execute(
                """
                SELECT created_at FROM agent_inboxes
                WHERE namespace=? AND thread_id=? AND run_id=?
                """,
                (namespace, thread_id, run_id),
            ).fetchone()
            self._conn.execute(
                _UPSERT_INBOX,
                (
                    namespace,
                    thread_id,
                    run_id,
                    self._serialize(notifications),
                    created_at[0] if created_at else now,
                    now,
                ),
            )
            self._conn.commit()

    def clear(
        self,
        namespace: str | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
        *,
        older_than: float | None = None,
    ) -> int:
        query = "DELETE FROM agent_inboxes WHERE 1=1"
        params: List[Any] = []
        if namespace is not None:
            query += " AND namespace=?"
            params.append(namespace)
        if thread_id is not None:
            query += " AND thread_id=?"
            params.append(thread_id)
        if run_id is not None:
            query += " AND run_id=?"
            params.append(run_id)
        if older_than is not None:
            query += " AND updated_at < ?"
            params.append(time.time() - older_than)

        with self._lock:
            deleted = self._conn.execute(query, tuple(params)).rowcount
            self._conn.commit()
        return deleted or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
