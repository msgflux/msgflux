from __future__ import annotations

# SQL statements are parameterized; dynamic text only expands placeholder count.
# ruff: noqa: E501, S608
import json
import sqlite3
import time
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, List, Mapping
from uuid import uuid4

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

CREATE TABLE IF NOT EXISTS agent_inbox_notifications (
    namespace TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    notification_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    dedupe_key TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    lease_id TEXT,
    lease_until REAL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (namespace, thread_id, run_id, notification_id)
);
CREATE INDEX IF NOT EXISTS idx_agent_inbox_notifications_claim
    ON agent_inbox_notifications(namespace, thread_id, run_id, status, lease_until);
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
            rows = self._conn.execute(
                """SELECT payload FROM agent_inbox_notifications
                WHERE namespace=? AND thread_id=? AND run_id=?
                ORDER BY rowid""",
                (namespace, thread_id, run_id),
            ).fetchall()
            if rows:
                return [self._deserialize("[" + row[0] + "]")[0] for row in rows]
            row = self._conn.execute(
                "SELECT notifications FROM agent_inboxes WHERE namespace=? AND thread_id=? AND run_id=?",
                (namespace, thread_id, run_id),
            ).fetchone()
        return [] if row is None else self._deserialize(row[0])

    def save_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        notifications: Iterable[Mapping[str, Any]],
    ) -> None:
        payloads = [dict(item) for item in notifications]
        with self._lock:
            now = time.time()
            created_at = self._conn.execute(
                "SELECT created_at FROM agent_inboxes WHERE namespace=? AND thread_id=? AND run_id=?",
                (namespace, thread_id, run_id),
            ).fetchone()
            self._conn.execute(
                _UPSERT_INBOX,
                (
                    namespace,
                    thread_id,
                    run_id,
                    self._serialize(payloads),
                    created_at[0] if created_at else now,
                    now,
                ),
            )
            self._conn.execute(
                "DELETE FROM agent_inbox_notifications WHERE namespace=? AND thread_id=? AND run_id=?",
                (namespace, thread_id, run_id),
            )
            self._conn.executemany(
                """INSERT INTO agent_inbox_notifications
                (namespace, thread_id, run_id, notification_id, payload, dedupe_key,
                 status, lease_id, lease_until, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, NULL, ?, ?)""",
                [
                    (
                        namespace,
                        thread_id,
                        run_id,
                        item.get("notification_id") or uuid4().hex,
                        self._serialize([item])[1:-1],
                        item.get("dedupe_key"),
                        now,
                        now,
                    )
                    for item in payloads
                ],
            )
            self._conn.commit()

    def publish_notification(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        notification: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        with self._lock:
            now = time.time()
            self._conn.execute("BEGIN IMMEDIATE")
            dedupe_key = notification.get("dedupe_key")
            row = None
            if dedupe_key:
                row = self._conn.execute(
                    "SELECT notification_id FROM agent_inbox_notifications WHERE namespace=? AND thread_id=? AND run_id=? AND dedupe_key=?",
                    (namespace, thread_id, run_id, dedupe_key),
                ).fetchone()
            notification_id = notification.get("notification_id") or uuid4().hex
            payload = self._serialize([dict(notification)])[1:-1]
            if row is not None:
                self._conn.execute(
                    "UPDATE agent_inbox_notifications SET notification_id=?, payload=?, updated_at=? WHERE namespace=? AND thread_id=? AND run_id=? AND dedupe_key=?",
                    (
                        notification_id,
                        payload,
                        now,
                        namespace,
                        thread_id,
                        run_id,
                        dedupe_key,
                    ),
                )
            else:
                self._conn.execute(
                    """INSERT INTO agent_inbox_notifications
                    (namespace,thread_id,run_id,notification_id,payload,dedupe_key,status,lease_id,lease_until,created_at,updated_at)
                    VALUES (?,?,?,?,?,?, 'pending',NULL,NULL,?,?)""",
                    (
                        namespace,
                        thread_id,
                        run_id,
                        notification_id,
                        payload,
                        dedupe_key,
                        now,
                        now,
                    ),
                )
            self._sync_legacy_row(namespace, thread_id, run_id)
            self._conn.commit()
        return dict(notification, notification_id=notification_id)

    def claim_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        *,
        lease_id: str,
        lease_seconds: float,
        limit: int | None = None,
    ) -> List[Mapping[str, Any]]:
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            self._migrate_legacy_locked(namespace, thread_id, run_id)
            query = """SELECT notification_id, payload FROM agent_inbox_notifications
                WHERE namespace=? AND thread_id=? AND run_id=?
                AND (status='pending' OR lease_until IS NULL OR lease_until<=?)
                ORDER BY rowid"""
            params = [namespace, thread_id, run_id, now]
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            rows = self._conn.execute(query, params).fetchall()
            ids = [row[0] for row in rows]
            if ids:
                self._conn.executemany(
                    """UPDATE agent_inbox_notifications SET status='leased', lease_id=?,
                    lease_until=?, updated_at=? WHERE namespace=? AND thread_id=? AND
                    run_id=? AND notification_id=?""",
                    [
                        (
                            lease_id,
                            now + max(0.001, lease_seconds),
                            now,
                            namespace,
                            thread_id,
                            run_id,
                            item_id,
                        )
                        for item_id in ids
                    ],
                )
            self._conn.commit()
        return [self._deserialize("[" + row[1] + "]")[0] for row in rows]

    def ack_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        notification_ids: Iterable[str],
        *,
        lease_id: str | None = None,
    ) -> None:
        ids = list(set(notification_ids))
        if not ids:
            return
        with self._lock:
            clauses = " AND lease_id=?" if lease_id is not None else ""
            params = [namespace, thread_id, run_id, *ids]
            if lease_id is not None:
                params.append(lease_id)
            self._conn.execute(
                f"DELETE FROM agent_inbox_notifications WHERE namespace=? AND thread_id=? AND run_id=? AND notification_id IN ({','.join('?' for _ in ids)}){clauses}",
                params,
            )
            self._sync_legacy_row(namespace, thread_id, run_id)
            self._conn.commit()

    def release_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        *,
        lease_id: str,
        notification_ids: Iterable[str] | None = None,
    ) -> None:
        ids = set(notification_ids) if notification_ids is not None else None
        if ids is not None and not ids:
            return
        with self._lock:
            query = """UPDATE agent_inbox_notifications SET status='pending', lease_id=NULL,
                lease_until=NULL, updated_at=? WHERE namespace=? AND thread_id=? AND
                run_id=? AND lease_id=?"""
            params: list[object] = [time.time(), namespace, thread_id, run_id, lease_id]
            if ids is not None:
                query += f" AND notification_id IN ({','.join('?' for _ in ids)})"
                params.extend(ids)
            self._conn.execute(query, params)
            self._conn.commit()

    def move_notifications(
        self,
        previous: tuple[str, str, str],
        current: tuple[str, str, str],
    ) -> None:
        if previous == current:
            return
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            rows = self._conn.execute(
                "SELECT notification_id, payload, dedupe_key, created_at FROM agent_inbox_notifications WHERE namespace=? AND thread_id=? AND run_id=?",
                previous,
            ).fetchall()
            for item_id, payload, dedupe_key, created_at in rows:
                self._conn.execute(
                    """INSERT OR IGNORE INTO agent_inbox_notifications
                    (namespace,thread_id,run_id,notification_id,payload,dedupe_key,status,lease_id,lease_until,created_at,updated_at)
                    VALUES (?,?,?,?,?,?, 'pending',NULL,NULL,?,?)""",
                    (*current, item_id, payload, dedupe_key, created_at, time.time()),
                )
            self._conn.execute(
                "DELETE FROM agent_inbox_notifications WHERE namespace=? AND thread_id=? AND run_id=?",
                previous,
            )
            self._sync_legacy_row(*previous)
            self._sync_legacy_row(*current)
            self._conn.commit()

    def _migrate_legacy_locked(
        self, namespace: str, thread_id: str, run_id: str
    ) -> None:
        exists = self._conn.execute(
            "SELECT 1 FROM agent_inbox_notifications WHERE namespace=? AND thread_id=? AND run_id=? LIMIT 1",
            (namespace, thread_id, run_id),
        ).fetchone()
        if exists is not None:
            return
        row = self._conn.execute(
            "SELECT notifications FROM agent_inboxes WHERE namespace=? AND thread_id=? AND run_id=?",
            (namespace, thread_id, run_id),
        ).fetchone()
        if row is None:
            return
        now = time.time()
        payloads = self._deserialize(row[0])
        self._conn.executemany(
            """INSERT OR IGNORE INTO agent_inbox_notifications
            (namespace,thread_id,run_id,notification_id,payload,dedupe_key,status,lease_id,lease_until,created_at,updated_at)
            VALUES (?,?,?,?,?,?, 'pending',NULL,NULL,?,?)""",
            [
                (
                    namespace,
                    thread_id,
                    run_id,
                    item.get("notification_id") or uuid4().hex,
                    self._serialize([item])[1:-1],
                    item.get("dedupe_key"),
                    now,
                    now,
                )
                for item in payloads
            ],
        )

    def _sync_legacy_row(self, namespace: str, thread_id: str, run_id: str) -> None:
        rows = self._conn.execute(
            "SELECT payload FROM agent_inbox_notifications WHERE namespace=? AND thread_id=? AND run_id=? ORDER BY rowid",
            (namespace, thread_id, run_id),
        ).fetchall()
        payloads = [self._deserialize("[" + row[0] + "]")[0] for row in rows]
        now = time.time()
        existing = self._conn.execute(
            "SELECT created_at FROM agent_inboxes WHERE namespace=? AND thread_id=? AND run_id=?",
            (namespace, thread_id, run_id),
        ).fetchone()
        self._conn.execute(
            _UPSERT_INBOX,
            (
                namespace,
                thread_id,
                run_id,
                self._serialize(payloads),
                existing[0] if existing else now,
                now,
            ),
        )

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
            child_query = "DELETE FROM agent_inbox_notifications WHERE 1=1"
            child_params: List[Any] = []
            if namespace is not None:
                child_query += " AND namespace=?"
                child_params.append(namespace)
            if thread_id is not None:
                child_query += " AND thread_id=?"
                child_params.append(thread_id)
            if run_id is not None:
                child_query += " AND run_id=?"
                child_params.append(run_id)
            if older_than is not None:
                child_query += " AND updated_at < ?"
                child_params.append(time.time() - older_than)
            self._conn.execute(child_query, tuple(child_params))
            deleted = self._conn.execute(query, tuple(params)).rowcount
            self._conn.commit()
        return deleted or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
