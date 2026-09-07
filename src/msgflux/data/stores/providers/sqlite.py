# Checkpoints are split into small run records and immutable message items.
# A run belongs to `(namespace, thread_id, run_id)` and stores the non-message
# state plus the ChatMessages container metadata and an ordered list of item
# references. The message items themselves are stored once per thread by a
# content hash, so continuing a conversation with a new run_id only writes a new
# run record that points at the existing items. Loading reverses that shape and
# returns the public state with `messages.items` restored. Forking to another
# thread copies only the referenced items needed by the fork; deleting a run
# removes message items when no remaining run in that thread references them.
#
# In SQLite this is represented by three tables. `checkpoints` is keyed by
# `(namespace, thread_id, run_id)` and stores the normalized JSON state plus the
# run status/timestamps. `checkpoint_message_items` is keyed by
# `(namespace, thread_id, item_ref)` and stores one frozen ChatMessages item JSON
# payload per content hash. `checkpoint_events` keeps append-only events for a
# run and is removed by SQLite's FK cascade when the run is deleted. The
# normalized state stored in `checkpoints.state` replaces `messages.items` with
# `_messages = {"state": <messages without items>, "item_entries": [...]}`.
# `item_entries` is an ordered list, not a join table, so rehydration can rebuild
# the exact public message order and occurrence ids while item payloads remain
# deduplicated.

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from copy import deepcopy
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Literal, Mapping

from msgflux._private.chat_items import (
    restore_item_occurrence,
    split_item_occurrence,
)
from msgflux.data.stores.base import (
    CheckpointCommit,
    CheckpointConflictError,
    CheckpointStore,
)
from msgflux.data.stores.registry import register_store
from msgflux.data.stores.types import CheckpointStoreType

_UPSERT_STATE = """\
INSERT INTO checkpoints
    (namespace, thread_id, run_id, status, state, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(namespace, thread_id, run_id) DO UPDATE SET
    status = excluded.status,
    state = excluded.state,
    updated_at = excluded.updated_at
"""

_UPSERT_MESSAGE_ITEM = """\
INSERT INTO checkpoint_message_items
    (namespace, thread_id, item_ref, item, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(namespace, thread_id, item_ref) DO NOTHING
"""

_INSERT_EVENT = """\
INSERT INTO checkpoint_events
    (namespace, thread_id, run_id, event_type, timestamp, data)
VALUES (?, ?, ?, ?, ?, ?)
"""

_SELECT_STATE = """\
SELECT state FROM checkpoints WHERE namespace=? AND thread_id=? AND run_id=?
"""

_DELETE_RUN = "DELETE FROM checkpoints WHERE namespace=? AND thread_id=? AND run_id=?"

_CREATE_TABLES = """\
CREATE TABLE IF NOT EXISTS checkpoints (
    namespace   TEXT NOT NULL,
    thread_id  TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'running',
    state       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    PRIMARY KEY (namespace, thread_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
    ON checkpoints(namespace, thread_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_checkpoints_status
    ON checkpoints(namespace, thread_id, status);

CREATE TABLE IF NOT EXISTS checkpoint_message_items (
    namespace  TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    item_ref  TEXT NOT NULL,
    item      TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (namespace, thread_id, item_ref)
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_message_items_thread
    ON checkpoint_message_items(namespace, thread_id, item_ref);

CREATE TABLE IF NOT EXISTS checkpoint_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace   TEXT NOT NULL,
    thread_id  TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    data        TEXT,
    FOREIGN KEY (namespace, thread_id, run_id)
        REFERENCES checkpoints(namespace, thread_id, run_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_events_run
    ON checkpoint_events(namespace, thread_id, run_id);
"""


def _locked(method):
    """Serialize operations that share the store's SQLite connection."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


@register_store()
class SQLiteCheckpointStore(CheckpointStore, CheckpointStoreType):
    """SQLite-backed checkpoint store."""

    provider = "sqlite"
    supports_atomic_commit = True

    def __init__(self, path: str = ".msgflux/checkpoints.sqlite3") -> None:
        self.path = path
        self._lock = RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
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

    def _normalize_state(
        self,
        namespace: str,
        thread_id: str,
        state: Mapping[str, Any],
        now: float,
        executor: Any | None = None,
    ) -> Dict[str, Any]:
        normalized = dict(state)
        messages = normalized.pop("messages", None)
        if not isinstance(messages, Mapping):
            return normalized

        items = messages.get("items")
        if not isinstance(items, list):
            return normalized

        target = executor or self._conn
        item_entries = []
        item_ids: set[str] = set()
        for index, item in enumerate(items):
            payload, entry = split_item_occurrence(
                namespace=namespace,
                thread_id=thread_id,
                index=index,
                item=item,
            )
            item_id = entry["item_id"]
            if item_id in item_ids:
                raise ValueError(f"Duplicate ChatMessages item_id `{item_id}`.")
            item_ids.add(item_id)
            item_ref = self._item_ref(payload)
            entry["item_ref"] = item_ref
            item_entries.append(entry)
            target.execute(
                _UPSERT_MESSAGE_ITEM,
                (
                    namespace,
                    thread_id,
                    item_ref,
                    self._serialize(payload),
                    now,
                    now,
                ),
            )

        message_state = dict(messages)
        message_state.pop("items", None)
        normalized["_messages"] = {
            "state": message_state,
            "item_entries": item_entries,
        }
        return normalized

    def _denormalize_state(
        self,
        namespace: str,
        thread_id: str,
        state_text: str,
    ) -> Dict[str, Any]:
        state = self._deserialize(state_text)

        normalized_messages = state.pop("_messages", None)
        if not isinstance(normalized_messages, Mapping):
            return state

        message_state = normalized_messages.get("state")
        item_entries = normalized_messages.get("item_entries")
        if not isinstance(message_state, Mapping) or not isinstance(item_entries, list):
            return state

        messages = dict(message_state)
        messages["items"] = []
        for entry in item_entries:
            if not isinstance(entry, Mapping):
                raise ValueError("Checkpoint message entry is corrupted")
            item_ref = entry.get("item_ref")
            if not isinstance(item_ref, str):
                raise ValueError(
                    "Checkpoint message item is missing or corrupted: "
                    f"{namespace}/{thread_id}/{item_ref!r}"
                )
            row = self._conn.execute(
                "SELECT item FROM checkpoint_message_items "
                "WHERE namespace=? AND thread_id=? AND item_ref=?",
                (namespace, thread_id, item_ref),
            ).fetchone()
            if row is None:
                raise ValueError(
                    "Checkpoint message item is missing or corrupted: "
                    f"{namespace}/{thread_id}/{item_ref!r}"
                )
            item = restore_item_occurrence(self._deserialize(row[0]), entry)
            messages["items"].append(item)
        state["messages"] = messages
        self._validate_checkpoint_envelope(state.get("_checkpoint"))
        return state

    @staticmethod
    def _item_ref(item: Any) -> str:
        payload = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        return "item_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _list_live_item_refs(self, namespace: str, thread_id: str) -> set[str]:
        rows = self._conn.execute(
            "SELECT state FROM checkpoints WHERE namespace=? AND thread_id=?",
            (namespace, thread_id),
        ).fetchall()
        live: set[str] = set()
        for row in rows:
            state = self._deserialize(row[0])
            live.update(self._collect_refs_from_state(state))
        return live

    @staticmethod
    def _collect_refs_from_state(state: Any) -> set[str]:
        if not isinstance(state, Mapping):
            return set()
        messages = state.get("_messages")
        if not isinstance(messages, Mapping):
            return set()
        item_entries = messages.get("item_entries")
        if not isinstance(item_entries, list):
            return set()
        return {
            entry["item_ref"]
            for entry in item_entries
            if isinstance(entry, Mapping) and isinstance(entry.get("item_ref"), str)
        }

    def _cleanup_orphaned_items(self, namespace: str, thread_id: str) -> None:
        live = self._list_live_item_refs(namespace, thread_id)
        if live:
            self._conn.execute(
                "CREATE TEMP TABLE IF NOT EXISTS live_checkpoint_item_refs "
                "(item_ref TEXT PRIMARY KEY)"
            )
            self._conn.execute("DELETE FROM live_checkpoint_item_refs")
            self._conn.executemany(
                "INSERT INTO live_checkpoint_item_refs(item_ref) VALUES (?)",
                [(item_ref,) for item_ref in live],
            )
            self._conn.execute(
                "DELETE FROM checkpoint_message_items "
                "WHERE namespace=? AND thread_id=? "
                "AND item_ref NOT IN (SELECT item_ref FROM live_checkpoint_item_refs)",
                (namespace, thread_id),
            )
            self._conn.execute("DELETE FROM live_checkpoint_item_refs")
            return
        self._conn.execute(
            "DELETE FROM checkpoint_message_items WHERE namespace=? AND thread_id=?",
            (namespace, thread_id),
        )

    @staticmethod
    def _clear_queries(
        *,
        namespace: str | None,
        thread_id: str | None,
        older_than: float | None,
    ) -> tuple[str, str, List[Any]]:
        params: List[Any] = []
        if namespace is not None and thread_id is not None and older_than is not None:
            params.extend([namespace, thread_id, time.time() - older_than])
            return (
                "SELECT namespace, thread_id FROM checkpoints "
                "WHERE namespace=? AND thread_id=? AND updated_at < ?",
                "DELETE FROM checkpoints "
                "WHERE namespace=? AND thread_id=? AND updated_at < ?",
                params,
            )
        if namespace is not None and thread_id is not None:
            params.extend([namespace, thread_id])
            return (
                "SELECT namespace, thread_id FROM checkpoints "
                "WHERE namespace=? AND thread_id=?",
                "DELETE FROM checkpoints WHERE namespace=? AND thread_id=?",
                params,
            )
        if namespace is not None and older_than is not None:
            params.extend([namespace, time.time() - older_than])
            return (
                "SELECT namespace, thread_id FROM checkpoints "
                "WHERE namespace=? AND updated_at < ?",
                "DELETE FROM checkpoints WHERE namespace=? AND updated_at < ?",
                params,
            )
        if namespace is not None:
            params.append(namespace)
            return (
                "SELECT namespace, thread_id FROM checkpoints WHERE namespace=?",
                "DELETE FROM checkpoints WHERE namespace=?",
                params,
            )
        if thread_id is not None and older_than is not None:
            params.extend([thread_id, time.time() - older_than])
            return (
                "SELECT namespace, thread_id FROM checkpoints "
                "WHERE thread_id=? AND updated_at < ?",
                "DELETE FROM checkpoints WHERE thread_id=? AND updated_at < ?",
                params,
            )
        if thread_id is not None:
            params.append(thread_id)
            return (
                "SELECT namespace, thread_id FROM checkpoints WHERE thread_id=?",
                "DELETE FROM checkpoints WHERE thread_id=?",
                params,
            )
        if older_than is not None:
            params.append(time.time() - older_than)
            return (
                "SELECT namespace, thread_id FROM checkpoints WHERE updated_at < ?",
                "DELETE FROM checkpoints WHERE updated_at < ?",
                params,
            )
        return (
            "SELECT namespace, thread_id FROM checkpoints",
            "DELETE FROM checkpoints",
            params,
        )

    @_locked
    def save_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
    ) -> None:
        now = time.time()
        self._validate_checkpoint_envelope(state.get("_checkpoint"))
        normalized = self._normalize_state(namespace, thread_id, state, now)
        payload = self._serialize(normalized)
        status = state.get("status", "running")
        self._conn.execute(
            _UPSERT_STATE,
            (namespace, thread_id, run_id, status, payload, now, now),
        )
        self._conn.commit()

    @_locked
    def load_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> Mapping[str, Any] | None:
        row = self._conn.execute(
            _SELECT_STATE,
            (namespace, thread_id, run_id),
        ).fetchone()
        if row is None:
            return None
        return self._denormalize_state(namespace, thread_id, row[0])

    @_locked
    def append_event(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        event: Mapping[str, Any],
    ) -> None:
        now = time.time()
        event_type = event.get("event_type", "unknown")
        data = self._serialize(event)
        self._conn.execute(
            _INSERT_EVENT,
            (namespace, thread_id, run_id, event_type, now, data),
        )
        self._conn.commit()

    @_locked
    def load_events(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT data FROM checkpoint_events
            WHERE namespace=? AND thread_id=? AND run_id=?
            ORDER BY id ASC
            """,
            (namespace, thread_id, run_id),
        ).fetchall()
        return [self._deserialize(r[0]) for r in rows if r[0]]

    @_locked
    def commit_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
        event: Mapping[str, Any] | None = None,
        branch_id: str | None = None,
        head_item_id: str | None = None,
        extension_state: Mapping[str, Any] | None = None,
    ) -> CheckpointCommit:
        now = time.time()
        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            row = cur.execute(_SELECT_STATE, (namespace, thread_id, run_id)).fetchone()
            current = (
                self._denormalize_state(namespace, thread_id, row[0]) if row else {}
            )
            checkpoint = current.get("_checkpoint", {})
            self._validate_checkpoint_envelope(checkpoint)
            revision = checkpoint.get("revision", 0)
            if expected_revision is not None and revision != expected_revision:
                self._conn.rollback()
                raise CheckpointConflictError(
                    f"Checkpoint revision conflict: expected {expected_revision}, "
                    f"found {revision}."
                )
            next_revision = revision + 1
            committed = deepcopy(dict(state))
            committed["_checkpoint"] = {
                "schema_version": 1,
                "revision": next_revision,
                "branch_id": branch_id
                if branch_id is not None
                else checkpoint.get("branch_id"),
                "head_item_id": head_item_id
                if head_item_id is not None
                else checkpoint.get("head_item_id"),
                "extensions": deepcopy(
                    dict(
                        checkpoint.get("extensions", {})
                        if extension_state is None
                        else extension_state
                    )
                ),
            }
            normalized = self._normalize_state(
                namespace, thread_id, committed, now, executor=cur
            )
            cur.execute(
                _UPSERT_STATE,
                (
                    namespace,
                    thread_id,
                    run_id,
                    committed.get("status", "running"),
                    self._serialize(normalized),
                    now,
                    now,
                ),
            )
            if event is not None:
                cur.execute(
                    _INSERT_EVENT,
                    (
                        namespace,
                        thread_id,
                        run_id,
                        event.get("event_type", "unknown"),
                        now,
                        self._serialize(event),
                    ),
                )
            self._conn.commit()
            return CheckpointCommit(
                next_revision,
                committed,
                committed["_checkpoint"]["branch_id"],
                committed["_checkpoint"]["head_item_id"],
            )
        except Exception:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise

    @_locked
    def fork_run(
        self,
        namespace: str,
        source_thread_id: str,
        source_run_id: str,
        *,
        target_thread_id: str,
        target_run_id: str,
        status: str | None = None,
        at_item_id: str | None = None,
        position: Literal["before", "at"] = "at",
    ) -> Mapping[str, Any]:
        row = self._conn.execute(
            _SELECT_STATE,
            (namespace, source_thread_id, source_run_id),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Checkpoint run `{source_run_id}` not found in thread "
                f"`{source_thread_id}`."
            )

        source_state = self._denormalize_state(
            namespace,
            source_thread_id,
            row[0],
        )
        state = self._prepare_fork_state(
            source_state,
            at_item_id=at_item_id,
            position=position,
        )
        if status is not None:
            state["status"] = status
        checkpoint = state.get("_checkpoint")
        self._validate_checkpoint_envelope(checkpoint)
        source_checkpoint = dict(checkpoint or {})
        state["_checkpoint"] = {
            "schema_version": 1,
            "revision": 0,
            "branch_id": "root",
            "head_item_id": None,
            "extensions": source_checkpoint.get("extensions", {}),
            "fork_of": {
                "namespace": namespace,
                "thread_id": source_thread_id,
                "run_id": source_run_id,
                "item_id": at_item_id,
                "branch_id": source_checkpoint.get("branch_id"),
                "head_item_id": source_checkpoint.get("head_item_id"),
            },
        }
        messages = state.get("messages")
        if isinstance(messages, dict):
            messages["thread_id"] = target_thread_id
        now = time.time()

        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN")
            normalized = self._normalize_state(
                namespace,
                target_thread_id,
                state,
                now,
                executor=cur,
            )
            payload = self._serialize(normalized)
            cur.execute(
                _UPSERT_STATE,
                (
                    namespace,
                    target_thread_id,
                    target_run_id,
                    state.get("status", "running"),
                    payload,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        loaded = self.load_state(namespace, target_thread_id, target_run_id)
        if loaded is None:
            raise ValueError(
                f"Forked checkpoint `{target_run_id}` could not be loaded."
            )
        return loaded

    @_locked
    def save_with_event(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        now = time.time()
        status = state.get("status", "running")
        event_type = event.get("event_type", "unknown")
        event_data = self._serialize(event)

        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN")
            normalized = self._normalize_state(
                namespace, thread_id, state, now, executor=cur
            )
            payload = self._serialize(normalized)
            cur.execute(
                _UPSERT_STATE,
                (namespace, thread_id, run_id, status, payload, now, now),
            )
            cur.execute(
                _INSERT_EVENT,
                (namespace, thread_id, run_id, event_type, now, event_data),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    @_locked
    def list_runs(
        self,
        namespace: str,
        thread_id: str,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> List[Mapping[str, Any]]:
        query = (
            "SELECT run_id, status, updated_at FROM checkpoints "
            "WHERE namespace=? AND thread_id=?"
        )
        params: List[Any] = [namespace, thread_id]
        if status is not None:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY updated_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        rows = self._conn.execute(query, tuple(params)).fetchall()
        return [{"run_id": r[0], "status": r[1], "updated_at": r[2]} for r in rows]

    @_locked
    def delete_run(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> bool:
        deleted = self._conn.execute(
            _DELETE_RUN,
            (namespace, thread_id, run_id),
        ).rowcount
        self._cleanup_orphaned_items(namespace, thread_id)
        self._conn.commit()
        return bool(deleted)

    @_locked
    def clear(
        self,
        namespace: str | None = None,
        thread_id: str | None = None,
        *,
        older_than: float | None = None,
    ) -> int:
        select_query, delete_query, params = self._clear_queries(
            namespace=namespace,
            thread_id=thread_id,
            older_than=older_than,
        )

        affected_threads = self._conn.execute(select_query, tuple(params)).fetchall()
        deleted = self._conn.execute(delete_query, tuple(params)).rowcount
        for ns, tid in set(affected_threads):
            self._cleanup_orphaned_items(ns, tid)
        self._conn.commit()
        return deleted or 0

    @_locked
    def close(self) -> None:
        self._conn.close()
