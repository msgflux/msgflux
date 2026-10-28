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
# In SQLite this is represented by four tables. `checkpoints` is keyed by
# `(namespace, thread_id, run_id)` and stores the normalized JSON state plus the
# run status/timestamps. `checkpoint_message_items` is keyed by
# `(namespace, thread_id, item_ref)` and stores one frozen ChatMessages item JSON
# payload per content hash. `checkpoint_manifest_segments` stores immutable
# content-addressed manifest pages: each page holds up to
# `MANIFEST_PAGE_SIZE` occurrence entries and links to its parent page.
# `checkpoint_events` keeps append-only events for a run and is removed by
# SQLite's FK cascade when the run is deleted.
#
# The normalized state stored in `checkpoints.state` replaces `messages.items`
# with `_messages = {"state": <messages without items>, ...}`. Manifests up to
# `MANIFEST_PAGE_SIZE` entries keep their ordered list inline as `item_entries`
# for compatibility with existing snapshots. Larger manifests store only
# `manifest = {"format": "segments_v1", "head": <page ref>, "count": <N>}` and
# leave the occurrence list in linked pages. Both layouts rebuild the exact
# public message order and occurrence ids while item payloads remain
# deduplicated.
#
# Replacing an existing run's segmented head (including a switch to inline or
# metadata-only state) collects pages no longer reachable from any retained run
# in the same thread, inside the same publication transaction. New runs and
# same-head updates skip that GC; root-reachability sharing keeps pages used by
# other retained runs alive.

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from functools import wraps
from itertools import islice
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping
from uuid import uuid4

from msgflux._private.chat_items import (
    restore_item_occurrence,
    split_item_occurrence,
)
from msgflux._private.sqlite import prepare_sqlite_file
from msgflux.data.stores._manifest import (
    MANIFEST_PAGE_SIZE,
    build_manifest,
    iter_manifest_entries,
    iter_manifest_pages,
    replacement_needs_segment_gc,
    resolve_manifest_layout,
)
from msgflux.data.stores.base import (
    CheckpointCommit,
    CheckpointStore,
)
from msgflux.data.stores.observation import make_page, validate_read
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

_INSERT_MANIFEST_SEGMENT = """\
INSERT INTO checkpoint_manifest_segments
    (namespace, thread_id, segment_ref, data, created_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(namespace, thread_id, segment_ref) DO NOTHING
"""

_INSERT_EVENT = """\
INSERT INTO checkpoint_events
    (namespace, thread_id, run_id, event_type, timestamp, data)
VALUES (?, ?, ?, ?, ?, ?)
"""

_SELECT_STATE = """\
SELECT state FROM checkpoints WHERE namespace=? AND thread_id=? AND run_id=?
"""

_SELECT_MANIFEST_SEGMENT = """\
SELECT data FROM checkpoint_manifest_segments
WHERE namespace=? AND thread_id=? AND segment_ref=?
"""

_ITEM_BATCH_SIZE = 500

_DELETE_RUN = "DELETE FROM checkpoints WHERE namespace=? AND thread_id=? AND run_id=?"

_DELETE_ORPHAN_MESSAGE_ITEMS = """\
DELETE FROM checkpoint_message_items
WHERE namespace=? AND thread_id=?
AND item_ref NOT IN (SELECT ref FROM live_checkpoint_refs)
"""

_DELETE_ORPHAN_MANIFEST_SEGMENTS = """\
DELETE FROM checkpoint_manifest_segments
WHERE namespace=? AND thread_id=?
AND segment_ref NOT IN (SELECT ref FROM live_checkpoint_refs)
"""

_DELETE_ALL_MESSAGE_ITEMS = """\
DELETE FROM checkpoint_message_items WHERE namespace=? AND thread_id=?
"""

_DELETE_ALL_MANIFEST_SEGMENTS = """\
DELETE FROM checkpoint_manifest_segments WHERE namespace=? AND thread_id=?
"""

_CREATE_TABLES = """\
CREATE TABLE IF NOT EXISTS checkpoint_store_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

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

CREATE TABLE IF NOT EXISTS checkpoint_manifest_segments (
    namespace   TEXT NOT NULL,
    thread_id  TEXT NOT NULL,
    segment_ref TEXT NOT NULL,
    data        BLOB NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (namespace, thread_id, segment_ref)
);

CREATE INDEX IF NOT EXISTS idx_checkpoint_manifest_segments_thread
    ON checkpoint_manifest_segments(namespace, thread_id, segment_ref);

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

CREATE TABLE IF NOT EXISTS checkpoint_commits (
    namespace TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    stream_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (namespace, thread_id, run_id, stream_id, revision),
    FOREIGN KEY (namespace, thread_id, run_id)
        REFERENCES checkpoints(namespace, thread_id, run_id) ON DELETE CASCADE
);
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

    @_locked
    def read_commits(self, namespace, thread_id, run_id, *, after=None, limit=100):
        try:
            self._conn.execute("BEGIN")
            state = self.load_state(namespace, thread_id, run_id)
            latest = validate_read(state, namespace, thread_id, run_id, after, limit)
            rows = (
                []
                if after is None
                else self._conn.execute(
                    "SELECT revision, data FROM checkpoint_commits "
                    "WHERE namespace=? AND thread_id=? AND run_id=? AND stream_id=? "
                    "AND revision>? ORDER BY revision LIMIT ?",
                    (
                        namespace,
                        thread_id,
                        run_id,
                        latest.stream_id,
                        after.revision,
                        limit,
                    ),
                ).fetchall()
            )
            page = make_page(
                state,
                latest,
                after,
                [(revision, self._deserialize(data)) for revision, data in rows],
            )
            self._conn.commit()
            return page
        except BaseException as error:
            self._rollback_preserving_error(error)
            raise

    def __init__(self, path: str = ".msgflux/checkpoints.sqlite3") -> None:
        self.path = path
        self._lock = RLock()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        prepare_sqlite_file(path)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_CREATE_TABLES)
        self._conn.execute(
            "INSERT OR IGNORE INTO checkpoint_store_meta (key, value) VALUES (?, ?)",
            ("routing_id", uuid4().hex),
        )
        self._routing_id = self._conn.execute(
            "SELECT value FROM checkpoint_store_meta WHERE key = ?",
            ("routing_id",),
        ).fetchone()[0]
        self._conn.commit()

    @property
    def routing_id(self) -> str:
        return f"sqlite:{self._routing_id}"

    def _rollback_preserving_error(self, error: BaseException) -> None:
        try:
            if self._conn.in_transaction:
                self._conn.rollback()
        except BaseException as rollback_error:
            error.add_note(
                "SQLite checkpoint rollback also failed "
                f"({type(rollback_error).__name__})."
            )

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
        if len(item_entries) <= MANIFEST_PAGE_SIZE:
            normalized["_messages"] = {
                "state": message_state,
                "item_entries": item_entries,
            }
            return normalized

        def write_page(segment_ref: str, data: bytes) -> None:
            target.execute(
                _INSERT_MANIFEST_SEGMENT,
                (namespace, thread_id, segment_ref, data, now),
            )
            row = target.execute(
                _SELECT_MANIFEST_SEGMENT,
                (namespace, thread_id, segment_ref),
            ).fetchone()
            if row is None or bytes(row[0]) != data:
                # A ref is the digest of its bytes. An existing row with the
                # same ref but different bytes means the shared page is corrupt;
                # never publish a run that would silently reuse it.
                raise ValueError(
                    "Checkpoint manifest page is corrupted or could not be "
                    f"published: {segment_ref!r}"
                )

        normalized["_messages"] = {
            "state": message_state,
            "manifest": build_manifest(item_entries, write_page),
        }
        return normalized

    def _denormalize_state(
        self,
        namespace: str,
        thread_id: str,
        state_text: str,
    ) -> Dict[str, Any]:
        state = self._deserialize(state_text)
        self._validate_checkpoint_envelope(state.get("_checkpoint"))

        normalized_messages = state.pop("_messages", None)
        if not isinstance(normalized_messages, Mapping):
            return state

        message_state = normalized_messages.get("state")
        item_entries = self._stored_manifest_entries(
            namespace, thread_id, normalized_messages
        )
        if not isinstance(message_state, Mapping):
            raise ValueError("Checkpoint manifest is missing its message state")

        messages = dict(message_state)
        messages["items"] = list(
            self._iter_message_items(namespace, thread_id, item_entries)
        )
        state["messages"] = messages
        return state

    def _stored_manifest_entries(
        self,
        namespace: str,
        thread_id: str,
        normalized_messages: Mapping[str, Any],
    ) -> Iterable[Any]:
        """Choose the inline or segmented manifest layout for one snapshot."""
        item_entries, manifest = resolve_manifest_layout(normalized_messages)
        if item_entries is not None:
            return item_entries
        return iter_manifest_entries(
            manifest,
            lambda segment_ref: self._read_manifest_segment(
                namespace, thread_id, segment_ref
            ),
        )

    def _read_manifest_segment(
        self,
        namespace: str,
        thread_id: str,
        segment_ref: str,
    ) -> bytes:
        row = self._conn.execute(
            _SELECT_MANIFEST_SEGMENT,
            (namespace, thread_id, segment_ref),
        ).fetchone()
        if row is None:
            raise KeyError(segment_ref)
        return bytes(row[0])

    def _iter_message_items(
        self,
        namespace: str,
        thread_id: str,
        item_entries: Iterable[Any],
    ) -> Iterator[Dict[str, Any]]:
        """Yield restored items while validating payloads in bounded batches.

        The manifest is walked in manifest order, but rows are fetched by an
        indexed ``item_ref IN (...)`` query per batch instead of one query per
        occurrence. Each referenced payload is still deserialized and restored,
        so malformed JSON, wrong payload shapes and corruption fail identically
        to the unbatched path. No full old timeline is accumulated.
        """
        entry_iterator = iter(item_entries)
        while True:
            batch = list(islice(entry_iterator, _ITEM_BATCH_SIZE))
            if not batch:
                return
            refs = self._batch_item_refs(namespace, thread_id, batch)
            item_text = self._fetch_item_texts(namespace, thread_id, refs)
            self._ensure_batch_refs_present(namespace, thread_id, batch, item_text)
            for entry in batch:
                item_ref = entry["item_ref"]
                payload = self._decode_item_payload(
                    namespace, thread_id, item_ref, item_text[item_ref]
                )
                yield restore_item_occurrence(payload, entry)

    @staticmethod
    def _batch_item_refs(
        namespace: str,
        thread_id: str,
        batch: List[Any],
    ) -> List[str]:
        refs: List[str] = []
        seen: set[str] = set()
        for entry in batch:
            if not isinstance(entry, Mapping):
                raise ValueError("Checkpoint message entry is corrupted")
            item_ref = entry.get("item_ref")
            if not isinstance(item_ref, str) or not item_ref:
                raise ValueError(
                    "Checkpoint message item is missing or corrupted: "
                    f"{namespace}/{thread_id}/{item_ref!r}"
                )
            if item_ref not in seen:
                seen.add(item_ref)
                refs.append(item_ref)
        return refs

    def _fetch_item_texts(
        self,
        namespace: str,
        thread_id: str,
        refs: List[str],
    ) -> Dict[str, str]:
        placeholders = ",".join("?" for _ in refs)
        rows = self._conn.execute(
            "SELECT item_ref, item FROM checkpoint_message_items "  # noqa: S608 - only `?` placeholders
            f"WHERE namespace=? AND thread_id=? AND item_ref IN ({placeholders})",
            (namespace, thread_id, *refs),
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    @staticmethod
    def _ensure_batch_refs_present(
        namespace: str,
        thread_id: str,
        batch: List[Any],
        item_text: Mapping[str, str],
    ) -> None:
        for entry in batch:
            item_ref = entry["item_ref"]
            if item_ref not in item_text:
                raise ValueError(
                    "Checkpoint message item is missing or corrupted: "
                    f"{namespace}/{thread_id}/{item_ref!r}"
                )

    def _decode_item_payload(
        self,
        namespace: str,
        thread_id: str,
        item_ref: str,
        text: str,
    ) -> Dict[str, Any]:
        payload = self._deserialize(text)
        if not isinstance(payload, Mapping):
            raise ValueError(
                "Checkpoint message item is missing or corrupted: "
                f"{namespace}/{thread_id}/{item_ref!r}"
            )
        return payload

    def _load_commit_envelope(
        self,
        namespace: str,
        thread_id: str,
        state_text: str,
    ) -> Dict[str, Any]:
        """Return only the previous envelope needed by a revision commit.

        ``_prepare_revision_state`` reads ``_checkpoint`` from the current state
        and inherits its branch, head and extensions. Hydrating the prior
        timeline into a full ``messages`` mapping is unnecessary, but its
        payloads are still validated in bounded batches so corruption cannot be
        published.
        """
        state = self._deserialize(state_text)
        self._validate_checkpoint_envelope(state.get("_checkpoint"))
        normalized_messages = state.get("_messages")
        if isinstance(normalized_messages, Mapping):
            item_entries = self._stored_manifest_entries(
                namespace, thread_id, normalized_messages
            )
            if not isinstance(normalized_messages.get("state"), Mapping):
                raise ValueError("Checkpoint manifest is missing its message state")
            for _ in self._iter_message_items(namespace, thread_id, item_entries):
                pass
        if "_checkpoint" not in state:
            return {}
        return {"_checkpoint": state["_checkpoint"]}

    @staticmethod
    def _item_ref(item: Any) -> str:
        payload = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        return "item_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _list_live_refs(
        self, namespace: str, thread_id: str
    ) -> tuple[set[str], set[str]]:
        rows = self._conn.execute(
            "SELECT state FROM checkpoints WHERE namespace=? AND thread_id=?",
            (namespace, thread_id),
        ).fetchall()
        live_items: set[str] = set()
        live_segments: set[str] = set()
        seen_segments: set[str] = set()
        for row in rows:
            state = self._deserialize(row[0])
            messages = state.get("_messages")
            if not isinstance(messages, Mapping):
                continue
            item_entries, manifest = resolve_manifest_layout(messages)
            if not isinstance(messages.get("state"), Mapping):
                raise ValueError("Checkpoint manifest is missing its message state")
            if item_entries is not None:
                live_items.update(self._collect_entry_refs(item_entries))
            if manifest is None:
                continue
            for segment_ref, entries in iter_manifest_pages(
                manifest,
                lambda ref: self._read_manifest_segment(namespace, thread_id, ref),
            ):
                if segment_ref in seen_segments:
                    # The chain is content-addressed: a seen page implies all of
                    # its ancestors were already collected for another run.
                    break
                seen_segments.add(segment_ref)
                live_segments.add(segment_ref)
                live_items.update(self._collect_entry_refs(entries))
        return live_items, live_segments

    @staticmethod
    def _collect_entry_refs(entries: Iterable[Any]) -> set[str]:
        refs: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"Checkpoint message entry {index} is corrupted")
            item_ref = entry.get("item_ref")
            if not isinstance(item_ref, str) or not item_ref:
                raise ValueError(
                    f"Checkpoint message entry {index} is missing or corrupted"
                )
            refs.add(item_ref)
        return refs

    def _cleanup_orphaned_items(self, namespace: str, thread_id: str) -> None:
        live_items, live_segments = self._list_live_refs(namespace, thread_id)
        self._delete_unreachable(
            live_items,
            orphan_query=_DELETE_ORPHAN_MESSAGE_ITEMS,
            delete_all_query=_DELETE_ALL_MESSAGE_ITEMS,
            namespace=namespace,
            thread_id=thread_id,
        )
        self._delete_unreachable(
            live_segments,
            orphan_query=_DELETE_ORPHAN_MANIFEST_SEGMENTS,
            delete_all_query=_DELETE_ALL_MANIFEST_SEGMENTS,
            namespace=namespace,
            thread_id=thread_id,
        )

    def _delete_unreachable(
        self,
        live: set[str],
        *,
        orphan_query: str,
        delete_all_query: str,
        namespace: str,
        thread_id: str,
    ) -> None:
        if live:
            self._conn.execute(
                "CREATE TEMP TABLE IF NOT EXISTS live_checkpoint_refs "
                "(ref TEXT PRIMARY KEY)"
            )
            self._conn.execute("DELETE FROM live_checkpoint_refs")
            self._conn.executemany(
                "INSERT INTO live_checkpoint_refs(ref) VALUES (?)",
                [(ref,) for ref in live],
            )
            self._conn.execute(orphan_query, (namespace, thread_id))
            self._conn.execute("DELETE FROM live_checkpoint_refs")
            return
        self._conn.execute(delete_all_query, (namespace, thread_id))

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
        cur = self._conn.cursor()
        try:
            # Reserve the writer before reading the old head so an independent
            # connection cannot replace that head while this snapshot is built.
            cur.execute("BEGIN IMMEDIATE")
            old_row = cur.execute(
                _SELECT_STATE, (namespace, thread_id, run_id)
            ).fetchone()
            normalized = self._normalize_state(
                namespace, thread_id, state, now, executor=cur
            )
            gc_segments = old_row is not None and replacement_needs_segment_gc(
                self._deserialize(old_row[0]), normalized
            )
            payload = self._serialize(normalized)
            status = state.get("status", "running")
            cur.execute(
                _UPSERT_STATE,
                (namespace, thread_id, run_id, status, payload, now, now),
            )
            if gc_segments:
                # Snapshot replacement: release segment heads that only the
                # replaced state referenced. New runs never take this path.
                self._cleanup_orphaned_items(namespace, thread_id)
            self._conn.commit()
        except BaseException as error:
            self._rollback_preserving_error(error)
            raise

    @_locked
    def load_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> Mapping[str, Any] | None:
        # State and item rows must come from one snapshot; an independent
        # delete/clear between reads would otherwise look like corruption.
        started = not self._conn.in_transaction
        try:
            if started:
                self._conn.execute("BEGIN")
            row = self._conn.execute(
                _SELECT_STATE,
                (namespace, thread_id, run_id),
            ).fetchone()
            state = (
                None
                if row is None
                else self._denormalize_state(namespace, thread_id, row[0])
            )
            if started:
                self._conn.commit()
            return state
        except BaseException as error:
            if started:
                self._rollback_preserving_error(error)
            raise

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
        try:
            self._conn.execute(
                _INSERT_EVENT,
                (namespace, thread_id, run_id, event_type, now, data),
            )
            self._conn.commit()
        except BaseException as error:
            self._rollback_preserving_error(error)
            raise

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
                self._load_commit_envelope(namespace, thread_id, row[0]) if row else {}
            )
            committed, next_revision = self._prepare_revision_state(
                state,
                current,
                expected_revision=expected_revision,
                branch_id=branch_id,
                head_item_id=head_item_id,
                extension_state=extension_state,
            )
            normalized = self._normalize_state(
                namespace, thread_id, committed, now, executor=cur
            )
            gc_segments = row is not None and replacement_needs_segment_gc(
                self._deserialize(row[0]), normalized
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
            if gc_segments:
                self._cleanup_orphaned_items(namespace, thread_id)
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
            cur.execute(
                "INSERT INTO checkpoint_commits VALUES (?, ?, ?, ?, ?, ?)",
                (
                    namespace,
                    thread_id,
                    run_id,
                    committed["_checkpoint"]["stream_id"],
                    next_revision,
                    self._serialize(
                        event
                        if event is not None
                        else {
                            "event_type": "checkpoint",
                            "status": committed.get("status"),
                        }
                    ),
                ),
            )
            self._conn.commit()
            return CheckpointCommit(
                next_revision,
                committed,
                committed["_checkpoint"]["branch_id"],
                committed["_checkpoint"]["head_item_id"],
            )
        except BaseException as error:
            self._rollback_preserving_error(error)
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
        cur = self._conn.cursor()
        try:
            # Keep source hydration and target publication in one transaction.
            cur.execute("BEGIN IMMEDIATE")
            row = cur.execute(
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
            target_row = cur.execute(
                _SELECT_STATE,
                (namespace, target_thread_id, target_run_id),
            ).fetchone()
            state = self._prepare_fork_state(
                source_state,
                at_item_id=at_item_id,
                position=position,
            )
            if status is not None:
                state["status"] = status
            self._set_fork_metadata(
                state,
                namespace=namespace,
                source_thread_id=source_thread_id,
                source_run_id=source_run_id,
                target_thread_id=target_thread_id,
                target_run_id=target_run_id,
                at_item_id=at_item_id,
            )
            messages = state.get("messages")
            if isinstance(messages, dict):
                messages["thread_id"] = target_thread_id
            now = time.time()
            normalized = self._normalize_state(
                namespace,
                target_thread_id,
                state,
                now,
                executor=cur,
            )
            payload = self._serialize(normalized)
            gc_segments = target_row is not None and replacement_needs_segment_gc(
                self._deserialize(target_row[0]), normalized
            )
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
            if gc_segments:
                self._cleanup_orphaned_items(namespace, target_thread_id)
            self._conn.commit()
        except BaseException as error:
            self._rollback_preserving_error(error)
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
            cur.execute("BEGIN IMMEDIATE")
            old_row = cur.execute(
                _SELECT_STATE, (namespace, thread_id, run_id)
            ).fetchone()
            normalized = self._normalize_state(
                namespace, thread_id, state, now, executor=cur
            )
            gc_segments = old_row is not None and replacement_needs_segment_gc(
                self._deserialize(old_row[0]), normalized
            )
            payload = self._serialize(normalized)
            cur.execute(
                _UPSERT_STATE,
                (namespace, thread_id, run_id, status, payload, now, now),
            )
            if gc_segments:
                self._cleanup_orphaned_items(namespace, thread_id)
            cur.execute(
                _INSERT_EVENT,
                (namespace, thread_id, run_id, event_type, now, event_data),
            )
            self._conn.commit()
        except BaseException as error:
            self._rollback_preserving_error(error)
            raise

    @_locked
    def load_latest_run(
        self,
        namespace: str,
        thread_id: str,
    ) -> Mapping[str, Any] | None:
        # Selection and hydration must observe the same committed snapshot.
        started = not self._conn.in_transaction
        try:
            if started:
                self._conn.execute("BEGIN")
            state = super().load_latest_run(namespace, thread_id)
            if started:
                self._conn.commit()
            return state
        except BaseException as error:
            if started:
                self._rollback_preserving_error(error)
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
        try:
            deleted = self._conn.execute(
                _DELETE_RUN,
                (namespace, thread_id, run_id),
            ).rowcount
            self._cleanup_orphaned_items(namespace, thread_id)
            self._conn.commit()
            return bool(deleted)
        except BaseException as error:
            self._rollback_preserving_error(error)
            raise

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

        try:
            affected_threads = self._conn.execute(
                select_query, tuple(params)
            ).fetchall()
            deleted = self._conn.execute(delete_query, tuple(params)).rowcount
            for ns, tid in set(affected_threads):
                self._cleanup_orphaned_items(ns, tid)
            self._conn.commit()
            return deleted or 0
        except BaseException as error:
            self._rollback_preserving_error(error)
            raise

    @_locked
    def close(self) -> None:
        self._conn.close()
