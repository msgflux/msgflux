"""SQL write-volume and garbage-collection semantics for incremental appends.

These tests observe the provider through its public checkpoint API and the
actual SQLite statements it executes. They do not reimplement the collector:
each assertion checks a user-visible archive invariant, a written SQL shape or
a transaction boundary.
"""

import json
import sqlite3
from contextlib import contextmanager
from copy import deepcopy

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.data.stores._manifest import (
    iter_manifest_entries,
    iter_manifest_pages,
    resolve_manifest_layout,
)

NS = "gc-writes"
THREAD = "thread"
RUN = "run"

OTHER_NS = "gc-writes-other-namespace"
OTHER_THREAD = "gc-writes-other-thread"

_READ_SEGMENT = """\
SELECT data FROM checkpoint_manifest_segments
WHERE namespace=? AND thread_id=? AND segment_ref=?
"""

_COUNT_PAYLOADS = """\
SELECT COUNT(*) FROM checkpoint_message_items
WHERE namespace=? AND thread_id=?
"""

_COUNT_SEGMENTS = """\
SELECT COUNT(*) FROM checkpoint_manifest_segments
WHERE namespace=? AND thread_id=?
"""

_COUNT_EVENTS = """\
SELECT COUNT(*) FROM checkpoint_events
WHERE namespace=? AND thread_id=? AND run_id=?
"""

_COUNT_COMMITS = """\
SELECT COUNT(*) FROM checkpoint_commits
WHERE namespace=? AND thread_id=? AND run_id=?
"""

_SELECT_STORED_PAYLOADS = """\
SELECT item_ref FROM checkpoint_message_items
WHERE namespace=? AND thread_id=?
"""

_SELECT_STORED_SEGMENTS = """\
SELECT segment_ref FROM checkpoint_manifest_segments
WHERE namespace=? AND thread_id=?
"""

_SELECT_RAW_STATE = """\
SELECT state FROM checkpoints
WHERE namespace=? AND thread_id=? AND run_id=?
"""

_INSERT_PAYLOAD = """\
INSERT INTO checkpoint_message_items
    (namespace, thread_id, item_ref, item, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
"""

_INSERT_SEGMENT = """\
INSERT INTO checkpoint_manifest_segments
    (namespace, thread_id, segment_ref, data, created_at)
VALUES (?, ?, ?, ?, ?)
"""

_FAIL_EVENT_TRIGGER = """\
CREATE TRIGGER gated_event_insert
BEFORE INSERT ON checkpoint_events
BEGIN
    SELECT RAISE(ABORT, 'event insert blocked');
END
"""


@pytest.fixture
def sqlite_store(tmp_path):
    store = SQLiteCheckpointStore(str(tmp_path / "checkpoints.sqlite3"))
    try:
        yield store
    finally:
        store.close()


def _items(count, *, prefix="itm", content_prefix="content"):
    return [
        {
            "role": "user",
            "content": f"{content_prefix}-{index}",
            "item_id": f"{prefix}_{index}",
        }
        for index in range(count)
    ]


def _new_item():
    return {"role": "assistant", "content": "appended", "item_id": "itm_appended"}


def _state(namespace, thread_id, items, *, status="running"):
    return {
        "status": status,
        "messages": ChatMessages(
            items, namespace=namespace, thread_id=thread_id
        )._to_state(),
    }


def _commit_segmented(
    store,
    *,
    namespace=NS,
    thread_id=THREAD,
    run_id=RUN,
    count=300,
    prefix="itm",
    content_prefix="content",
    items=None,
):
    seeded = (
        items
        if items is not None
        else _items(count, prefix=prefix, content_prefix=content_prefix)
    )
    state = _state(namespace, thread_id, seeded)
    store.commit_state(
        namespace,
        thread_id,
        run_id,
        state,
        expected_revision=0,
    )
    return deepcopy(state["messages"]["items"])


def _raw_state(store, namespace=NS, thread_id=THREAD, run_id=RUN):
    row = store._conn.execute(
        _SELECT_RAW_STATE, (namespace, thread_id, run_id)
    ).fetchone()
    assert row is not None
    return json.loads(row[0])


def _manifest_head(store, namespace=NS, thread_id=THREAD, run_id=RUN):
    manifest = _raw_state(store, namespace, thread_id, run_id)["_messages"]["manifest"]
    return manifest["head"]


def _read_segment(store, namespace, thread_id, segment_ref):
    row = store._conn.execute(
        _READ_SEGMENT, (namespace, thread_id, segment_ref)
    ).fetchone()
    if row is None:
        raise KeyError(segment_ref)
    return bytes(row[0])


def _refs_from_state(store, namespace, thread_id, raw):
    entries, manifest = resolve_manifest_layout(raw["_messages"])
    item_refs = set()
    segment_refs = set()
    if entries is not None:
        item_refs.update(entry["item_ref"] for entry in entries)
    if manifest is not None:
        for segment_ref, page_entries in iter_manifest_pages(
            manifest,
            lambda ref: _read_segment(store, namespace, thread_id, ref),
        ):
            segment_refs.add(segment_ref)
            item_refs.update(entry["item_ref"] for entry in page_entries)
    return item_refs, segment_refs


def _live_refs(store, namespace=NS, thread_id=THREAD):
    rows = store._conn.execute(
        """\
SELECT run_id, state FROM checkpoints
WHERE namespace=? AND thread_id=?
""",
        (namespace, thread_id),
    ).fetchall()
    item_refs = set()
    segment_refs = set()
    for _run_id, state_text in rows:
        run_items, run_segments = _refs_from_state(
            store, namespace, thread_id, json.loads(state_text)
        )
        item_refs.update(run_items)
        segment_refs.update(run_segments)
    return item_refs, segment_refs


def _entry_refs(store, namespace=NS, thread_id=THREAD, run_id=RUN):
    """Return manifest occurrence refs in stored timeline order."""
    raw = _raw_state(store, namespace, thread_id, run_id)
    entries, manifest = resolve_manifest_layout(raw["_messages"])
    if entries is not None:
        return [entry["item_ref"] for entry in entries]
    return [
        entry["item_ref"]
        for entry in iter_manifest_entries(
            manifest,
            lambda ref: _read_segment(store, namespace, thread_id, ref),
        )
    ]


def _stored_payload_refs(store, namespace=NS, thread_id=THREAD):
    if isinstance(store, SQLiteCheckpointStore):
        return {
            row[0]
            for row in store._conn.execute(
                _SELECT_STORED_PAYLOADS, (namespace, thread_id)
            )
        }
    return set(store._message_items.get(namespace, {}).get(thread_id, {}))


def _stored_segment_refs(store, namespace=NS, thread_id=THREAD):
    if isinstance(store, SQLiteCheckpointStore):
        return {
            row[0]
            for row in store._conn.execute(
                _SELECT_STORED_SEGMENTS, (namespace, thread_id)
            )
        }
    return set(store._manifest_pages.get(namespace, {}).get(thread_id, {}))


def _count_payloads(store, namespace=NS, thread_id=THREAD):
    return store._conn.execute(_COUNT_PAYLOADS, (namespace, thread_id)).fetchone()[0]


def _count_segments(store, namespace=NS, thread_id=THREAD):
    return store._conn.execute(_COUNT_SEGMENTS, (namespace, thread_id)).fetchone()[0]


def _count_events(store, namespace=NS, thread_id=THREAD, run_id=RUN):
    return store._conn.execute(
        _COUNT_EVENTS, (namespace, thread_id, run_id)
    ).fetchone()[0]


def _count_commits(store, namespace=NS, thread_id=THREAD, run_id=RUN):
    return store._conn.execute(
        _COUNT_COMMITS, (namespace, thread_id, run_id)
    ).fetchone()[0]


def _insert_orphan_payloads(store, refs, *, namespace=NS, thread_id=THREAD):
    store._conn.executemany(
        _INSERT_PAYLOAD,
        [(namespace, thread_id, ref, "{}", 1.0, 1.0) for ref in refs],
    )
    store._conn.commit()


def _insert_orphan_segments(store, refs, *, namespace=NS, thread_id=THREAD):
    store._conn.executemany(
        _INSERT_SEGMENT,
        [(namespace, thread_id, ref, b"orphan-page", 1.0) for ref in refs],
    )
    store._conn.commit()


def _assert_archive_matches_live_refs(store, namespace=NS, thread_id=THREAD):
    live_items, live_segments = _live_refs(store, namespace, thread_id)
    assert _stored_payload_refs(store, namespace, thread_id) == live_items
    assert _stored_segment_refs(store, namespace, thread_id) == live_segments


def _statements_starting(statements, prefix):
    marker = prefix.lstrip().upper()
    return [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith(marker)
    ]


class _InjectedDeleteError(RuntimeError):
    """Raised by the recording connection to abort a chosen delete batch."""


class _RecordingConnection:
    """Transparent SQLite proxy that records SQL text and bound parameters."""

    def __init__(self, connection, *, fail_on_delete=None):
        self._conn = connection
        self.calls = []
        self.delete_calls = 0
        self.fail_on_delete = fail_on_delete

    def execute(self, sql, parameters=()):
        bound = tuple(parameters) if parameters is not None else ()
        self.calls.append((sql, bound))
        if sql.lstrip().upper().startswith("DELETE FROM CHECKPOINT_MESSAGE_ITEMS"):
            self.delete_calls += 1
            if self.delete_calls == self.fail_on_delete:
                raise _InjectedDeleteError(
                    "injected failure in checkpoint payload delete batch"
                )
        return self._conn.execute(sql, parameters)

    def executemany(self, sql, parameters):
        self.calls.append((sql, None))
        return self._conn.executemany(sql, parameters)

    def cursor(self):
        return self._conn.cursor()

    def __getattr__(self, name):
        return getattr(self._conn, name)


@contextmanager
def _recording_connection(store, *, fail_on_delete=None):
    original = store._conn
    proxy = _RecordingConnection(original, fail_on_delete=fail_on_delete)
    store._conn = proxy
    try:
        yield proxy
    finally:
        store._conn = original


_DELETE_MARKERS = {
    "checkpoint_message_items": "DELETE FROM CHECKPOINT_MESSAGE_ITEMS",
    "checkpoint_manifest_segments": "DELETE FROM CHECKPOINT_MANIFEST_SEGMENTS",
}


def _deletes(proxy, table):
    marker = _DELETE_MARKERS[table]
    return [
        (sql, parameters)
        for sql, parameters in proxy.calls
        if sql.lstrip().upper().startswith(marker)
    ]


def test_large_append_traces_no_temp_live_set_and_only_delta_payload_writes(
    sqlite_store,
):
    store = sqlite_store
    initial_items = _commit_segmented(store, count=300)
    before_head = _manifest_head(store)
    before_payloads = _stored_payload_refs(store)
    tail = store.load_tail(NS, THREAD, RUN, limit=0)

    statements = []
    store._conn.set_trace_callback(statements.append)
    try:
        acknowledgment = store.append_state(
            NS,
            THREAD,
            RUN,
            [_new_item()],
            base=tail.base,
            event={"event_type": "gc_write_probe"},
        )
    finally:
        store._conn.set_trace_callback(None)

    lowered = "\n".join(statements).lower()
    assert "live_checkpoint_refs" not in lowered
    assert "temp" not in lowered
    assert (
        store._conn.execute("SELECT COUNT(*) FROM sqlite_temp_master").fetchone()[0]
        == 0
    )

    payload_inserts = _statements_starting(
        statements, "INSERT INTO checkpoint_message_items"
    )
    segment_inserts = _statements_starting(
        statements, "INSERT INTO checkpoint_manifest_segments"
    )
    assert len(payload_inserts) == 1
    assert len(segment_inserts) == 1
    assert all(ref not in payload_inserts[0] for ref in before_payloads)
    assert not _statements_starting(statements, "DELETE FROM checkpoint_message_items")

    assert acknowledgment.start == 300
    assert acknowledgment.base.revision == 2
    assert acknowledgment.base.item_count == 301
    state = store.load_state(NS, THREAD, RUN)
    assert state["messages"]["items"] == [*initial_items, _new_item()]
    assert state["_checkpoint"]["revision"] == 2
    assert store.load_events(NS, THREAD, RUN) == [{"event_type": "gc_write_probe"}]
    assert store.load_tail(NS, THREAD, RUN, limit=0).base == acknowledgment.base

    # The single-run append replaces the partial head page; that page becomes
    # garbage while every full prefix page and every payload stays live.
    assert before_head not in _stored_segment_refs(store, NS, THREAD)
    assert _count_payloads(store) == 301
    assert _count_segments(store) == 3
    _assert_archive_matches_live_refs(store)


def test_shared_alias_and_shared_root_survive_replacement_gc(sqlite_store):
    store = sqlite_store
    aliased = _items(300)
    aliased[1]["content"] = aliased[0]["content"]
    initial_items = _commit_segmented(store, run_id=RUN, items=aliased)
    _commit_segmented(store, run_id="shared-run", items=aliased)

    refs = _entry_refs(store, run_id=RUN)
    assert refs[0] == refs[1]
    alias_ref = refs[0]
    shared_head = _manifest_head(store, run_id=RUN)
    tail = store.load_tail(NS, THREAD, RUN, limit=0)

    statements = []
    store._conn.set_trace_callback(statements.append)
    try:
        store.append_state(
            NS,
            THREAD,
            RUN,
            [_new_item()],
            base=tail.base,
            event={"event_type": "shared_probe"},
        )
    finally:
        store._conn.set_trace_callback(None)

    # The sweep still reads stored keys, but neither provider rows nor pages
    # are garbage while the second run keeps the old head reachable.
    assert any(
        "SELECT item_ref FROM checkpoint_message_items" in statement
        for statement in statements
    )
    assert any(
        "SELECT segment_ref FROM checkpoint_manifest_segments" in statement
        for statement in statements
    )
    assert not _statements_starting(statements, "DELETE FROM checkpoint_message_items")
    assert not _statements_starting(
        statements, "DELETE FROM checkpoint_manifest_segments"
    )

    stored_payloads = _stored_payload_refs(store)
    stored_segments = _stored_segment_refs(store)
    assert alias_ref in stored_payloads
    assert shared_head in stored_segments
    assert _count_payloads(store) == 300
    assert _count_segments(store) == 4

    appended_refs = _entry_refs(store, run_id=RUN)
    assert appended_refs[0] == appended_refs[1] == alias_ref
    assert store.load_state(NS, THREAD, RUN)["messages"]["items"] == [
        *initial_items,
        _new_item(),
    ]
    shared_state = store.load_state(NS, THREAD, "shared-run")
    assert shared_state["messages"]["items"] == initial_items
    assert store.load_events(NS, THREAD, "shared-run") == []
    assert store.load_events(NS, THREAD, RUN) == [{"event_type": "shared_probe"}]
    _assert_archive_matches_live_refs(store)


def test_preexisting_payload_and_page_garbage_is_collected_during_append(
    sqlite_store,
):
    store = sqlite_store
    initial_items = _commit_segmented(store, count=300)
    before_head = _manifest_head(store)
    orphan_payloads = {f"item_preexisting_dead_{index:03d}" for index in range(7)}
    orphan_segments = {f"seg_preexisting_dead_{index:03d}" for index in range(3)}
    _insert_orphan_payloads(store, orphan_payloads)
    _insert_orphan_segments(store, orphan_segments)
    assert orphan_payloads <= _stored_payload_refs(store)
    assert orphan_segments <= _stored_segment_refs(store)
    tail = store.load_tail(NS, THREAD, RUN, limit=0)

    statements = []
    store._conn.set_trace_callback(statements.append)
    try:
        store.append_state(
            NS,
            THREAD,
            RUN,
            [_new_item()],
            base=tail.base,
            event={"event_type": "preexisting_probe"},
        )
    finally:
        store._conn.set_trace_callback(None)

    payload_deletes = _statements_starting(
        statements, "DELETE FROM checkpoint_message_items"
    )
    segment_deletes = _statements_starting(
        statements, "DELETE FROM checkpoint_manifest_segments"
    )
    assert len(payload_deletes) == 1
    assert len(segment_deletes) == 1
    assert all(ref in payload_deletes[0] for ref in orphan_payloads)

    assert not (orphan_payloads & _stored_payload_refs(store))
    assert not (orphan_segments & _stored_segment_refs(store))
    assert before_head not in _stored_segment_refs(store)
    assert store.load_state(NS, THREAD, RUN)["messages"]["items"] == [
        *initial_items,
        _new_item(),
    ]
    assert store.load_events(NS, THREAD, RUN) == [{"event_type": "preexisting_probe"}]
    assert _count_payloads(store) == 301
    assert _count_segments(store) == 3
    _assert_archive_matches_live_refs(store)


def test_more_than_one_thousand_orphans_use_bounded_parameterized_batches(
    sqlite_store,
):
    store = sqlite_store
    _commit_segmented(store, count=300)
    before_head = _manifest_head(store)
    orphan_payloads = {f"item_orphan_batch_{index:04d}" for index in range(1201)}
    orphan_segments = {f"seg_orphan_batch_{index:04d}" for index in range(1005)}
    _insert_orphan_payloads(store, orphan_payloads)
    _insert_orphan_segments(store, orphan_segments)
    tail = store.load_tail(NS, THREAD, RUN, limit=0)

    with _recording_connection(store) as proxy:
        store.append_state(
            NS,
            THREAD,
            RUN,
            [_new_item()],
            base=tail.base,
            event={"event_type": "batch_probe"},
        )

    payload_deletes = _deletes(proxy, "checkpoint_message_items")
    segment_deletes = _deletes(proxy, "checkpoint_manifest_segments")
    assert len(payload_deletes) == 3
    assert len(segment_deletes) == 3
    for sql, parameters in [*payload_deletes, *segment_deletes]:
        assert sql.count("?") == len(parameters)
        assert 2 < len(parameters) <= 999
        assert len(parameters) - 2 <= 500
        assert parameters[0] == NS
        assert parameters[1] == THREAD
    assert max(len(parameters) for _sql, parameters in payload_deletes) == 502

    bound_payloads = {
        ref for _sql, parameters in payload_deletes for ref in parameters[2:]
    }
    bound_segments = {
        ref for _sql, parameters in segment_deletes for ref in parameters[2:]
    }
    assert bound_payloads == orphan_payloads
    assert bound_segments == orphan_segments | {before_head}
    assert not (orphan_payloads & _stored_payload_refs(store))
    assert not (orphan_segments & _stored_segment_refs(store))
    assert store.load_events(NS, THREAD, RUN) == [{"event_type": "batch_probe"}]
    assert _count_payloads(store) == 301
    assert _count_segments(store) == 3
    _assert_archive_matches_live_refs(store)


def test_gc_is_scoped_to_namespace_and_thread(sqlite_store):
    store = sqlite_store
    first_items = _commit_segmented(store, count=300)
    _commit_segmented(
        store,
        namespace=OTHER_NS,
        thread_id=THREAD,
        run_id="other-namespace-run",
        count=130,
        prefix="other_ns",
    )
    _commit_segmented(
        store,
        namespace=NS,
        thread_id=OTHER_THREAD,
        run_id="other-thread-run",
        count=130,
        prefix="other_thread",
    )
    other_scopes = [
        (OTHER_NS, THREAD, "other-namespace-run"),
        (NS, OTHER_THREAD, "other-thread-run"),
    ]
    for namespace, thread_id, run_id in other_scopes:
        _insert_orphan_payloads(
            store,
            [f"item_scoped_dead_{run_id}"],
            namespace=namespace,
            thread_id=thread_id,
        )
        _insert_orphan_segments(
            store,
            [f"seg_scoped_dead_{run_id}"],
            namespace=namespace,
            thread_id=thread_id,
        )
    snapshots = {
        (namespace, thread_id, run_id): (
            json.dumps(_raw_state(store, namespace, thread_id, run_id)),
            _stored_payload_refs(store, namespace, thread_id),
            _stored_segment_refs(store, namespace, thread_id),
        )
        for namespace, thread_id, run_id in other_scopes
    }
    tail = store.load_tail(NS, THREAD, RUN, limit=0)

    with _recording_connection(store) as proxy:
        store.append_state(
            NS,
            THREAD,
            RUN,
            [_new_item()],
            base=tail.base,
            event={"event_type": "isolation_probe"},
        )

    for _sql, parameters in _deletes(proxy, "checkpoint_message_items"):
        assert parameters[:2] == (NS, THREAD)
    for _sql, parameters in _deletes(proxy, "checkpoint_manifest_segments"):
        assert parameters[:2] == (NS, THREAD)

    for namespace, thread_id, run_id in other_scopes:
        raw, payloads, segments = snapshots[(namespace, thread_id, run_id)]
        assert json.dumps(_raw_state(store, namespace, thread_id, run_id)) == raw
        assert _stored_payload_refs(store, namespace, thread_id) == payloads
        assert _stored_segment_refs(store, namespace, thread_id) == segments
        assert f"item_scoped_dead_{run_id}" in payloads
        assert f"seg_scoped_dead_{run_id}" in segments
        assert store.load_events(namespace, thread_id, run_id) == []

    assert store.load_state(NS, THREAD, RUN)["messages"]["items"] == [
        *first_items,
        _new_item(),
    ]
    assert store.load_events(NS, THREAD, RUN) == [{"event_type": "isolation_probe"}]
    _assert_archive_matches_live_refs(store)


def test_quoted_stored_refs_are_bound_and_only_orphans_are_deleted(sqlite_store):
    store = sqlite_store
    _commit_segmented(store, count=300)
    quoted_payloads = {
        "item_orphan_single'quote",
        'item_orphan_double"quote',
        "item_orphan_semicolon; DROP TABLE checkpoints; --",
        "item_orphan_question_?_and_percent_%",
        "item_orphan_comma,closing_paren)",
    }
    quoted_segments = {
        "seg_orphan_single'quote",
        "seg_orphan_semicolon; DROP TABLE checkpoint_manifest_segments; --",
    }
    _insert_orphan_payloads(store, quoted_payloads)
    _insert_orphan_segments(store, quoted_segments)
    tail = store.load_tail(NS, THREAD, RUN, limit=0)

    with _recording_connection(store) as proxy:
        store.append_state(
            NS,
            THREAD,
            RUN,
            [_new_item()],
            base=tail.base,
            event={"event_type": "quoted_probe"},
        )

    all_deletes = [
        *_deletes(proxy, "checkpoint_message_items"),
        *_deletes(proxy, "checkpoint_manifest_segments"),
    ]
    assert all_deletes
    for sql, parameters in all_deletes:
        assert sql.count("?") == len(parameters)
        assert all(value not in sql for value in quoted_payloads | quoted_segments)
        assert parameters[0] == NS
        assert parameters[1] == THREAD
    bound_values = {
        parameter for _sql, parameters in all_deletes for parameter in parameters[2:]
    }
    assert quoted_payloads <= bound_values
    assert quoted_segments <= bound_values
    assert not (quoted_payloads & _stored_payload_refs(store))
    assert not (quoted_segments & _stored_segment_refs(store))
    assert (
        store._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='checkpoints'"
        ).fetchone()[0]
        == 1
    )
    assert store.load_state(NS, THREAD, RUN)["messages"]["items"][-1] == _new_item()
    assert store.load_events(NS, THREAD, RUN) == [{"event_type": "quoted_probe"}]
    _assert_archive_matches_live_refs(store)


def test_invalid_retained_root_aborts_append_before_publication(sqlite_store):
    store = sqlite_store
    _commit_segmented(store, count=300)
    _commit_segmented(
        store,
        run_id="broken-run",
        count=130,
        prefix="broken",
        content_prefix="broken",
    )
    broken_head = _manifest_head(store, run_id="broken-run")
    store._conn.execute(
        """\
DELETE FROM checkpoint_manifest_segments
WHERE namespace=? AND thread_id=? AND segment_ref=?
""",
        (NS, THREAD, broken_head),
    )
    store._conn.commit()

    before_state = json.dumps(_raw_state(store))
    before_payloads = _stored_payload_refs(store)
    before_segments = _stored_segment_refs(store)
    before_events = _count_events(store)
    before_commits = _count_commits(store)
    tail = store.load_tail(NS, THREAD, RUN, limit=0)

    with pytest.raises(ValueError, match="missing"):
        store.append_state(
            NS,
            THREAD,
            RUN,
            [_new_item()],
            base=tail.base,
            event={"event_type": "must_not_publish"},
        )

    assert json.dumps(_raw_state(store)) == before_state
    assert _stored_payload_refs(store) == before_payloads
    assert _stored_segment_refs(store) == before_segments
    assert _count_payloads(store) == len(before_payloads)
    assert _count_segments(store) == len(before_segments)
    assert _count_events(store) == before_events
    assert _count_commits(store) == before_commits
    assert store.load_events(NS, THREAD, RUN) == []
    assert store.load_tail(NS, THREAD, RUN, limit=0).base == tail.base


def test_second_delete_batch_failure_rolls_back_append_and_earlier_batches(
    sqlite_store,
):
    store = sqlite_store
    initial_items = _commit_segmented(store, count=300)
    orphan_payloads = {f"item_orphan_rollback_{index:04d}" for index in range(1201)}
    _insert_orphan_payloads(store, orphan_payloads)
    before_state = json.dumps(_raw_state(store))
    before_payloads = _stored_payload_refs(store)
    before_segments = _stored_segment_refs(store)
    before_events = _count_events(store)
    before_commits = _count_commits(store)
    tail = store.load_tail(NS, THREAD, RUN, limit=0)

    with _recording_connection(store, fail_on_delete=2) as proxy:
        with pytest.raises(
            _InjectedDeleteError,
            match="injected failure in checkpoint payload delete batch",
        ):
            store.append_state(
                NS,
                THREAD,
                RUN,
                [_new_item()],
                base=tail.base,
                event={"event_type": "must_not_publish"},
            )

    assert proxy.delete_calls == 2
    assert json.dumps(_raw_state(store)) == before_state
    assert _stored_payload_refs(store) == before_payloads
    assert _stored_segment_refs(store) == before_segments
    assert _count_payloads(store) == len(before_payloads)
    assert _count_segments(store) == len(before_segments)
    assert _count_events(store) == before_events
    assert _count_commits(store) == before_commits
    assert store.load_state(NS, THREAD, RUN)["messages"]["items"] == initial_items
    assert store.load_events(NS, THREAD, RUN) == []
    assert store.load_tail(NS, THREAD, RUN, limit=0).base == tail.base


def test_late_event_failure_rolls_back_gc_and_publication(sqlite_store):
    store = sqlite_store
    initial_items = _commit_segmented(store, count=300)
    _insert_orphan_payloads(store, ["item_late_event_dead"])
    _insert_orphan_segments(store, ["seg_late_event_dead"])
    store._conn.execute(_FAIL_EVENT_TRIGGER)
    store._conn.commit()

    before_state = json.dumps(_raw_state(store))
    before_payloads = _stored_payload_refs(store)
    before_segments = _stored_segment_refs(store)
    before_commits = _count_commits(store)
    tail = store.load_tail(NS, THREAD, RUN, limit=0)

    with pytest.raises(sqlite3.IntegrityError, match="event insert blocked"):
        store.append_state(
            NS,
            THREAD,
            RUN,
            [_new_item()],
            base=tail.base,
            event={"event_type": "must_not_publish"},
        )

    assert json.dumps(_raw_state(store)) == before_state
    assert _stored_payload_refs(store) == before_payloads
    assert _stored_segment_refs(store) == before_segments
    assert "item_late_event_dead" in before_payloads
    assert "seg_late_event_dead" in before_segments
    assert _count_payloads(store) == len(before_payloads)
    assert _count_segments(store) == len(before_segments)
    assert _count_events(store) == 0
    assert _count_commits(store) == before_commits
    assert store.load_state(NS, THREAD, RUN)["messages"]["items"] == initial_items
    assert store.load_events(NS, THREAD, RUN) == []
    assert store.load_tail(NS, THREAD, RUN, limit=0).base == tail.base


def test_empty_live_set_uses_scoped_delete_all_without_touching_other_scopes(
    sqlite_store,
):
    store = sqlite_store
    _commit_segmented(store, count=300)
    _commit_segmented(
        store,
        namespace=OTHER_NS,
        thread_id=THREAD,
        run_id="survivor-run",
        count=130,
        prefix="survivor",
    )
    _insert_orphan_payloads(
        store, ["item_survivor_dead"], namespace=OTHER_NS, thread_id=THREAD
    )
    _insert_orphan_segments(
        store, ["seg_survivor_dead"], namespace=OTHER_NS, thread_id=THREAD
    )
    survivor_payloads = _stored_payload_refs(store, OTHER_NS, THREAD)
    survivor_segments = _stored_segment_refs(store, OTHER_NS, THREAD)

    with _recording_connection(store) as proxy:
        commit = store.commit_state(
            NS,
            THREAD,
            RUN,
            _state(NS, THREAD, []),
            expected_revision=1,
        )

    assert commit.revision == 2
    assert store.load_state(NS, THREAD, RUN)["messages"]["items"] == []
    payload_deletes = _deletes(proxy, "checkpoint_message_items")
    segment_deletes = _deletes(proxy, "checkpoint_manifest_segments")
    assert len(payload_deletes) == 1
    assert len(segment_deletes) == 1
    for sql, parameters in [*payload_deletes, *segment_deletes]:
        assert " IN (" not in sql
        assert parameters == (NS, THREAD)
    assert _count_payloads(store, NS, THREAD) == 0
    assert _count_segments(store, NS, THREAD) == 0
    assert _stored_payload_refs(store, OTHER_NS, THREAD) == survivor_payloads
    assert _stored_segment_refs(store, OTHER_NS, THREAD) == survivor_segments
    assert "item_survivor_dead" in survivor_payloads
    assert "seg_survivor_dead" in survivor_segments


def test_memory_and_sqlite_keep_identical_replacement_gc_archive(tmp_path):
    sqlite_store = SQLiteCheckpointStore(str(tmp_path / "parity.sqlite3"))
    memory_store = InMemoryCheckpointStore()
    stores = [sqlite_store, memory_store]
    try:
        for store in stores:
            aliased = _items(300)
            aliased[1]["content"] = aliased[0]["content"]
            _commit_segmented(store, run_id=RUN, items=aliased)
            _commit_segmented(store, run_id="shared-run", items=aliased)
            if isinstance(store, SQLiteCheckpointStore):
                _insert_orphan_payloads(store, ["item_parity_dead"])
                _insert_orphan_segments(store, ["seg_parity_dead"])
            else:
                store._message_items.setdefault(NS, {}).setdefault(THREAD, {})[
                    "item_parity_dead"
                ] = b"dead"
                store._manifest_pages.setdefault(NS, {}).setdefault(THREAD, {})[
                    "seg_parity_dead"
                ] = b"dead"
            tail = store.load_tail(NS, THREAD, RUN, limit=0)
            store.append_state(
                NS,
                THREAD,
                RUN,
                [_new_item()],
                base=tail.base,
                event={"event_type": "parity_probe"},
            )

        sqlite_state = deepcopy(sqlite_store.load_state(NS, THREAD, RUN))
        memory_state = deepcopy(memory_store.load_state(NS, THREAD, RUN))
        sqlite_state["_checkpoint"]["stream_id"] = "<stream>"
        memory_state["_checkpoint"]["stream_id"] = "<stream>"
        assert memory_state == sqlite_state
        assert (
            memory_store.load_events(NS, THREAD, RUN)
            == sqlite_store.load_events(NS, THREAD, RUN)
            == [{"event_type": "parity_probe"}]
        )
        assert memory_store.load_tail(NS, THREAD, RUN).base.revision == (
            sqlite_store.load_tail(NS, THREAD, RUN).base.revision
        )
        assert (
            memory_store.load_tail(NS, THREAD, RUN).base.item_count
            == (sqlite_store.load_tail(NS, THREAD, RUN).base.item_count)
            == 301
        )
        assert _stored_payload_refs(memory_store) == _stored_payload_refs(sqlite_store)
        assert _stored_segment_refs(memory_store) == _stored_segment_refs(sqlite_store)
        assert _count_payloads(sqlite_store) == 300
        assert _count_segments(sqlite_store) == 4
        _assert_archive_matches_live_refs(sqlite_store)
        assert "item_parity_dead" not in _stored_payload_refs(memory_store)
        assert "seg_parity_dead" not in _stored_segment_refs(memory_store)
    finally:
        sqlite_store.close()
