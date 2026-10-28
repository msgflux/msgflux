"""Provider integration for suffix recovery and same-run incremental appends."""

import json
from copy import deepcopy

import pytest

from msgflux._private.chat_items import legacy_item_id
from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import (
    CheckpointConflictError,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from msgflux.data.stores._manifest import MANIFEST_PAGE_SIZE
from msgflux.data.stores.incremental import CheckpointBase
from msgflux.data.stores.providers import in_memory as in_memory_module
from msgflux.data.stores.providers import sqlite as sqlite_module

NAMESPACE = "incremental-tests"
THREAD = "thread"
RUN = "run"


@pytest.fixture(params=["in_memory", "sqlite"])
def store(request, tmp_path):
    instance = (
        InMemoryCheckpointStore()
        if request.param == "in_memory"
        else SQLiteCheckpointStore(str(tmp_path / "incremental.sqlite3"))
    )
    try:
        yield instance
    finally:
        close = getattr(instance, "close", None)
        if close is not None:
            close()


def _items(start, count, *, prefix="item"):
    return [
        {
            "role": "user",
            "content": f"{prefix}-{index}",
            "item_id": f"itm_{index}",
        }
        for index in range(start, start + count)
    ]


def _state(items, *, status="running", metadata=None):
    messages = ChatMessages(items, namespace=NAMESPACE, thread_id=THREAD)
    if metadata is not None:
        messages.metadata = deepcopy(metadata)
    return {"status": status, "messages": messages._to_state()}


def _new_item(item_id="itm_new", content="appended"):
    return {
        "role": "assistant",
        "content": content,
        "item_id": item_id,
    }


def _raw(store, run=RUN):
    if isinstance(store, SQLiteCheckpointStore):
        row = store._conn.execute(
            "SELECT state FROM checkpoints "
            "WHERE namespace=? AND thread_id=? AND run_id=?",
            (NAMESPACE, THREAD, run),
        ).fetchone()
        return json.loads(row[0])
    return deepcopy(store._data[NAMESPACE][THREAD][run]["state"])


def _write_raw(store, raw, run=RUN):
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoints SET state=? "
            "WHERE namespace=? AND thread_id=? AND run_id=?",
            (json.dumps(raw), NAMESPACE, THREAD, run),
        )
        store._conn.commit()
    else:
        store._data[NAMESPACE][THREAD][run]["state"] = raw


def _segments(store):
    if isinstance(store, SQLiteCheckpointStore):
        return dict(
            store._conn.execute(
                "SELECT segment_ref, data FROM checkpoint_manifest_segments "
                "WHERE namespace=? AND thread_id=?",
                (NAMESPACE, THREAD),
            )
        )
    return dict(store._manifest_pages.get(NAMESPACE, {}).get(THREAD, {}))


def _payloads(store):
    if isinstance(store, SQLiteCheckpointStore):
        return dict(
            store._conn.execute(
                "SELECT item_ref, item FROM checkpoint_message_items "
                "WHERE namespace=? AND thread_id=?",
                (NAMESPACE, THREAD),
            )
        )
    return dict(store._message_items.get(NAMESPACE, {}).get(THREAD, {}))


def _commit_count(store, run=RUN):
    if isinstance(store, SQLiteCheckpointStore):
        return store._conn.execute(
            "SELECT COUNT(*) FROM checkpoint_commits "
            "WHERE namespace=? AND thread_id=? AND run_id=?",
            (NAMESPACE, THREAD, run),
        ).fetchone()[0]
    return len(store._data[NAMESPACE][THREAD][run].get("commits", []))


def _commit_records(store, run=RUN):
    if isinstance(store, SQLiteCheckpointStore):
        return [
            json.loads(row[0])
            for row in store._conn.execute(
                "SELECT data FROM checkpoint_commits "
                "WHERE namespace=? AND thread_id=? AND run_id=? "
                "ORDER BY revision",
                (NAMESPACE, THREAD, run),
            )
        ]
    return [
        data
        for _stream, _revision, data in store._data[NAMESPACE][THREAD][run]["commits"]
    ]


def _corrupt_payload(store, item_ref):
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoint_message_items SET item=? "
            "WHERE namespace=? AND thread_id=? AND item_ref=?",
            ("{invalid payload", NAMESPACE, THREAD, item_ref),
        )
        store._conn.commit()
    else:
        store._message_items[NAMESPACE][THREAD][item_ref] = b"\xc1"


def _seed(store, count=5, *, status="running", metadata=None, run=RUN):
    store.commit_state(
        NAMESPACE,
        THREAD,
        run,
        _state(_items(0, count), status=status, metadata=metadata),
        expected_revision=0,
    )
    return store.load_tail(NAMESPACE, THREAD, run)


def test_suffix_window_absolute_start_and_metadata_are_partial(store):
    state = _state(_items(0, 5), metadata={"kind": "suffix"})
    store.commit_state(NAMESPACE, THREAD, RUN, state, expected_revision=0)

    expected = state["messages"]["items"]
    for limit in (0, 1, 2, 5, 6, 128):
        tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=limit)
        start = max(0, len(expected) - limit)
        assert tail is not None
        assert tail.start == start
        assert tail.base.item_count == len(expected)
        assert tail.base.revision == 1
        assert tail.base.namespace == NAMESPACE
        assert tail.base.thread_id == THREAD
        assert tail.base.run_id == RUN
        assert tail.state["messages"]["items"] == expected[start:]
        assert tail.state["messages"]["metadata"] == {"kind": "suffix"}
        assert tail.state["messages"]["thread_id"] == THREAD
        assert tail.state["messages"]["namespace"] == NAMESPACE
        assert tail.state["status"] == "running"
        assert "_messages" not in tail.state
        assert "items" not in tail.state.get("_messages", {})
        complete = store.load_state(NAMESPACE, THREAD, RUN)
        assert complete["messages"] == state["messages"]
        assert complete["status"] == state["status"]


def test_suffix_report_only_does_not_materialize_history(store, monkeypatch):
    store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        _state(_items(0, 300)),
        expected_revision=0,
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("tail recovery must not hydrate full history")

    monkeypatch.setattr(store, "_denormalize_state", fail)
    monkeypatch.setattr(store, "_stored_manifest_entries", fail)
    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=3)
    assert tail is not None
    assert tail.start == 297
    assert [item["item_id"] for item in tail.state["messages"]["items"]] == [
        "itm_297",
        "itm_298",
        "itm_299",
    ]


def test_segmented_suffix_reads_at_most_suffix_and_boundary_pages(store, monkeypatch):
    store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        _state(_items(0, MANIFEST_PAGE_SIZE * 3 + 5)),
        expected_revision=0,
    )
    reads = []
    original = store._read_manifest_segment

    def read_segment(namespace, thread_id, segment_ref):
        reads.append(segment_ref)
        return original(namespace, thread_id, segment_ref)

    monkeypatch.setattr(store, "_read_manifest_segment", read_segment)
    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=130)

    assert tail is not None
    assert tail.start == MANIFEST_PAGE_SIZE * 3 + 5 - 130
    assert len(reads) <= 3
    assert [item["item_id"] for item in tail.state["messages"]["items"]][:2] == [
        f"itm_{tail.start}",
        f"itm_{tail.start + 1}",
    ]


def test_tail_state_is_isolated_from_storage(store):
    store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        _state(_items(0, 3)),
        expected_revision=0,
    )
    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=2)
    before = store.load_state(NAMESPACE, THREAD, RUN)

    tail.state["messages"]["items"][0]["content"] = "mutated"
    tail.state["messages"]["metadata"]["leak"] = True
    tail.state["status"] = "mutated"

    assert store.load_state(NAMESPACE, THREAD, RUN) == before


def test_load_tail_limit_validation_and_missing_run(store):
    for bad in (-1, True, 1.5, "1", None):
        with pytest.raises(ValueError, match="limit"):
            store.load_tail(NAMESPACE, THREAD, "missing", limit=bad)
    assert store.load_tail(NAMESPACE, THREAD, "missing") is None
    assert store.load_tail("other", THREAD, RUN) is None


def test_append_returns_only_new_occurrences_and_preserves_opaque_payloads(store):
    opaque = {
        "type": "anthropic_native_block",
        "role": "assistant",
        "content": [{"type": "text", "text": "opaque"}],
        "provider_payload": {
            "signature": "sig-1",
            "blocks": [{"kind": "thinking", "values": [1, 2, 3]}],
        },
        "item_id": "itm_opaque",
    }
    state = _state([*_items(0, 2), opaque], metadata={"kind": "opaque"})
    store.commit_state(NAMESPACE, THREAD, RUN, state, expected_revision=0)
    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=2)

    appended = store.append_state(
        NAMESPACE,
        THREAD,
        RUN,
        [_new_item("itm_reply", "reply")],
        base=tail.base,
        state_updates={"status": "completed"},
    )

    assert appended.start == 3
    assert appended.base.item_count == 4
    assert appended.base.revision == 2
    assert appended.state["messages"]["items"] == [_new_item("itm_reply", "reply")]
    assert appended.state["status"] == "completed"
    assert appended.state["messages"]["metadata"] == {"kind": "opaque"}

    complete = store.load_state(NAMESPACE, THREAD, RUN)
    assert complete["messages"]["items"] == [
        *state["messages"]["items"],
        _new_item("itm_reply", "reply"),
    ]
    assert complete["messages"]["items"][2]["provider_payload"] == {
        "signature": "sig-1",
        "blocks": [{"kind": "thinking", "values": [1, 2, 3]}],
    }


def test_inline_to_segment_threshold_and_changed_head_cleanup(store, monkeypatch):
    store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        _state(_items(0, MANIFEST_PAGE_SIZE - 1)),
        expected_revision=0,
    )
    assert len(_raw(store)["_messages"]["item_entries"]) == MANIFEST_PAGE_SIZE - 1
    assert _segments(store) == {}

    def fail(*_args, **_kwargs):
        raise AssertionError("incremental append must not normalize full history")

    if isinstance(store, SQLiteCheckpointStore):
        monkeypatch.setattr(store, "_normalize_state", fail)
    else:
        monkeypatch.setattr(store, "_stage_state", fail)

    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=1)
    store.append_state(NAMESPACE, THREAD, RUN, [_new_item("itm_127")], base=tail.base)
    assert len(_raw(store)["_messages"]["item_entries"]) == MANIFEST_PAGE_SIZE
    assert _segments(store) == {}

    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=1)
    store.append_state(NAMESPACE, THREAD, RUN, [_new_item("itm_128")], base=tail.base)
    raw = _raw(store)
    assert "manifest" in raw["_messages"]
    assert raw["_messages"]["manifest"]["count"] == MANIFEST_PAGE_SIZE + 1
    assert len(_segments(store)) == 2

    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=1)
    store.append_state(NAMESPACE, THREAD, RUN, [_new_item("itm_129")], base=tail.base)
    assert len(_segments(store)) == 2
    complete = store.load_state(NAMESPACE, THREAD, RUN)
    assert [item["item_id"] for item in complete["messages"]["items"]] == [
        f"itm_{index}" for index in range(MANIFEST_PAGE_SIZE + 2)
    ]


def test_segmented_append_reuses_pages_without_rebuilding(store, monkeypatch):
    store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        _state(_items(0, MANIFEST_PAGE_SIZE * 3 + 5)),
        expected_revision=0,
    )
    before_pages = _segments(store)
    before_payload_count = len(_payloads(store))

    def fail(*_args, **_kwargs):
        raise AssertionError("segmented append must not rebuild the manifest")

    monkeypatch.setattr(
        sqlite_module if isinstance(store, SQLiteCheckpointStore) else in_memory_module,
        "build_manifest",
        fail,
    )
    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=2)
    store.append_state(NAMESPACE, THREAD, RUN, [_new_item("itm_tail")], base=tail.base)

    after_pages = _segments(store)
    assert len(after_pages) == len(before_pages)
    assert set(before_pages) - set(after_pages)
    assert set(after_pages) - set(before_pages)
    assert len(_payloads(store)) == before_payload_count + 1
    complete = store.load_state(NAMESPACE, THREAD, RUN)
    assert complete["messages"]["items"][-1] == _new_item("itm_tail")


def test_append_controls_events_revisions_and_default_head(store):
    first = store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        _state(_items(0, 1), metadata={"phase": 1}),
        expected_revision=0,
        branch_id="root",
        head_item_id="itm_0",
        extension_state={"budget": {"remaining": 2}},
    )
    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=1)

    appending = store.append_state(
        NAMESPACE,
        THREAD,
        RUN,
        [_new_item("itm_a")],
        base=tail.base,
        state_updates={"status": "paused"},
        message_updates={"metadata": {"phase": 2}},
        event={"event_type": "appended_a"},
        branch_id="review",
    )

    envelope = appending.state["_checkpoint"]
    assert envelope["revision"] == first.revision + 1
    assert envelope["stream_id"] == first.state["_checkpoint"]["stream_id"]
    assert envelope["branch_id"] == "review"
    assert envelope["head_item_id"] == "itm_a"
    assert envelope["extensions"] == {"budget": {"remaining": 2}}
    assert appending.state["messages"]["metadata"] == {"phase": 2}
    assert appending.state["messages"]["thread_id"] == THREAD
    assert store.load_events(NAMESPACE, THREAD, RUN) == [{"event_type": "appended_a"}]
    assert _commit_count(store) == 2
    assert _commit_records(store)[-1] == {"event_type": "appended_a"}

    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=1)
    control_only = store.append_state(
        NAMESPACE,
        THREAD,
        RUN,
        [],
        base=tail.base,
        state_updates={"status": "completed"},
        message_updates={"metadata": {"phase": 3}},
    )
    assert control_only.start == 2
    assert control_only.base.item_count == 2
    assert control_only.state["messages"]["items"] == []
    assert control_only.state["_checkpoint"]["head_item_id"] == "itm_a"
    assert control_only.state["_checkpoint"]["revision"] == 3
    assert control_only.state["status"] == "completed"
    assert control_only.state["messages"]["metadata"] == {"phase": 3}

    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=1)
    explicit = store.append_state(
        NAMESPACE,
        THREAD,
        RUN,
        [_new_item("itm_b")],
        base=tail.base,
        head_item_id="custom-head",
        extension_state={"budget": {"remaining": 1}},
    )
    assert explicit.state["_checkpoint"]["head_item_id"] == "custom-head"
    assert explicit.state["_checkpoint"]["extensions"] == {"budget": {"remaining": 1}}
    assert store.load_state(NAMESPACE, THREAD, RUN)["messages"]["items"][-1] == (
        _new_item("itm_b")
    )


def test_invalid_control_updates_and_ids_roll_back(store):
    tail = _seed(store, 2)
    before_raw = _raw(store)
    before_payloads = _payloads(store)
    before_pages = _segments(store)
    before_events = store.load_events(NAMESPACE, THREAD, RUN)
    before_commits = _commit_count(store)

    cases = [
        ({"items": [_new_item("itm_x")]}, {"state_updates": {"messages": {}}}),
        ({"items": [_new_item("itm_x")]}, {"state_updates": {"_messages": {}}}),
        (
            {"items": [_new_item("itm_x")]},
            {"state_updates": {"_checkpoint": {}}},
        ),
        (
            {"items": [_new_item("itm_x")]},
            {"message_updates": {"items": []}},
        ),
        ({"items": [{"role": "user", "content": "no id"}]}, {}),
        (
            {
                "items": [
                    _new_item("itm_dup"),
                    _new_item("itm_dup", "second"),
                ]
            },
            {},
        ),
        ({"items": [_new_item("itm_0")]}, {}),
        ({"items": ["not-a-mapping"]}, {}),
    ]
    for kwargs, extra in cases:
        with pytest.raises(ValueError):
            store.append_state(
                NAMESPACE,
                THREAD,
                RUN,
                kwargs["items"],
                base=tail.base,
                event={"event_type": "should.not.persist"},
                **extra,
            )
        assert _raw(store) == before_raw
        assert _payloads(store) == before_payloads
        assert _segments(store) == before_pages
        assert store.load_events(NAMESPACE, THREAD, RUN) == before_events
        assert _commit_count(store) == before_commits


def test_new_payload_colliding_with_another_runs_corruption_rolls_back(store):
    seed_item = _new_item("itm_old", "old")
    colliding = _new_item("itm_shared", "shared payload")
    store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        _state([seed_item]),
        expected_revision=0,
    )
    store.commit_state(
        NAMESPACE,
        THREAD,
        "other",
        _state([colliding]),
        expected_revision=0,
    )
    other_item = store.load_state(NAMESPACE, THREAD, "other")["messages"]["items"][0]
    other_raw = _raw(store, "other")
    other_ref = other_raw["_messages"]["item_entries"][0]["item_ref"]
    _corrupt_payload(store, other_ref)
    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=1)
    before_raw = _raw(store)
    before_events = store.load_events(NAMESPACE, THREAD, RUN)
    before_commits = _commit_count(store)

    with pytest.raises(ValueError, match=r"corrupted|could not be published"):
        store.append_state(
            NAMESPACE,
            THREAD,
            RUN,
            [dict(other_item, item_id="itm_new")],
            base=tail.base,
            event={"event_type": "should.not.persist"},
        )

    assert _raw(store) == before_raw
    assert store.load_events(NAMESPACE, THREAD, RUN) == before_events
    assert _commit_count(store) == before_commits


def test_base_scope_store_staleness_and_tampering_rejections(store, tmp_path):
    tail = _seed(store, 2)

    for kwargs in (
        {"namespace": "other", "thread_id": THREAD, "run_id": RUN},
        {"namespace": NAMESPACE, "thread_id": "other", "run_id": RUN},
        {"namespace": NAMESPACE, "thread_id": THREAD, "run_id": "other"},
    ):
        with pytest.raises(CheckpointConflictError):
            store.append_state(
                kwargs["namespace"],
                kwargs["thread_id"],
                kwargs["run_id"],
                [_new_item()],
                base=tail.base,
            )

    with pytest.raises(CheckpointConflictError):
        store.append_state(NAMESPACE, THREAD, "missing", [_new_item()], base=tail.base)

    other_store = (
        InMemoryCheckpointStore()
        if isinstance(store, InMemoryCheckpointStore)
        else SQLiteCheckpointStore(str(tmp_path / "other.sqlite3"))
    )
    try:
        with pytest.raises(CheckpointConflictError, match="different checkpoint store"):
            other_store.append_state(
                NAMESPACE, THREAD, RUN, [_new_item()], base=tail.base
            )
    finally:
        close = getattr(other_store, "close", None)
        if close is not None:
            close()

    class BaseSubclass(CheckpointBase):
        pass

    fake = BaseSubclass(
        tail.base.routing_id,
        tail.base.namespace,
        tail.base.thread_id,
        tail.base.run_id,
        tail.base.revision,
        tail.base.item_count,
        tail.base.digest,
    )
    with pytest.raises(ValueError, match="CheckpointBase"):
        store.append_state(NAMESPACE, THREAD, RUN, [_new_item()], base=fake)

    tampered_revision = base_with(tail.base, revision=True)
    with pytest.raises(ValueError):
        store.append_state(
            NAMESPACE, THREAD, RUN, [_new_item()], base=tampered_revision
        )

    tampered_digest = base_with(tail.base, digest="0" * 64)
    with pytest.raises(CheckpointConflictError):
        store.append_state(NAMESPACE, THREAD, RUN, [_new_item()], base=tampered_digest)

    stale = tail.base
    store.append_state(NAMESPACE, THREAD, RUN, [_new_item()], base=stale)
    with pytest.raises(CheckpointConflictError):
        store.append_state(NAMESPACE, THREAD, RUN, [_new_item("itm_stale")], base=stale)


def base_with(base, **changes):
    clone = object.__new__(CheckpointBase)
    for field in (
        "routing_id",
        "namespace",
        "thread_id",
        "run_id",
        "revision",
        "item_count",
        "digest",
    ):
        object.__setattr__(
            clone,
            field,
            changes.get(field, getattr(base, field)),
        )
    return clone


def test_same_revision_save_invalidates_tail_base(store):
    tail = _seed(store, 3)
    replaced = deepcopy(store.load_state(NAMESPACE, THREAD, RUN))
    replaced["status"] = "paused"
    store.save_state(NAMESPACE, THREAD, RUN, replaced)

    with pytest.raises(CheckpointConflictError):
        store.append_state(NAMESPACE, THREAD, RUN, [_new_item()], base=tail.base)
    assert store.load_state(NAMESPACE, THREAD, RUN) == replaced


def test_alias_pages_survive_append_and_orphans_do_not_grow(store):
    state = _state(_items(0, MANIFEST_PAGE_SIZE * 2 + 44))
    store.commit_state(NAMESPACE, THREAD, RUN, state, expected_revision=0)
    store.save_state(NAMESPACE, THREAD, "alias", state)
    initial_pages = set(_segments(store))
    assert len(initial_pages) == 3

    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=1)
    store.append_state(NAMESPACE, THREAD, RUN, [_new_item("itm_300")], base=tail.base)
    after_first = set(_segments(store))
    assert len(after_first) == 4
    assert initial_pages <= after_first

    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=1)
    store.append_state(NAMESPACE, THREAD, RUN, [_new_item("itm_301")], base=tail.base)
    after_second = set(_segments(store))
    assert len(after_second) == 4
    assert initial_pages <= after_second

    assert store.load_state(NAMESPACE, THREAD, "alias") == state
    assert store.load_state(NAMESPACE, THREAD, RUN)["messages"]["items"][-1] == (
        _new_item("itm_301")
    )

    assert store.delete_run(NAMESPACE, THREAD, "alias") is True
    assert len(set(_segments(store))) == 3
    assert store.load_state(NAMESPACE, THREAD, RUN)["messages"]["items"][-1] == (
        _new_item("itm_301")
    )


def test_legacy_inline_idless_prefix_migrates_deterministic_ids(store):
    store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        _state(_items(0, 2)),
        expected_revision=0,
    )
    raw = _raw(store)
    for entry in raw["_messages"]["item_entries"]:
        entry.pop("item_id")
    _write_raw(store, raw)
    before = store.load_state(NAMESPACE, THREAD, RUN)
    assert all("item_id" not in item for item in before["messages"]["items"])

    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=2)
    store.append_state(NAMESPACE, THREAD, RUN, [_new_item("itm_2")], base=tail.base)

    complete = store.load_state(NAMESPACE, THREAD, RUN)
    expected_ids = [
        legacy_item_id(
            namespace=NAMESPACE,
            thread_id=THREAD,
            index=index,
            item=before["messages"]["items"][index],
        )
        for index in range(2)
    ]
    assert [item["item_id"] for item in complete["messages"]["items"]] == [
        *expected_ids,
        "itm_2",
    ]


def test_sqlite_restart_and_independent_writer_cas(tmp_path):
    path = str(tmp_path / "restart.sqlite3")
    first = SQLiteCheckpointStore(path)
    first.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        _state(_items(0, MANIFEST_PAGE_SIZE + 2)),
        expected_revision=0,
    )
    tail = first.load_tail(NAMESPACE, THREAD, RUN, limit=2)
    first.close()

    second = SQLiteCheckpointStore(path)
    try:
        restored_tail = second.load_tail(NAMESPACE, THREAD, RUN, limit=2)
        assert restored_tail == tail
        appended = second.append_state(
            NAMESPACE,
            THREAD,
            RUN,
            [_new_item("itm_restart")],
            base=restored_tail.base,
        )
        assert appended.base.revision == 2
    finally:
        second.close()

    writer_a = SQLiteCheckpointStore(path)
    writer_b = SQLiteCheckpointStore(path)
    try:
        base = writer_a.load_tail(NAMESPACE, THREAD, RUN, limit=1).base
        writer_b.append_state(NAMESPACE, THREAD, RUN, [_new_item("itm_b")], base=base)
        with pytest.raises(CheckpointConflictError):
            writer_a.append_state(
                NAMESPACE, THREAD, RUN, [_new_item("itm_a")], base=base
            )
        fresh = writer_a.load_tail(NAMESPACE, THREAD, RUN, limit=1)
        writer_a.append_state(
            NAMESPACE, THREAD, RUN, [_new_item("itm_a")], base=fresh.base
        )
        assert writer_a.load_state(NAMESPACE, THREAD, RUN)["messages"]["items"][
            -1
        ] == _new_item("itm_a")
    finally:
        writer_a.close()
        writer_b.close()


def test_control_only_append_does_not_rewrite_segment_pages(store):
    store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        _state(_items(0, MANIFEST_PAGE_SIZE + 7)),
        expected_revision=0,
    )
    before_pages = _segments(store)
    tail = store.load_tail(NAMESPACE, THREAD, RUN, limit=1)
    appended = store.append_state(
        NAMESPACE,
        THREAD,
        RUN,
        [],
        base=tail.base,
        state_updates={"status": "completed"},
    )
    assert appended.start == MANIFEST_PAGE_SIZE + 7
    assert appended.state["messages"]["items"] == []
    assert appended.state["status"] == "completed"
    assert _segments(store) == before_pages
