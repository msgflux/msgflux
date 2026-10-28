"""Provider integration for content-addressed checkpoint manifest segments."""

import json
from copy import deepcopy

import pytest

from msgflux._private.chat_items import split_item_occurrence
from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.data.stores._manifest import MANIFEST_PAGE_SIZE
from msgflux.data.stores.providers import in_memory as in_memory_module

NAMESPACE = "agent:manifest"
THREAD = "thread"
OTHER_NAMESPACE = "agent:other"


@pytest.fixture(params=["in_memory", "sqlite"])
def store(request, tmp_path):
    instance = (
        InMemoryCheckpointStore()
        if request.param == "in_memory"
        else SQLiteCheckpointStore(str(tmp_path / "manifest.sqlite3"))
    )
    try:
        yield instance
    finally:
        close = getattr(instance, "close", None)
        if close is not None:
            close()


def _items(count, prefix="item", *, start=0):
    return [
        {"role": "user", "content": f"{prefix}-{index}"}
        for index in range(start, start + count)
    ]


def _message_state(items, *, namespace=NAMESPACE, thread_id=THREAD):
    messages = ChatMessages(items, namespace=namespace, thread_id=thread_id)
    messages.metadata = {"kind": "manifest-test"}
    return messages._to_state()


def _state(count, prefix="item", *, start=0, namespace=NAMESPACE, thread_id=THREAD):
    return {
        "status": "completed",
        "messages": _message_state(
            _items(count, prefix, start=start),
            namespace=namespace,
            thread_id=thread_id,
        ),
    }


def _append_state(
    base_state, count, prefix="tail", *, namespace=NAMESPACE, thread_id=THREAD
):
    return _state_from_prefix(
        base_state,
        count,
        prefix,
        namespace=namespace,
        thread_id=thread_id,
    )


def _state_from_prefix(
    base_state, count=0, prefix="tail", *, namespace=NAMESPACE, thread_id=THREAD
):
    items = list(base_state["messages"]["items"]) + _items(count, prefix)
    return {
        "status": "completed",
        "messages": _message_state(
            items,
            namespace=namespace,
            thread_id=thread_id,
        ),
    }


def _raw_state(store, run, *, namespace=NAMESPACE, thread_id=THREAD):
    if isinstance(store, SQLiteCheckpointStore):
        row = store._conn.execute(
            "SELECT state FROM checkpoints "
            "WHERE namespace=? AND thread_id=? AND run_id=?",
            (namespace, thread_id, run),
        ).fetchone()
        return json.loads(row[0]), row[0]
    state = store._data[namespace][thread_id][run]["state"]
    return state, json.dumps(state, ensure_ascii=False, default=str)


def _write_raw_state(store, run, state, *, namespace=NAMESPACE, thread_id=THREAD):
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoints SET state=? "
            "WHERE namespace=? AND thread_id=? AND run_id=?",
            (json.dumps(state), namespace, thread_id, run),
        )
        store._conn.commit()
    else:
        store._data[namespace][thread_id][run]["state"] = state


def _set_updated(store, run, updated_at, *, namespace=NAMESPACE, thread_id=THREAD):
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoints SET updated_at=? "
            "WHERE namespace=? AND thread_id=? AND run_id=?",
            (updated_at, namespace, thread_id, run),
        )
        store._conn.commit()
    else:
        store._data[namespace][thread_id][run]["updated_at"] = updated_at


def _segments(store, *, namespace=NAMESPACE, thread_id=THREAD):
    if isinstance(store, SQLiteCheckpointStore):
        rows = store._conn.execute(
            "SELECT segment_ref, data FROM checkpoint_manifest_segments "
            "WHERE namespace=? AND thread_id=?",
            (namespace, thread_id),
        ).fetchall()
        return {row[0]: bytes(row[1]) for row in rows}
    return dict(store._manifest_pages.get(namespace, {}).get(thread_id, {}))


def _write_segment(store, ref, data, *, namespace=NAMESPACE, thread_id=THREAD):
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoint_manifest_segments SET data=? "
            "WHERE namespace=? AND thread_id=? AND segment_ref=?",
            (data, namespace, thread_id, ref),
        )
        store._conn.commit()
    else:
        store._manifest_pages[namespace][thread_id][ref] = data


def _delete_segment(store, ref, *, namespace=NAMESPACE, thread_id=THREAD):
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "DELETE FROM checkpoint_manifest_segments "
            "WHERE namespace=? AND thread_id=? AND segment_ref=?",
            (namespace, thread_id, ref),
        )
        store._conn.commit()
    else:
        del store._manifest_pages[namespace][thread_id][ref]


def _payloads(store, *, namespace=NAMESPACE, thread_id=THREAD):
    if isinstance(store, SQLiteCheckpointStore):
        rows = store._conn.execute(
            "SELECT item_ref, item FROM checkpoint_message_items "
            "WHERE namespace=? AND thread_id=?",
            (namespace, thread_id),
        ).fetchall()
        return {row[0]: row[1] for row in rows}
    return dict(store._message_items.get(namespace, {}).get(thread_id, {}))


def _corrupt_payload(store, *, namespace=NAMESPACE, thread_id=THREAD):
    ref = next(iter(_payloads(store, namespace=namespace, thread_id=thread_id)))
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoint_message_items SET item=? "
            "WHERE namespace=? AND thread_id=? AND item_ref=?",
            ("{invalid json", namespace, thread_id, ref),
        )
        store._conn.commit()
    else:
        store._message_items[namespace][thread_id][ref] = b"\xc1"
    return ref


def _commit_count(store, run, *, namespace=NAMESPACE, thread_id=THREAD):
    if isinstance(store, SQLiteCheckpointStore):
        return store._conn.execute(
            "SELECT COUNT(*) FROM checkpoint_commits "
            "WHERE namespace=? AND thread_id=? AND run_id=?",
            (namespace, thread_id, run),
        ).fetchone()[0]
    return len(store._data[namespace][thread_id][run].get("commits", []))


def _inline_state_bytes(store, run, *, namespace=NAMESPACE, thread_id=THREAD):
    raw, _ = _raw_state(store, run, namespace=namespace, thread_id=thread_id)
    loaded = store.load_state(namespace, thread_id, run)
    entries = []
    for index, item in enumerate(loaded["messages"]["items"]):
        payload, entry = split_item_occurrence(
            namespace=namespace,
            thread_id=thread_id,
            index=index,
            item=item,
        )
        entry["item_ref"] = store._item_ref(payload)
        entries.append(entry)
    inline = {key: value for key, value in raw.items() if key != "_messages"}
    inline["_messages"] = {
        "state": raw["_messages"]["state"],
        "item_entries": entries,
    }
    return len(json.dumps(inline, ensure_ascii=False, default=str))


def _write_snapshot(
    store,
    method,
    run_id,
    state,
    *,
    revision=0,
    namespace=NAMESPACE,
    thread_id=THREAD,
):
    if method == "save_state":
        store.save_state(namespace, thread_id, run_id, state)
    elif method == "save_with_event":
        store.save_with_event(
            namespace,
            thread_id,
            run_id,
            state,
            {"event_type": "snapshot", "revision": revision},
        )
    else:
        store.commit_state(
            namespace,
            thread_id,
            run_id,
            state,
            expected_revision=revision,
            event={"event_type": "snapshot", "revision": revision},
        )


@pytest.mark.parametrize("method", ["save_state", "save_with_event", "commit_state"])
def test_snapshot_replacements_collect_old_segment_tails(store, method):
    state = _state(200, "base")
    initial = deepcopy(state)
    _write_snapshot(store, method, "run", state, revision=0)
    assert len(_segments(store)) == 2

    for revision in range(1, 41):
        state = _append_state(state, 4, f"tail-{revision}")
        input_copy = deepcopy(state)
        _write_snapshot(store, method, "run", state, revision=revision)
        assert state == input_copy
        count = 200 + 4 * revision
        assert len(_segments(store)) == (count + MANIFEST_PAGE_SIZE - 1) // (
            MANIFEST_PAGE_SIZE
        )
        assert (
            store.load_state(NAMESPACE, THREAD, "run")["messages"] == state["messages"]
        )

    assert initial != state


def test_replacement_keeps_pages_referenced_by_retained_alias(store):
    first = _state(200, "alias")
    store.save_state(NAMESPACE, THREAD, "run_1", first)
    store.save_state(
        NAMESPACE,
        THREAD,
        "run_2",
        _state_from_prefix(first),
    )
    assert len(_segments(store)) == 2

    appended = _append_state(first, 40, "replacement")
    store.save_state(NAMESPACE, THREAD, "run_1", appended)

    # run_2 still points at the old partial tail; run_1 has a new tail.
    assert len(_segments(store)) == 3
    assert (
        store.load_state(NAMESPACE, THREAD, "run_2")["messages"]
        == _state_from_prefix(first)["messages"]
    )
    assert (
        store.load_state(NAMESPACE, THREAD, "run_1")["messages"] == appended["messages"]
    )

    assert store.delete_run(NAMESPACE, THREAD, "run_2") is True
    assert len(_segments(store)) == 2
    assert (
        store.load_state(NAMESPACE, THREAD, "run_1")["messages"] == appended["messages"]
    )


@pytest.mark.parametrize("replacement", ["inline", "metadata"])
def test_segmented_replacement_to_inline_or_metadata_releases_pages(store, replacement):
    store.save_state(NAMESPACE, THREAD, "run", _state(300, "segmented"))
    assert len(_segments(store)) == 3

    if replacement == "inline":
        state = _state(2, "small")
        store.save_state(NAMESPACE, THREAD, "run", state)
        assert (
            store.load_state(NAMESPACE, THREAD, "run")["messages"] == state["messages"]
        )
    else:
        state = {"status": "completed"}
        store.save_state(NAMESPACE, THREAD, "run", state)
        assert store.load_state(NAMESPACE, THREAD, "run") == state

    assert _segments(store) == {}


@pytest.mark.parametrize("method", ["save_state", "save_with_event", "commit_state"])
def test_snapshot_replacement_gc_failure_rolls_back(store, method):
    target = _state(300, "target")
    survivor = _state(300, "survivor")
    store.save_state(NAMESPACE, THREAD, "target", target)
    store.save_state(NAMESPACE, THREAD, "survivor", survivor)
    survivor_head = _raw_state(store, "survivor")[0]["_messages"]["manifest"]["head"]
    _delete_segment(store, survivor_head)

    before_target = deepcopy(_raw_state(store, "target")[0])
    before_survivor = deepcopy(_raw_state(store, "survivor")[0])
    before_pages = _segments(store)
    before_payloads = _payloads(store)
    before_events = store.load_events(NAMESPACE, THREAD, "target")
    before_commits = _commit_count(store, "target")

    with pytest.raises(ValueError):
        _write_snapshot(
            store,
            method,
            "target",
            _state(300, "replacement"),
            revision=0,
        )

    assert _raw_state(store, "target")[0] == before_target
    assert _raw_state(store, "survivor")[0] == before_survivor
    assert _segments(store) == before_pages
    assert _payloads(store) == before_payloads
    assert store.load_events(NAMESPACE, THREAD, "target") == before_events
    assert _commit_count(store, "target") == before_commits


def test_fork_replacement_collects_unreachable_target_pages(store):
    source = _state(300, "source")
    store.save_state(NAMESPACE, THREAD, "source", source)
    store.save_state(
        NAMESPACE,
        "target-thread",
        "target",
        _state(200, "old-target"),
    )
    assert len(_segments(store, thread_id="target-thread")) == 2

    forked = store.fork_run(
        NAMESPACE,
        THREAD,
        "source",
        target_thread_id="target-thread",
        target_run_id="target",
    )

    expected_messages = dict(source["messages"])
    expected_messages["thread_id"] = "target-thread"
    assert forked["messages"] == expected_messages
    assert len(_segments(store, thread_id="target-thread")) == 3


def test_unchanged_manifest_head_replacement_skips_segment_gc(store, monkeypatch):
    state = _state(300, "same-head")
    store.save_state(NAMESPACE, THREAD, "run", state)

    def fail(*args, **kwargs):
        raise AssertionError("GC must not run when the manifest head is unchanged")

    if isinstance(store, SQLiteCheckpointStore):
        monkeypatch.setattr(store, "_cleanup_orphaned_items", fail)
    else:
        monkeypatch.setattr(store, "_collect_live_refs", fail)

    store.save_state(NAMESPACE, THREAD, "run", state)
    assert store.load_state(NAMESPACE, THREAD, "run")["messages"] == state["messages"]


def test_small_manifest_stays_inline_and_large_manifest_uses_segments(store):
    small = _state(MANIFEST_PAGE_SIZE, "small")
    store.save_state(NAMESPACE, THREAD, "small", small)
    raw_small, _ = _raw_state(store, "small")
    assert isinstance(raw_small["_messages"]["item_entries"], list)
    assert "manifest" not in raw_small["_messages"]
    assert _segments(store) == {}

    large = _state(MANIFEST_PAGE_SIZE + 1, "large")
    store.save_state(NAMESPACE, THREAD, "large", large)
    raw_large, _ = _raw_state(store, "large")
    assert "item_entries" not in raw_large["_messages"]
    manifest = raw_large["_messages"]["manifest"]
    assert manifest["format"] == "segments_v1"
    assert manifest["count"] == MANIFEST_PAGE_SIZE + 1
    assert len(_segments(store)) == 2

    assert store.load_state(NAMESPACE, THREAD, "small") == small
    assert store.load_state(NAMESPACE, THREAD, "large") == large


def test_append_reuses_prefix_pages_and_round_trips(store):
    first = _state(200, "prefix")
    second = _append_state(first, 1, "tail")
    store.save_state(NAMESPACE, THREAD, "run_1", first)
    pages_after_first = _segments(store)
    store.save_state(NAMESPACE, THREAD, "run_2", second)
    pages_after_second = _segments(store)

    assert len(pages_after_first) == 2
    # The first full page is shared; only the partial tail page is replaced.
    assert len(pages_after_second) == 3
    assert set(pages_after_first) <= set(pages_after_second)
    assert store.load_state(NAMESPACE, THREAD, "run_1")["messages"] == first["messages"]
    assert (
        store.load_state(NAMESPACE, THREAD, "run_2")["messages"] == second["messages"]
    )


def test_legacy_inline_and_segmented_runs_coexist(store):
    legacy = _state(3, "legacy")
    segmented = _state(300, "segmented")
    store.save_state(NAMESPACE, THREAD, "legacy", legacy)
    store.save_state(NAMESPACE, THREAD, "segmented", segmented)

    assert store.load_state(NAMESPACE, THREAD, "legacy") == legacy
    assert store.load_state(NAMESPACE, THREAD, "segmented") == segmented
    assert store.delete_run(NAMESPACE, THREAD, "segmented") is True
    assert _segments(store) == {}
    assert store.load_state(NAMESPACE, THREAD, "legacy") == legacy


def test_gc_retains_pages_shared_by_remaining_runs(store):
    first = _state(300, "shared")
    second = _state_from_prefix(first, 0)
    store.save_state(NAMESPACE, THREAD, "run_1", first)
    store.save_state(NAMESPACE, THREAD, "run_2", second)
    assert len(_segments(store)) == 3

    assert store.delete_run(NAMESPACE, THREAD, "run_1") is True
    assert len(_segments(store)) == 3
    assert (
        store.load_state(NAMESPACE, THREAD, "run_2")["messages"] == second["messages"]
    )

    assert store.delete_run(NAMESPACE, THREAD, "run_2") is True
    assert _segments(store) == {}
    assert _payloads(store) == {}


def test_divergent_histories_do_not_delete_each_other(store):
    first = _state(300, "base")
    prefix = {"messages": {"items": first["messages"]["items"][:200]}}
    second = _state_from_prefix(prefix, 100, "divergent")
    store.save_state(NAMESPACE, THREAD, "run_1", first)
    store.save_state(NAMESPACE, THREAD, "run_2", second)
    # One shared first page; each run owns a distinct second and tail page.
    assert len(_segments(store)) == 5

    assert store.delete_run(NAMESPACE, THREAD, "run_1") is True
    assert (
        store.load_state(NAMESPACE, THREAD, "run_2")["messages"] == second["messages"]
    )
    assert len(_segments(store)) == 3

    assert store.delete_run(NAMESPACE, THREAD, "run_2") is True
    assert _segments(store) == {}


def test_fork_rebuilds_segments_in_the_target_thread(store):
    source = _state(300, "source")
    store.save_state(NAMESPACE, THREAD, "run", source)

    forked = store.fork_run(
        NAMESPACE,
        THREAD,
        "run",
        target_thread_id="forked-thread",
        target_run_id="forked-run",
    )
    expected_messages = dict(source["messages"])
    expected_messages["thread_id"] = "forked-thread"
    assert forked["messages"] == expected_messages
    assert len(_segments(store, thread_id="forked-thread")) == 3
    assert len(_segments(store)) == 3

    assert store.delete_run(NAMESPACE, THREAD, "run") is True
    assert _segments(store) == {}
    assert (
        store.load_state(NAMESPACE, "forked-thread", "forked-run")["messages"]
        == expected_messages
    )


def test_clear_is_scoped_and_removes_unreachable_segments(store):
    store.save_state(NAMESPACE, THREAD, "run", _state(300, "one"))
    store.save_state(
        OTHER_NAMESPACE,
        THREAD,
        "run",
        _state(300, "two", namespace=OTHER_NAMESPACE),
    )
    assert len(_segments(store)) == 3
    assert len(_segments(store, namespace=OTHER_NAMESPACE)) == 3

    assert store.clear(NAMESPACE) == 1
    assert _segments(store) == {}
    assert len(_segments(store, namespace=OTHER_NAMESPACE)) == 3
    assert store.load_state(OTHER_NAMESPACE, THREAD, "run")["messages"]["items"]


def test_clear_rolls_back_across_namespaces_when_survivor_is_corrupt(store):
    # Create the valid namespace first so a non-transactional implementation
    # would partially clear it before reaching the corrupt survivor.
    for namespace, prefix in (
        (OTHER_NAMESPACE, "other"),
        (NAMESPACE, "primary"),
    ):
        store.save_state(
            namespace,
            THREAD,
            "keep",
            _state(300, f"{prefix}-keep", namespace=namespace),
        )
        store.save_state(
            namespace,
            THREAD,
            "delete",
            _state(1, f"{prefix}-delete", namespace=namespace),
        )
        _set_updated(store, "delete", 0.0, namespace=namespace)
        _set_updated(store, "keep", 1e15, namespace=namespace)

    before_delete = {
        namespace: deepcopy(store.load_state(namespace, THREAD, "delete"))
        for namespace in (OTHER_NAMESPACE, NAMESPACE)
    }
    before_keep = {
        namespace: deepcopy(_raw_state(store, "keep", namespace=namespace)[0])
        for namespace in (OTHER_NAMESPACE, NAMESPACE)
    }
    before_pages = {
        namespace: _segments(store, namespace=namespace)
        for namespace in (OTHER_NAMESPACE, NAMESPACE)
    }
    _delete_segment(
        store,
        sorted(before_pages[NAMESPACE])[-1],
        namespace=NAMESPACE,
    )
    corrupted_pages = _segments(store, namespace=NAMESPACE)

    with pytest.raises(ValueError):
        store.clear(older_than=1)

    for namespace in (OTHER_NAMESPACE, NAMESPACE):
        assert store.load_state(namespace, THREAD, "delete") == before_delete[namespace]
        assert (
            _raw_state(store, "keep", namespace=namespace)[0] == before_keep[namespace]
        )
        expected_pages = (
            corrupted_pages if namespace == NAMESPACE else before_pages[namespace]
        )
        assert _segments(store, namespace=namespace) == expected_pages


def test_in_memory_fork_replacement_preflights_corrupt_survivors():
    store = InMemoryCheckpointStore()
    source = _state(300, "source")
    store.save_state(NAMESPACE, THREAD, "source", source)
    store.save_state(
        NAMESPACE,
        "target-thread",
        "old-target",
        _state(2, "old"),
    )
    store.save_state(
        NAMESPACE,
        "target-thread",
        "corrupt",
        _state(300, "corrupt"),
    )
    _delete_segment(
        store,
        sorted(_segments(store, thread_id="target-thread"))[-1],
        thread_id="target-thread",
    )
    before_old = deepcopy(store.load_state(NAMESPACE, "target-thread", "old-target"))
    before_target_raw = deepcopy(
        _raw_state(store, "old-target", thread_id="target-thread")[0]
    )
    before_corrupt_raw = deepcopy(
        _raw_state(store, "corrupt", thread_id="target-thread")[0]
    )
    before_pages = _segments(store, thread_id="target-thread")
    before_payloads = _payloads(store, thread_id="target-thread")

    with pytest.raises(ValueError):
        store.fork_run(
            NAMESPACE,
            THREAD,
            "source",
            target_thread_id="target-thread",
            target_run_id="old-target",
        )

    assert store.load_state(NAMESPACE, "target-thread", "old-target") == before_old
    assert (
        _raw_state(store, "old-target", thread_id="target-thread")[0]
        == before_target_raw
    )
    assert (
        _raw_state(store, "corrupt", thread_id="target-thread")[0] == before_corrupt_raw
    )
    assert _segments(store, thread_id="target-thread") == before_pages
    assert _payloads(store, thread_id="target-thread") == before_payloads


def test_missing_corrupt_and_swapped_segments_are_rejected(store):
    state = _state(300, "corrupt")
    store.save_state(NAMESPACE, THREAD, "run", state)
    refs = sorted(_segments(store))

    _delete_segment(store, refs[-1])
    with pytest.raises(ValueError):
        store.load_state(NAMESPACE, THREAD, "run")


def test_corrupt_segment_bytes_are_rejected(store):
    state = _state(300, "corrupt")
    store.save_state(NAMESPACE, THREAD, "run", state)
    ref = sorted(_segments(store))[0]

    _write_segment(store, ref, b"{not-json")
    with pytest.raises(ValueError):
        store.load_state(NAMESPACE, THREAD, "run")


def test_segment_ref_digest_mismatch_is_rejected(store):
    state = _state(300, "digest")
    store.save_state(NAMESPACE, THREAD, "run", state)
    pages = _segments(store)
    first, second = sorted(pages)[:2]

    _write_segment(store, second, pages[first])
    with pytest.raises(ValueError):
        store.load_state(NAMESPACE, THREAD, "run")


def test_truncated_or_inconsistent_manifest_count_is_rejected(store):
    state = _state(300, "count")
    store.save_state(NAMESPACE, THREAD, "run", state)
    raw, _ = _raw_state(store, "run")
    raw["_messages"]["manifest"]["count"] += 1
    _write_raw_state(store, "run", raw)

    with pytest.raises(ValueError):
        store.load_state(NAMESPACE, THREAD, "run")


@pytest.mark.parametrize("delta", [1, -1])
def test_commit_rejects_corrupt_prior_manifest_before_publishing(store, delta):
    state = _state(300, "commit")
    store.save_state(NAMESPACE, THREAD, "run", state)
    before = store.commit_state(
        NAMESPACE,
        THREAD,
        "run",
        _state(300, "new"),
        expected_revision=0,
        event={"event_type": "first"},
    )
    raw, _ = _raw_state(store, "run")
    raw["_messages"]["manifest"]["count"] += delta
    _write_raw_state(store, "run", raw)
    before_state = _raw_state(store, "run")[0]
    events = store.load_events(NAMESPACE, THREAD, "run")
    commits = _commit_count(store, "run")

    with pytest.raises(ValueError):
        store.commit_state(
            NAMESPACE,
            THREAD,
            "run",
            _state(300, "replacement"),
            expected_revision=before.revision,
            event={"event_type": "should.not.persist"},
        )

    assert _raw_state(store, "run")[0] == before_state
    assert store.load_events(NAMESPACE, THREAD, "run") == events
    assert _commit_count(store, "run") == commits


def test_strict_commit_validates_every_prior_segment_payload(store):
    state = _state(300, "old")
    store.commit_state(NAMESPACE, THREAD, "run", state, expected_revision=0)
    _corrupt_payload(store)
    before_state = _raw_state(store, "run")[0]
    events = store.load_events(NAMESPACE, THREAD, "run")

    with pytest.raises(ValueError):
        store.commit_state(
            NAMESPACE,
            THREAD,
            "run",
            _state(300, "replacement"),
            expected_revision=1,
            event={"event_type": "should.not.persist"},
        )

    assert _raw_state(store, "run")[0] == before_state
    assert store.load_events(NAMESPACE, THREAD, "run") == events
    assert _commit_count(store, "run") == 1


def test_segmented_occurrences_keep_nested_payloads_isolated(store):
    item = {
        "type": "message",
        "role": "user",
        "content": [{"type": "text", "text": "original"}],
        "metadata": {"tags": ["original"]},
    }
    items = [item for _ in range(2)] + _items(MANIFEST_PAGE_SIZE - 1, "padding")
    messages = ChatMessages(items, namespace=NAMESPACE, thread_id=THREAD)
    state = {"status": "completed", "messages": messages._to_state()}
    store.save_state(NAMESPACE, THREAD, "run", state)

    first = store.load_state(NAMESPACE, THREAD, "run")
    second = store.load_state(NAMESPACE, THREAD, "run")
    first_items = first["messages"]["items"]
    assert first_items[0]["item_id"] != first_items[1]["item_id"]
    assert first_items[0]["content"] is not first_items[1]["content"]
    first_items[0]["content"][0]["text"] = "changed"
    first_items[0]["metadata"]["tags"].append("changed")

    assert first_items[1]["content"][0]["text"] == "original"
    assert second["messages"]["items"][0]["content"][0]["text"] == "original"
    assert (
        store.load_state(NAMESPACE, THREAD, "run")["messages"]["items"][0]["content"][
            0
        ]["text"]
        == "original"
    )


def test_cross_run_storage_counts_states_and_every_segment(store):
    base = _state(200, "base")
    store.save_state(NAMESPACE, THREAD, "run_0", base)
    runs = [base]
    for index in range(1, 5):
        runs.append(_append_state(runs[-1], 50, f"append-{index}"))
        store.save_state(NAMESPACE, THREAD, f"run_{index}", runs[-1])

    state_bytes = 0
    inline_bytes = 0
    for index in range(5):
        raw, state_text = _raw_state(store, f"run_{index}")
        state_bytes += len(state_text.encode("utf-8"))
        assert "item_entries" not in raw["_messages"]
        assert raw["_messages"]["manifest"]["count"] == 200 + 50 * index
        inline_bytes += _inline_state_bytes(store, f"run_{index}")
    segment_bytes = sum(len(data) for data in _segments(store).values())
    item_bytes = sum(
        len(payload.encode("utf-8") if isinstance(payload, str) else payload)
        for payload in _payloads(store).values()
    )

    # Three shared full pages plus one unique partial tail per run length.
    assert len(_segments(store)) == 8
    assert state_bytes < inline_bytes // 4
    assert state_bytes + segment_bytes < inline_bytes
    assert item_bytes > 0


def test_sqlite_commit_rolls_back_partially_inserted_segments(tmp_path):
    store = SQLiteCheckpointStore(str(tmp_path / "rollback.sqlite3"))
    try:
        store.commit_state(
            NAMESPACE, THREAD, "run", _state(130, "base"), expected_revision=0
        )
        before_state = _raw_state(store, "run")[0]
        existing = len(_segments(store))
        assert existing == 2
        store._conn.execute(
            "CREATE TRIGGER fail_segment "
            "BEFORE INSERT ON checkpoint_manifest_segments "
            "WHEN (SELECT COUNT(*) FROM checkpoint_manifest_segments "
            "WHERE namespace='agent:manifest' AND thread_id='thread') >= 3 "
            "BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
        store._conn.commit()

        with pytest.raises(Exception, match="injected"):
            store.commit_state(
                NAMESPACE,
                THREAD,
                "run",
                _state(300, "divergent"),
                expected_revision=1,
                event={"event_type": "should.not.persist"},
            )

        assert _raw_state(store, "run")[0] == before_state
        assert len(_segments(store)) == existing
        assert store.load_events(NAMESPACE, THREAD, "run") == []
        assert _commit_count(store, "run") == 1
    finally:
        store.close()


def test_in_memory_failed_publication_leaks_no_segments(monkeypatch):
    store = InMemoryCheckpointStore()
    original = _state(130, "original")
    store.save_state(NAMESPACE, THREAD, "run", original)
    before_pages = _segments(store)
    before_items = _payloads(store)

    def fail_after_one_page(entries, write_page):
        write_page("segment_failure", b"partial")
        raise ValueError("injected manifest failure")

    monkeypatch.setattr(in_memory_module, "build_manifest", fail_after_one_page)
    with pytest.raises(ValueError, match="injected"):
        store.save_state(NAMESPACE, THREAD, "new-run", _state(300, "new"))

    assert _segments(store) == before_pages
    assert _payloads(store) == before_items
    assert store.load_state(NAMESPACE, THREAD, "new-run") is None


def test_independent_sqlite_connections_see_committed_segments(tmp_path):
    path = str(tmp_path / "concurrent.sqlite3")
    first = SQLiteCheckpointStore(path)
    second = SQLiteCheckpointStore(path)
    try:
        commit = first.commit_state(
            NAMESPACE, THREAD, "run", _state(300, "first"), expected_revision=0
        )
        assert (
            second.load_state(NAMESPACE, THREAD, "run")["messages"]["items"][-1][
                "content"
            ]
            == "first-299"
        )

        resumed = second.commit_state(
            NAMESPACE,
            THREAD,
            "run",
            _state(300, "second"),
            expected_revision=commit.revision,
        )
        assert resumed.revision == 2
        assert (
            first.load_state(NAMESPACE, THREAD, "run")["messages"]["items"][-1][
                "content"
            ]
            == "second-299"
        )
    finally:
        first.close()
        second.close()


def test_legacy_inline_entries_still_validate_item_refs(store):
    state = _state(2, "legacy")
    store.save_state(NAMESPACE, THREAD, "run", state)
    raw, _ = _raw_state(store, "run")
    raw["_messages"]["item_entries"][0]["item_ref"] = "item_missing"
    _write_raw_state(store, "run", raw)

    with pytest.raises(ValueError, match="missing or corrupted"):
        store.load_state(NAMESPACE, THREAD, "run")
