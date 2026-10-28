"""Fail-closed boundaries for shared checkpoint manifest storage."""

import json
import sqlite3
from copy import deepcopy

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.data.stores._manifest import iter_manifest_entries

NS = "manifest-integrity"
THREAD = "one-thread"


def _state(count=129):
    return {
        "status": "completed",
        "messages": ChatMessages(
            [
                {"role": "user", "content": str(i), "item_id": f"itm_{i}"}
                for i in range(count)
            ],
            namespace=NS,
            thread_id=THREAD,
        )._to_state(),
    }


def _raw(store, run):
    if isinstance(store, SQLiteCheckpointStore):
        return json.loads(
            store._conn.execute(
                "SELECT state FROM checkpoints WHERE namespace=? AND thread_id=? AND run_id=?",
                (NS, THREAD, run),
            ).fetchone()[0]
        )
    return deepcopy(store._data[NS][THREAD][run]["state"])


def _replace_raw(store, run, value):
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoints SET state=? WHERE namespace=? AND thread_id=? AND run_id=?",
            (json.dumps(value), NS, THREAD, run),
        )
        store._conn.commit()
    else:
        store._data[NS][THREAD][run]["state"] = value


def _pages(store):
    if isinstance(store, SQLiteCheckpointStore):
        return dict(
            store._conn.execute(
                "SELECT segment_ref, data FROM checkpoint_manifest_segments WHERE namespace=? AND thread_id=?",
                (NS, THREAD),
            )
        )
    return dict(store._manifest_pages[NS][THREAD])


def test_new_run_rejects_reusing_a_corrupted_shared_segment(checkpoint_store):
    store = checkpoint_store
    state = _state()
    store.commit_state(NS, THREAD, "original", state, expected_revision=0)
    head = _raw(store, "original")["_messages"]["manifest"]["head"]
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoint_manifest_segments SET data=? WHERE namespace=? AND thread_id=? AND segment_ref=?",
            (b"invalid-page", NS, THREAD, head),
        )
        store._conn.commit()
    else:
        store._manifest_pages[NS][THREAD][head] = b"invalid-page"
    before = _pages(store)

    with pytest.raises(ValueError):
        store.commit_state(NS, THREAD, "new-run", state, expected_revision=0)

    assert store.load_state(NS, THREAD, "new-run") is None
    assert _pages(store) == before


@pytest.mark.parametrize("operation", ["delete", "clear"])
@pytest.mark.parametrize("manifest", [None, "invalid", [], 1])
def test_delete_refuses_to_collect_an_invalid_remaining_manifest(
    checkpoint_store, manifest, operation
):
    store = checkpoint_store
    store.commit_state(NS, THREAD, "keep", _state(), expected_revision=0)
    store.commit_state(NS, THREAD, "delete", _state(1), expected_revision=0)
    raw = _raw(store, "keep")
    raw["_messages"]["manifest"] = manifest
    _replace_raw(store, "keep", raw)
    before_pages = _pages(store)
    before_delete = store.load_state(NS, THREAD, "delete")

    if operation == "clear":
        if isinstance(store, SQLiteCheckpointStore):
            store._conn.execute(
                "UPDATE checkpoints SET updated_at=1e15 WHERE run_id='keep'"
            )
            store._conn.execute(
                "UPDATE checkpoints SET updated_at=0 WHERE run_id='delete'"
            )
            store._conn.commit()
        else:
            store._data[NS][THREAD]["keep"]["updated_at"] = 1e15
            store._data[NS][THREAD]["delete"]["updated_at"] = 0
    with pytest.raises(ValueError):
        if operation == "delete":
            store.delete_run(NS, THREAD, "delete")
        else:
            store.clear(namespace=NS, thread_id=THREAD, older_than=1)

    assert store.load_state(NS, THREAD, "delete") == before_delete
    assert _pages(store) == before_pages


def test_mixed_manifest_layout_is_rejected_on_read_and_commit(checkpoint_store):
    store = checkpoint_store
    state = _state()
    store.commit_state(NS, THREAD, "mixed", state, expected_revision=0)
    raw = _raw(store, "mixed")
    raw["_messages"]["item_entries"] = []
    _replace_raw(store, "mixed", raw)

    with pytest.raises(ValueError):
        store.load_state(NS, THREAD, "mixed")
    with pytest.raises(ValueError):
        store.commit_state(NS, THREAD, "mixed", state, expected_revision=1)
    assert _raw(store, "mixed")["_checkpoint"]["revision"] == 1


def test_large_legacy_inline_snapshot_survives_fork_and_gc(checkpoint_store):
    store = checkpoint_store
    state = _state(300)
    store.commit_state(NS, THREAD, "legacy", state, expected_revision=0)
    expected = store.load_state(NS, THREAD, "legacy")
    raw = _raw(store, "legacy")
    pages = _pages(store)
    entries = list(
        iter_manifest_entries(raw["_messages"].pop("manifest"), pages.__getitem__)
    )
    raw["_messages"]["item_entries"] = entries
    _replace_raw(store, "legacy", raw)
    # A real old inline snapshot has no segment rows at all.
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute("DELETE FROM checkpoint_manifest_segments")
        store._conn.commit()
    else:
        store._manifest_pages[NS][THREAD].clear()

    assert store.load_state(NS, THREAD, "legacy") == expected
    boundary = state["messages"]["items"][150]["item_id"]
    forked = store.fork_run(
        NS,
        THREAD,
        "legacy",
        target_thread_id="forked",
        target_run_id="fork",
        at_item_id=boundary,
    )
    assert forked["messages"]["items"] == state["messages"]["items"][:151]
    assert store.load_state(NS, THREAD, "legacy") == expected
    assert store.delete_run(NS, THREAD, "legacy") is True
    assert (
        store.load_state(NS, "forked", "fork")["messages"]["items"]
        == state["messages"]["items"][:151]
    )


@pytest.mark.parametrize("entry", [None, {}, {"item_ref": 42}, {"item_ref": None}])
def test_gc_rejects_malformed_legacy_inline_entries(checkpoint_store, entry):
    store = checkpoint_store
    store.commit_state(NS, THREAD, "keep", _state(2), expected_revision=0)
    store.commit_state(NS, THREAD, "delete", _state(1), expected_revision=0)
    raw = _raw(store, "keep")
    raw["_messages"]["item_entries"] = [entry]
    _replace_raw(store, "keep", raw)
    before_delete = store.load_state(NS, THREAD, "delete")

    with pytest.raises(ValueError):
        store.delete_run(NS, THREAD, "delete")

    assert store.load_state(NS, THREAD, "delete") == before_delete
    assert _raw(store, "keep") == raw


def test_missing_manifest_layout_rejects_read_commit_and_gc(checkpoint_store):
    store = checkpoint_store
    state = _state(2)
    store.commit_state(NS, THREAD, "keep", state, expected_revision=0)
    store.commit_state(NS, THREAD, "delete", _state(1), expected_revision=0)
    raw = _raw(store, "keep")
    raw["_messages"].pop("item_entries")
    _replace_raw(store, "keep", raw)
    before_delete = store.load_state(NS, THREAD, "delete")

    with pytest.raises(ValueError):
        store.load_state(NS, THREAD, "keep")
    with pytest.raises(ValueError):
        store.commit_state(NS, THREAD, "keep", state, expected_revision=1)
    with pytest.raises(ValueError):
        store.delete_run(NS, THREAD, "delete")
    assert store.load_state(NS, THREAD, "delete") == before_delete
    assert _raw(store, "keep") == raw


def test_legacy_inline_entries_without_occurrence_ids_remain_collectable(
    checkpoint_store,
):
    store = checkpoint_store
    state = _state(300)
    store.commit_state(NS, THREAD, "legacy", state, expected_revision=0)
    raw = _raw(store, "legacy")
    pages = _pages(store)
    entries = list(
        iter_manifest_entries(raw["_messages"].pop("manifest"), pages.__getitem__)
    )
    for entry in entries:
        entry.pop("item_id")
    raw["_messages"]["item_entries"] = entries
    _replace_raw(store, "legacy", raw)
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute("DELETE FROM checkpoint_manifest_segments")
        store._conn.commit()
    else:
        store._manifest_pages[NS][THREAD].clear()
    before = store.load_state(NS, THREAD, "legacy")
    assert all("item_id" not in item for item in before["messages"]["items"])
    store.commit_state(NS, THREAD, "delete", _state(1), expected_revision=0)

    assert store.delete_run(NS, THREAD, "delete") is True
    assert store.load_state(NS, THREAD, "legacy") == before


@pytest.mark.parametrize("method", ["save_state", "save_with_event"])
def test_sqlite_replacement_reserves_writer_before_preparing_snapshot(
    tmp_path, monkeypatch, method
):
    path = str(tmp_path / "replacement-lock.sqlite")
    first = SQLiteCheckpointStore(path)
    second = SQLiteCheckpointStore(path)
    state = _state()
    alternate = deepcopy(state)
    alternate["messages"]["items"][-1]["content"] = "other-writer"
    try:
        first.save_state(NS, THREAD, "target", state)
        second._conn.execute("PRAGMA busy_timeout=0")
        original = first._normalize_state

        def prepare(*args, **kwargs):
            # The old head was already selected. An independent connection
            # must not replace it while this snapshot is being prepared.
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                second.save_state(NS, THREAD, "target", alternate)
            return original(*args, **kwargs)

        monkeypatch.setattr(first, "_normalize_state", prepare)
        if method == "save_state":
            first.save_state(NS, THREAD, "target", state)
        else:
            first.save_with_event(NS, THREAD, "target", state, {"event_type": "saved"})
        assert first.load_state(NS, THREAD, "target")["messages"] == state["messages"]
    finally:
        first.close()
        second.close()
