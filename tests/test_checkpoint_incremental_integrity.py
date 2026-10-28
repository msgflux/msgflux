"""Independent archive integrity and crash boundaries for incremental writes."""

import json
import multiprocessing
import os
from copy import deepcopy

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import CheckpointConflictError, SQLiteCheckpointStore

NS = "incremental-integrity"
THREAD = "one-thread"
RUN = "one-run"


def _state(count=300):
    return {
        "status": "running",
        "messages": ChatMessages(
            [
                {"role": "user", "content": str(i), "item_id": f"itm_{i}"}
                for i in range(count)
            ],
            namespace=NS,
            thread_id=THREAD,
        )._to_state(),
    }


def _new_item():
    return {"role": "assistant", "content": "appended", "item_id": "itm_new"}


def _corrupt_first_payload(store):
    if isinstance(store, SQLiteCheckpointStore):
        ref = store._conn.execute(
            "SELECT item_ref FROM checkpoint_message_items WHERE namespace=? AND thread_id=? ORDER BY rowid LIMIT 1",
            (NS, THREAD),
        ).fetchone()[0]
        store._conn.execute(
            "UPDATE checkpoint_message_items SET item=? WHERE namespace=? AND thread_id=? AND item_ref=?",
            ("malformed-json", NS, THREAD, ref),
        )
        store._conn.commit()
    else:
        items = store._message_items[NS][THREAD]
        items[next(iter(items))] = b"\xc1"


def test_suffix_does_not_claim_cold_integrity_but_append_remains_strict(
    checkpoint_store,
):
    store = checkpoint_store
    store.commit_state(NS, THREAD, RUN, _state(), expected_revision=0)
    before = store.load_tail(NS, THREAD, RUN, limit=3)
    _corrupt_first_payload(store)
    assert store.load_tail(NS, THREAD, RUN, limit=3) == before
    with pytest.raises((ValueError, TypeError)):
        store.append_state(NS, THREAD, RUN, [_new_item()], base=before.base)
    assert store.load_tail(NS, THREAD, RUN, limit=3) == before
    assert store.load_events(NS, THREAD, RUN) == []
    with pytest.raises((ValueError, TypeError)):
        store.load_state(NS, THREAD, RUN)


def test_same_revision_full_save_invalidates_tail_base(checkpoint_store):
    store = checkpoint_store
    store.commit_state(NS, THREAD, RUN, _state(), expected_revision=0)
    tail = store.load_tail(NS, THREAD, RUN, limit=1)
    replaced = deepcopy(store.load_state(NS, THREAD, RUN))
    replaced["status"] = "paused"
    store.save_state(NS, THREAD, RUN, replaced)
    with pytest.raises(CheckpointConflictError):
        store.append_state(NS, THREAD, RUN, [_new_item()], base=tail.base)
    assert store.load_state(NS, THREAD, RUN) == replaced


def _crash_append(path, phase):
    store = SQLiteCheckpointStore(path)
    tail = store.load_tail(NS, THREAD, RUN, limit=1)
    if phase == "before_commit":

        def crash(*_args, **_kwargs):
            os._exit(77)

        store._cleanup_orphaned_items = crash
    store.append_state(
        NS,
        THREAD,
        RUN,
        [_new_item()],
        base=tail.base,
        event={"event_type": "incremental_crash_probe"},
    )
    os._exit(78)


@pytest.mark.parametrize("phase", ["before_commit", "after_commit"])
def test_incremental_sqlite_crash_recovery_is_atomic(tmp_path, phase):
    path = str(tmp_path / "incremental-crash.db")
    store = SQLiteCheckpointStore(path)
    store.commit_state(NS, THREAD, RUN, _state(), expected_revision=0)
    before = store.load_state(NS, THREAD, RUN)
    store.close()
    worker = multiprocessing.get_context("spawn").Process(
        target=_crash_append, args=(path, phase)
    )
    worker.start()
    try:
        worker.join(30)
        assert not worker.is_alive(), "Incremental crash worker exceeded timeout"
        assert worker.exitcode == (77 if phase == "before_commit" else 78)
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join(5)
    store = SQLiteCheckpointStore(path)
    try:
        restored = store.load_state(NS, THREAD, RUN)
        events = store.load_events(NS, THREAD, RUN)
        if phase == "before_commit":
            assert restored == before
            assert events == []
        else:
            assert restored["messages"]["items"] == before["messages"]["items"] + [
                _new_item()
            ]
            assert restored["_checkpoint"]["revision"] == 2
            assert events == [{"event_type": "incremental_crash_probe"}]
        raw = json.loads(
            store._conn.execute(
                "SELECT state FROM checkpoints WHERE namespace=? AND thread_id=? AND run_id=?",
                (NS, THREAD, RUN),
            ).fetchone()[0]
        )
        from msgflux.data.stores._manifest import iter_manifest_pages

        live = {
            ref
            for ref, _ in iter_manifest_pages(
                raw["_messages"]["manifest"],
                lambda ref: store._read_manifest_segment(NS, THREAD, ref),
            )
        }
        stored = {
            row[0]
            for row in store._conn.execute(
                "SELECT segment_ref FROM checkpoint_manifest_segments"
            )
        }
        assert live == stored
    finally:
        store.close()


@pytest.mark.parametrize("existing", [True, False])
def test_append_deduplicates_equivalent_payloads_despite_mapping_key_order(
    checkpoint_store,
    existing,
):
    store = checkpoint_store
    original = {
        "role": "assistant",
        "content": "same payload",
        "metadata": {"a": 1, "b": 2},
        "item_id": "itm_old",
    }
    state = _state(0)
    state["messages"]["items"] = [original] if existing else []
    store.commit_state(NS, THREAD, RUN, state, expected_revision=0)
    tail = store.load_tail(NS, THREAD, RUN, limit=1)
    reordered = {
        "metadata": {"b": 2, "a": 1},
        "content": "same payload",
        "role": "assistant",
        "item_id": "itm_reordered",
    }
    store.append_state(
        NS,
        THREAD,
        RUN,
        [reordered] if existing else [original, reordered],
        base=tail.base,
    )
    assert store.load_state(NS, THREAD, RUN)["messages"]["items"] == [
        original,
        reordered,
    ]
    if isinstance(store, SQLiteCheckpointStore):
        assert (
            store._conn.execute(
                "SELECT COUNT(*) FROM checkpoint_message_items"
            ).fetchone()[0]
            == 1
        )
    else:
        assert len(store._message_items[NS][THREAD]) == 1


def test_new_payload_collision_preserves_json_scalar_types(checkpoint_store):
    import msgspec

    store = checkpoint_store
    store.commit_state(NS, THREAD, RUN, _state(0), expected_revision=0)
    item = {
        "role": "user",
        "content": "typed payload",
        "metadata": {"flag": True},
        "item_id": "itm_donor",
    }
    donor = _state(0)
    donor["messages"]["items"] = [item]
    store.commit_state(NS, THREAD, "donor", donor, expected_revision=0)
    payload = {key: value for key, value in item.items() if key != "item_id"}
    ref = store._item_ref(payload)
    corrupt = deepcopy(payload)
    corrupt["metadata"]["flag"] = 1
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoint_message_items SET item=? WHERE item_ref=?",
            (json.dumps(corrupt), ref),
        )
        store._conn.commit()
    else:
        store._message_items[NS][THREAD][ref] = msgspec.msgpack.encode(corrupt)
    tail = store.load_tail(NS, THREAD, RUN, limit=0)
    with pytest.raises(ValueError):
        store.append_state(
            NS, THREAD, RUN, [{**item, "item_id": "itm_append"}], base=tail.base
        )
    assert store.load_tail(NS, THREAD, RUN, limit=0) == tail


def test_memory_tail_rejects_ambiguous_non_json_control_metadata():
    from msgflux.data.stores import InMemoryCheckpointStore

    store = InMemoryCheckpointStore()
    state = _state(1)
    state["opaque_control"] = b"data"
    store.commit_state(NS, THREAD, RUN, state, expected_revision=0)
    assert store.load_state(NS, THREAD, RUN)["opaque_control"] == b"data"
    with pytest.raises(ValueError, match="JSON-compatible"):
        store.load_tail(NS, THREAD, RUN)


def test_append_acknowledgment_cannot_mutate_durable_control_state(checkpoint_store):
    store = checkpoint_store
    state = _state(1)
    state["runtime"] = {"extensions": {"notes": ["old"]}}
    state["messages"]["metadata"] = {"notes": ["old"]}
    store.commit_state(NS, THREAD, RUN, state, expected_revision=0)
    tail = store.load_tail(NS, THREAD, RUN)
    ack = store.append_state(NS, THREAD, RUN, [_new_item()], base=tail.base)
    persisted = store.load_state(NS, THREAD, RUN)
    ack.state["runtime"]["extensions"]["notes"].append("leaked")
    ack.state["_checkpoint"]["extensions"]["notes"].append("leaked")
    ack.state["messages"]["metadata"]["notes"].append("leaked")
    assert store.load_state(NS, THREAD, RUN) == persisted


def test_control_only_append_preserves_large_idless_inline_legacy_history(
    checkpoint_store,
):
    from msgflux.data.stores._manifest import iter_manifest_entries

    store = checkpoint_store
    store.commit_state(NS, THREAD, RUN, _state(), expected_revision=0)
    if isinstance(store, SQLiteCheckpointStore):
        raw = json.loads(
            store._conn.execute(
                "SELECT state FROM checkpoints WHERE namespace=? AND thread_id=? AND run_id=?",
                (NS, THREAD, RUN),
            ).fetchone()[0]
        )
    else:
        raw = deepcopy(store._data[NS][THREAD][RUN]["state"])
    messages = raw["_messages"]
    entries = list(
        iter_manifest_entries(
            messages.pop("manifest"),
            lambda ref: store._read_manifest_segment(NS, THREAD, ref),
        )
    )
    for entry in entries:
        entry.pop("item_id")
    messages["item_entries"] = entries
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoints SET state=? WHERE namespace=? AND thread_id=? AND run_id=?",
            (json.dumps(raw), NS, THREAD, RUN),
        )
        store._conn.commit()
    else:
        store._data[NS][THREAD][RUN]["state"] = raw
    before = store.load_state(NS, THREAD, RUN)["messages"]["items"]
    tail = store.load_tail(NS, THREAD, RUN, limit=1)
    ack = store.append_state(
        NS, THREAD, RUN, [], base=tail.base, state_updates={"status": "paused"}
    )
    assert ack.state["messages"]["items"] == []
    restored = store.load_state(NS, THREAD, RUN)
    assert restored["status"] == "paused"
    assert restored["messages"]["items"] == before


def test_append_uses_one_isolated_delta_for_ack_and_storage(
    checkpoint_store, monkeypatch
):
    store = checkpoint_store
    store.commit_state(NS, THREAD, RUN, _state(2), expected_revision=0)
    tail = store.load_tail(NS, THREAD, RUN)
    incoming = [_new_item()]
    expected = deepcopy(incoming)
    original = store._commit_envelope_from_state

    def mutate_caller_during_archive_scan(*args, **kwargs):
        incoming[0]["content"] = "caller changed this during validation"
        return original(*args, **kwargs)

    monkeypatch.setattr(
        store, "_commit_envelope_from_state", mutate_caller_during_archive_scan
    )
    ack = store.append_state(NS, THREAD, RUN, incoming, base=tail.base)
    assert ack.state["messages"]["items"] == expected
    assert store.load_state(NS, THREAD, RUN)["messages"]["items"][-1:] == expected


@pytest.mark.parametrize(
    "field", ["state_updates", "message_updates", "extension_state"]
)
def test_non_json_append_control_rejects_without_publication(checkpoint_store, field):
    store = checkpoint_store
    store.commit_state(NS, THREAD, RUN, _state(), expected_revision=0)
    tail = store.load_tail(NS, THREAD, RUN, limit=0)
    before = store.load_state(NS, THREAD, RUN)
    with pytest.raises(ValueError, match="JSON-compatible"):
        store.append_state(
            NS,
            THREAD,
            RUN,
            [_new_item()],
            base=tail.base,
            event={"event_type": "must_not_publish"},
            **{field: {"opaque": b"bytes"}},
        )
    assert store.load_state(NS, THREAD, RUN) == before
    assert store.load_events(NS, THREAD, RUN) == []
    assert store.load_tail(NS, THREAD, RUN, limit=0) == tail


def test_append_rejects_duplicate_ids_already_in_archive(checkpoint_store):
    store = checkpoint_store
    store.commit_state(NS, THREAD, RUN, _state(2), expected_revision=0)
    if isinstance(store, SQLiteCheckpointStore):
        raw = json.loads(
            store._conn.execute(
                "SELECT state FROM checkpoints WHERE namespace=? AND thread_id=? AND run_id=?",
                (NS, THREAD, RUN),
            ).fetchone()[0]
        )
    else:
        raw = deepcopy(store._data[NS][THREAD][RUN]["state"])
    raw["_messages"]["item_entries"][1]["item_id"] = "itm_0"
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoints SET state=? WHERE namespace=? AND thread_id=? AND run_id=?",
            (json.dumps(raw), NS, THREAD, RUN),
        )
        store._conn.commit()
    else:
        store._data[NS][THREAD][RUN]["state"] = raw
    tail = store.load_tail(NS, THREAD, RUN, limit=0)
    before = store.load_state(NS, THREAD, RUN)
    with pytest.raises(ValueError, match="duplicate item IDs"):
        store.append_state(NS, THREAD, RUN, [_new_item()], base=tail.base)
    assert store.load_state(NS, THREAD, RUN) == before
    assert store.load_tail(NS, THREAD, RUN, limit=0) == tail


@pytest.mark.asyncio
async def test_async_incremental_wrappers_preserve_archive_and_conflicts(
    checkpoint_store,
):
    store = checkpoint_store
    store.commit_state(NS, THREAD, RUN, _state(2), expected_revision=0)
    tail = await store.aload_tail(NS, THREAD, RUN, limit=1)
    ack = await store.aappend_state(NS, THREAD, RUN, [_new_item()], base=tail.base)
    assert ack.base.item_count == 3
    assert store.load_state(NS, THREAD, RUN)["messages"]["items"][-1:] == [_new_item()]
    with pytest.raises(CheckpointConflictError):
        await store.aappend_state(NS, THREAD, RUN, [], base=tail.base)
