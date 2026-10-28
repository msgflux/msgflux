"""Revision commits read only the previous checkpoint envelope."""

import json
import sqlite3
from copy import deepcopy

import msgspec
import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import (
    CheckpointConflictError,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)

NAMESPACE = "agent"
THREAD = "thread"
RUN = "run"


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    if request.param == "sqlite":
        instance = SQLiteCheckpointStore(str(tmp_path / "checkpoints.sqlite3"))
    else:
        instance = InMemoryCheckpointStore()
    try:
        yield instance
    finally:
        close = getattr(instance, "close", None)
        if close is not None:
            close()


def _messages(content="hello", *, count=1):
    return ChatMessages(
        [{"role": "user", "content": f"{content}-{index}"} for index in range(count)],
        namespace=NAMESPACE,
        thread_id=THREAD,
    )._to_state()


def _state(content="hello", *, count=1):
    return {"status": "running", "messages": _messages(content, count=count)}


def _stored_state(store):
    if isinstance(store, SQLiteCheckpointStore):
        row = store._conn.execute(
            "SELECT state FROM checkpoints "
            "WHERE namespace=? AND thread_id=? AND run_id=?",
            (NAMESPACE, THREAD, RUN),
        ).fetchone()
        return json.loads(row[0])
    return store._data[NAMESPACE][THREAD][RUN]["state"]


def _write_stored_state(store, state):
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoints SET state=? "
            "WHERE namespace=? AND thread_id=? AND run_id=?",
            (json.dumps(state), NAMESPACE, THREAD, RUN),
        )
        store._conn.commit()
    else:
        store._data[NAMESPACE][THREAD][RUN]["state"] = state


def _commit_count(store):
    if isinstance(store, SQLiteCheckpointStore):
        return store._conn.execute(
            "SELECT COUNT(*) FROM checkpoint_commits "
            "WHERE namespace=? AND thread_id=? AND run_id=?",
            (NAMESPACE, THREAD, RUN),
        ).fetchone()[0]
    return len(store._data[NAMESPACE][THREAD][RUN].get("commits", []))


def _item_refs(store):
    return [
        entry["item_ref"] for entry in _stored_state(store)["_messages"]["item_entries"]
    ]


def _replace_payload(store, item_ref, *, sqlite_item, memory_item):
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "UPDATE checkpoint_message_items SET item=? "
            "WHERE namespace=? AND thread_id=? AND item_ref=?",
            (sqlite_item, NAMESPACE, THREAD, item_ref),
        )
        store._conn.commit()
    else:
        store._message_items[NAMESPACE][THREAD][item_ref] = memory_item


def _corrupt_payload(store, item_ref):
    _replace_payload(
        store,
        item_ref,
        sqlite_item="{invalid json",
        memory_item=b"\xc1",
    )


def _drop_payloads(store):
    if isinstance(store, SQLiteCheckpointStore):
        store._conn.execute(
            "DELETE FROM checkpoint_message_items WHERE namespace=? AND thread_id=?",
            (NAMESPACE, THREAD),
        )
        store._conn.commit()
    else:
        store._message_items[NAMESPACE][THREAD].clear()


def _first_commit(store, *, state=None):
    return store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        _state() if state is None else state,
        expected_revision=0,
    )


def test_commit_does_not_hydrate_previous_state(store, monkeypatch):
    first = _first_commit(store)

    def fail(*args, **kwargs):
        raise AssertionError("commit must not hydrate the previous state")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_denormalize_state", fail)
        second = store.commit_state(
            NAMESPACE,
            THREAD,
            RUN,
            {"status": "paused"},
            expected_revision=first.revision,
        )

    assert second.revision == first.revision + 1
    assert store.load_state(NAMESPACE, THREAD, RUN)["status"] == "paused"


def test_sqlite_commit_validates_prior_payloads_in_bounded_batches(tmp_path):
    store = SQLiteCheckpointStore(str(tmp_path / "envelope.sqlite3"))
    try:
        _first_commit(store, state=_state(count=1200))
        statements = []
        store._conn.set_trace_callback(statements.append)
        try:
            commit = store.commit_state(
                NAMESPACE,
                THREAD,
                RUN,
                {"status": "paused"},
                expected_revision=1,
            )
        finally:
            store._conn.set_trace_callback(None)

        assert commit.revision == 2
        payload_queries = [
            statement
            for statement in statements
            if "SELECT item_ref, item FROM checkpoint_message_items" in statement
        ]
        assert len(payload_queries) == 3
        assert all(" IN (" in statement for statement in payload_queries)
        assert not [
            statement
            for statement in statements
            if "SELECT item FROM checkpoint_message_items" in statement
        ]
    finally:
        store.close()


def test_commit_inherits_previous_branch_head_and_extensions(store):
    first = store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        _state(),
        expected_revision=0,
        branch_id="review",
        head_item_id="head-1",
        extension_state={"budget": {"remaining": 2}},
    )
    second = store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        {"status": "paused"},
        expected_revision=first.revision,
    )

    envelope = second.state["_checkpoint"]
    assert envelope["revision"] == 2
    assert envelope["stream_id"] == first.state["_checkpoint"]["stream_id"]
    assert envelope["branch_id"] == "review"
    assert envelope["head_item_id"] == "head-1"
    assert envelope["extensions"] == {"budget": {"remaining": 2}}
    assert store.load_state(NAMESPACE, THREAD, RUN)["_checkpoint"] == envelope


def test_commit_upgrades_legacy_state_without_envelope(store):
    store.save_state(NAMESPACE, THREAD, RUN, _state())
    assert "_checkpoint" not in _stored_state(store)

    commit = store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        {"status": "paused"},
        expected_revision=0,
    )

    assert commit.revision == 1
    assert commit.state["_checkpoint"]["branch_id"] is None
    assert store.load_state(NAMESPACE, THREAD, RUN)["status"] == "paused"


@pytest.mark.parametrize(
    ("checkpoint", "match"),
    [
        ({"schema_version": 2}, "schema version"),
        ({"revision": -1}, "revision"),
        ({"branch_id": 4}, "branch_id"),
        ("not-a-mapping", "mapping"),
    ],
)
def test_corrupt_previous_envelope_rejects_commit(store, checkpoint, match):
    _first_commit(store)
    stored = _stored_state(store)
    stored["_checkpoint"] = checkpoint
    _write_stored_state(store, stored)
    before = deepcopy(_stored_state(store))

    with pytest.raises(ValueError, match=match):
        store.commit_state(
            NAMESPACE,
            THREAD,
            RUN,
            {"status": "paused"},
            expected_revision=1,
            event={"event_type": "should.not.persist"},
        )

    assert _stored_state(store) == before
    assert store.load_events(NAMESPACE, THREAD, RUN) == []
    assert _commit_count(store) == 1


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ("not-an-entry", "entry is corrupted"),
        ({"item_id": "missing-ref"}, "missing or corrupted"),
    ],
)
def test_corrupt_previous_manifest_rejects_commit(store, entry, match):
    _first_commit(store)
    stored = _stored_state(store)
    stored["_messages"]["item_entries"] = [entry]
    _write_stored_state(store, stored)
    before = deepcopy(_stored_state(store))

    with pytest.raises(ValueError, match=match):
        store.commit_state(
            NAMESPACE,
            THREAD,
            RUN,
            {"status": "paused"},
            expected_revision=1,
            event={"event_type": "should.not.persist"},
        )

    assert _stored_state(store) == before
    assert store.load_events(NAMESPACE, THREAD, RUN) == []
    assert _commit_count(store) == 1


def test_missing_previous_item_rejects_commit(store):
    _first_commit(store)
    _drop_payloads(store)
    before = deepcopy(_stored_state(store))

    with pytest.raises(ValueError, match="missing or corrupted"):
        store.commit_state(
            NAMESPACE,
            THREAD,
            RUN,
            {"status": "paused"},
            expected_revision=1,
            event={"event_type": "should.not.persist"},
        )

    assert _stored_state(store) == before
    assert store.load_events(NAMESPACE, THREAD, RUN) == []
    assert _commit_count(store) == 1


def test_commit_rejects_corrupt_previous_payload_even_when_replaced(store):
    _first_commit(store, state=_state("old"))
    old_ref = _item_refs(store)[0]
    _corrupt_payload(store, old_ref)
    before = deepcopy(_stored_state(store))

    with pytest.raises(ValueError):
        store.commit_state(
            NAMESPACE,
            THREAD,
            RUN,
            _state("replacement"),
            expected_revision=1,
            event={"event_type": "should.not.persist"},
        )

    assert _stored_state(store) == before
    assert store.load_events(NAMESPACE, THREAD, RUN) == []
    assert _commit_count(store) == 1


def test_commit_rejects_retained_corrupt_payload(store):
    _first_commit(store, state=_state("old"))
    old_ref = _item_refs(store)[0]
    _corrupt_payload(store, old_ref)
    before = deepcopy(_stored_state(store))

    with pytest.raises(ValueError):
        store.commit_state(
            NAMESPACE,
            THREAD,
            RUN,
            _state("old"),
            expected_revision=1,
            event={"event_type": "should.not.persist"},
        )

    assert _stored_state(store) == before
    assert _item_refs(store) == [old_ref]
    assert store.load_events(NAMESPACE, THREAD, RUN) == []
    assert _commit_count(store) == 1


def test_commit_rejects_wrong_shaped_previous_payload(store):
    _first_commit(store, state=_state("old"))
    old_ref = _item_refs(store)[0]
    _replace_payload(
        store,
        old_ref,
        sqlite_item=json.dumps("not-an-item"),
        memory_item=msgspec.msgpack.encode("not-an-item"),
    )
    before = deepcopy(_stored_state(store))

    with pytest.raises(ValueError, match="missing or corrupted"):
        store.commit_state(
            NAMESPACE,
            THREAD,
            RUN,
            _state("replacement"),
            expected_revision=1,
            event={"event_type": "should.not.persist"},
        )

    assert _stored_state(store) == before
    assert store.load_events(NAMESPACE, THREAD, RUN) == []
    assert _commit_count(store) == 1


def test_sqlite_commit_failure_after_validation_rolls_back(tmp_path):
    store = SQLiteCheckpointStore(str(tmp_path / "rollback.sqlite3"))
    try:
        _first_commit(store)
        before = deepcopy(_stored_state(store))
        store._conn.execute(
            "CREATE TRIGGER fail_commit BEFORE INSERT ON checkpoint_commits "
            "BEGIN SELECT RAISE(ABORT, 'injected'); END"
        )
        store._conn.commit()

        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            store.commit_state(
                NAMESPACE,
                THREAD,
                RUN,
                {"status": "paused"},
                expected_revision=1,
                event={"event_type": "should.not.persist"},
            )

        assert _stored_state(store) == before
        assert store.load_events(NAMESPACE, THREAD, RUN) == []
        assert _commit_count(store) == 1

        store._conn.execute("DROP TRIGGER fail_commit")
        store._conn.commit()
        assert (
            store.commit_state(
                NAMESPACE,
                THREAD,
                RUN,
                {"status": "paused"},
                expected_revision=1,
            ).revision
            == 2
        )
    finally:
        store.close()


def test_commit_cas_uses_the_previous_envelope(store):
    first = _first_commit(store)
    with pytest.raises(CheckpointConflictError, match="expected 0, found 1"):
        store.commit_state(
            NAMESPACE,
            THREAD,
            RUN,
            {"status": "paused"},
            expected_revision=0,
            event={"event_type": "stale"},
        )

    assert store.load_events(NAMESPACE, THREAD, RUN) == []
    assert _commit_count(store) == 1
    second = store.commit_state(
        NAMESPACE,
        THREAD,
        RUN,
        {"status": "paused"},
        expected_revision=first.revision,
    )
    assert second.revision == 2


@pytest.mark.parametrize("expected_revision", [True, -1, 1.0, "1"])
def test_invalid_expected_revision_rolls_back(store, expected_revision):
    _first_commit(store)
    before = deepcopy(_stored_state(store))

    with pytest.raises(ValueError, match="expected_revision"):
        store.commit_state(
            NAMESPACE,
            THREAD,
            RUN,
            {"status": "paused"},
            expected_revision=expected_revision,
            event={"event_type": "should.not.persist"},
        )

    assert _stored_state(store) == before
    assert store.load_events(NAMESPACE, THREAD, RUN) == []
    assert _commit_count(store) == 1


def test_invalid_new_state_rolls_back(store):
    _first_commit(store)
    before = deepcopy(_stored_state(store))
    bad = {
        "status": "running",
        "messages": {
            "items": [
                {"item_id": "duplicate", "role": "user", "content": "one"},
                {"item_id": "duplicate", "role": "user", "content": "two"},
            ]
        },
    }

    with pytest.raises(ValueError, match="Duplicate"):
        store.commit_state(
            NAMESPACE,
            THREAD,
            RUN,
            bad,
            expected_revision=1,
            event={"event_type": "should.not.persist"},
        )

    assert _stored_state(store) == before
    assert store.load_events(NAMESPACE, THREAD, RUN) == []
    assert _commit_count(store) == 1


def test_sqlite_independent_connections_use_the_durable_envelope(tmp_path):
    path = str(tmp_path / "concurrent.sqlite3")
    first, second = SQLiteCheckpointStore(path), SQLiteCheckpointStore(path)
    try:
        commit = _first_commit(first)
        with pytest.raises(CheckpointConflictError):
            second.commit_state(
                NAMESPACE,
                THREAD,
                RUN,
                {"status": "paused"},
                expected_revision=0,
            )
        resumed = second.commit_state(
            NAMESPACE,
            THREAD,
            RUN,
            {"status": "paused"},
            expected_revision=commit.revision,
        )
        assert resumed.revision == 2
        assert second.load_state(NAMESPACE, THREAD, RUN)["status"] == "paused"
    finally:
        first.close()
        second.close()
