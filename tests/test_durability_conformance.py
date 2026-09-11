"""Public contract gates shared by every supported atomic checkpoint adapter."""

import asyncio
import json
import sqlite3
from threading import Event

import pytest

from msgflux.data.stores import (
    CheckpointConflictError,
    CheckpointCursorError,
    SQLiteCheckpointStore,
)
from msgflux.runtime import ApprovalBinding


@pytest.mark.parametrize("legacy_revision", [0, 4])
def test_legacy_upgrade_preserves_state_and_starts_new_feed(
    checkpoint_store, legacy_revision
):
    store = checkpoint_store
    legacy = {"status": "paused", "runtime": {"extensions": {"budget": 2}}}
    if legacy_revision:
        legacy["_checkpoint"] = {"schema_version": 1, "revision": legacy_revision}
    store.save_state("a", "t", "r", legacy)
    store.append_event("a", "t", "r", {"event_type": "legacy"})
    with pytest.raises(CheckpointCursorError):
        store.read_commits("a", "t", "r")
    restored = store.load_state("a", "t", "r")
    committed = store.commit_state(
        "a", "t", "r", restored, expected_revision=legacy_revision
    )
    assert committed.revision == legacy_revision + 1
    assert committed.state["runtime"]["extensions"] == {"budget": 2}
    assert store.load_events("a", "t", "r") == [{"event_type": "legacy"}]
    page = store.read_commits("a", "t", "r")
    assert page.snapshot == committed.state
    fork = store.fork_run("a", "t", "r", target_thread_id="fork", target_run_id="child")
    child = store.commit_state("a", "fork", "child", fork, expected_revision=0)
    assert child.state["_checkpoint"]["stream_id"] != page.cursor.stream_id
    with pytest.raises(CheckpointCursorError):
        store.read_commits("a", "fork", "child", after=page.cursor)
    assert store.read_commits("a", "t", "r").cursor == page.cursor


@pytest.mark.parametrize("schema", [2, True, "1"])
def test_rejected_upgrade_publishes_nothing(checkpoint_store, schema):
    store = checkpoint_store
    first = store.commit_state("a", "t", "r", {"status": "paused"})
    with pytest.raises(ValueError):
        store.commit_state(
            "a",
            "t",
            "r",
            {"_checkpoint": {"schema_version": schema}},
            expected_revision=first.revision,
            event={"event_type": "bad"},
        )
    page = store.read_commits("a", "t", "r")
    assert page.snapshot == first.state
    assert store.load_events("a", "t", "r") == []


@pytest.mark.asyncio
async def test_cancelled_commit_await_does_not_imply_rollback(
    checkpoint_store, monkeypatch
):
    store = checkpoint_store
    first = store.commit_state("a", "t", "r", {})
    before = store.read_commits("a", "t", "r").cursor
    entered, release, finished = Event(), Event(), Event()
    original = store.commit_state

    def delayed(*args, **kwargs):
        entered.set()
        try:
            assert release.wait(timeout=5)
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(store, "commit_state", delayed)
    task = asyncio.create_task(
        asyncio.to_thread(
            store.commit_state,
            "a",
            "t",
            "r",
            {"status": "paused"},
            expected_revision=first.revision,
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
    page = await store.aread_commits("a", "t", "r", after=before)
    assert len(page.events) == 1
    assert page.cursor.revision == first.revision + 1
    with pytest.raises(CheckpointConflictError):
        original("a", "t", "r", {}, expected_revision=first.revision)


@pytest.mark.asyncio
async def test_cancelled_decision_can_commit_and_retry_is_idempotent(
    approval_journal, monkeypatch
):
    store, _ = approval_journal
    binding = ApprovalBinding.from_call(
        namespace="a",
        thread_id="t",
        run_id="r",
        principal="user",
        tool_call_id="call",
        tool_name="write",
        tool_revision="v1",
        policy_version="p1",
        arguments={},
        resources={},
        required_permissions=(),
    )
    store.request(binding, request_id="request", expires_at=200)
    entered, release, finished = Event(), Event(), Event()
    original = store._update

    def delayed_update(namespace, request_id, update):
        def inside_transaction(record):
            entered.set()
            assert release.wait(timeout=5)
            return update(record)

        try:
            return original(namespace, request_id, inside_transaction)
        finally:
            finished.set()

    monkeypatch.setattr(store, "_update", delayed_update)
    task = asyncio.create_task(
        store.adecide("a", "request", approved=True, decided_by="host")
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 5)
    monkeypatch.setattr(store, "_update", original)
    record = await store.adecide("a", "request", approved=True, decided_by="host")
    assert record.status == "approved"
    assert [e.status for e in store.events("a", "request")] == ["pending", "approved"]


@pytest.mark.parametrize("operation", ["commit", "fork", "save_with_event"])
def test_sqlite_base_exception_rolls_back_open_transaction(
    tmp_path, monkeypatch, operation
):
    store = SQLiteCheckpointStore(str(tmp_path / "cancel.db"))
    before = store.commit_state("a", "t", "r", {"before": True})
    original = store._serialize

    def cancel(value):
        if store._conn.in_transaction and (
            operation != "commit" or value.get("event_type") == "cancel"
        ):
            assert store._conn.in_transaction
            raise asyncio.CancelledError()
        return original(value)

    try:
        monkeypatch.setattr(store, "_serialize", cancel)
        with pytest.raises(asyncio.CancelledError):
            if operation == "fork":
                store.fork_run(
                    "a", "t", "r", target_thread_id="other", target_run_id="child"
                )
            elif operation == "save_with_event":
                store.save_with_event("a", "t", "r", {}, {"event_type": "cancel"})
            else:
                store.commit_state(
                    "a",
                    "t",
                    "r",
                    {"partial": True},
                    expected_revision=1,
                    event={"event_type": "cancel"},
                )
        assert not store._conn.in_transaction
        assert store.load_state("a", "t", "r") == before.state
        assert store.load_events("a", "t", "r") == []
        assert store.load_state("a", "other", "child") is None
        monkeypatch.setattr(store, "_serialize", original)
        store.commit_state("a", "t", "r", {"recovered": True}, expected_revision=1)
    finally:
        store.close()


def test_sqlite_open_upgrades_pre_cursor_database_without_backfill(tmp_path):
    path = str(tmp_path / "legacy.db")
    legacy = {"status": "paused", "_checkpoint": {"schema_version": 1, "revision": 4}}
    with sqlite3.connect(path) as database:
        database.execute("""CREATE TABLE checkpoints (
            namespace TEXT NOT NULL, thread_id TEXT NOT NULL, run_id TEXT NOT NULL,
            status TEXT NOT NULL, state TEXT NOT NULL, created_at REAL NOT NULL,
            updated_at REAL NOT NULL, PRIMARY KEY(namespace, thread_id, run_id))""")
        database.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("a", "t", "r", "paused", json.dumps(legacy), 1.0, 1.0),
        )
    store = SQLiteCheckpointStore(path)
    try:
        assert store.load_state("a", "t", "r") == legacy
        with pytest.raises(CheckpointCursorError):
            store.read_commits("a", "t", "r")
        committed = store.commit_state("a", "t", "r", legacy, expected_revision=4)
        assert committed.revision == 5
        assert store.read_commits("a", "t", "r").snapshot == committed.state
        assert (
            store._conn.execute("SELECT count(*) FROM checkpoint_commits").fetchone()[0]
            == 1
        )
    finally:
        store.close()
