import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import pytest

from msgflux.data.stores import (
    CheckpointCursor,
    CheckpointCursorError,
    CheckpointConflictError,
    SQLiteCheckpointStore,
)
from msgflux.data.stores.observation import observe_checkpoints


@pytest.fixture
def store(checkpoint_store):
    return checkpoint_store


def commit(store, revision, **kwargs):
    return store.commit_state(
        "a",
        "t",
        "r",
        {"status": "running", "count": revision + 1},
        expected_revision=revision,
        **kwargs,
    )


def test_snapshot_cursor_paging_and_legacy_events_unchanged(store):
    commit(store, 0, event={"event_type": "begin"})
    initial = store.read_commits("a", "t", "r")
    assert initial.snapshot["count"] == initial.cursor.revision == 1
    assert not initial.events
    cursor = CheckpointCursor(**asdict(initial.cursor))
    initial.snapshot["count"] = 900
    for revision in range(1, 5):
        commit(store, revision)
    page = store.read_commits("a", "t", "r", after=cursor, limit=2)
    assert page.snapshot is None
    assert [e.cursor.revision for e in page.events] == [2, 3]
    assert len({e.event_id for e in page.events}) == 2
    page.events[0].data["status"] = "tampered"
    again = store.read_commits("a", "t", "r", after=cursor, limit=2)
    assert again.events[0].data["status"] == "running"
    tail = store.read_commits("a", "t", "r", after=page.cursor, limit=2)
    assert [e.cursor.revision for e in tail.events] == [4, 5]
    assert not store.read_commits("a", "t", "r", after=tail.cursor).events
    assert store.load_events("a", "t", "r") == [{"event_type": "begin"}]


def test_invalid_missing_and_recreated_streams_fail_loudly(store):
    with pytest.raises(CheckpointCursorError):
        store.read_commits("a", "t", "missing")
    commit(store, 0)
    cursor = store.read_commits("a", "t", "r").cursor
    for invalid in (replace(cursor, revision=100), replace(cursor, run_id="other")):
        with pytest.raises(CheckpointCursorError):
            store.read_commits("a", "t", "r", after=invalid)
    for limit in (0, True, 1001):
        with pytest.raises(ValueError):
            store.read_commits("a", "t", "r", limit=limit)
    store.delete_run("a", "t", "r")
    commit(store, 0)
    with pytest.raises(CheckpointCursorError):
        store.read_commits("a", "t", "r", after=cursor)


def test_stale_commit_publishes_no_transition(store):
    commit(store, 0)
    cursor = store.read_commits("a", "t", "r").cursor
    with pytest.raises(CheckpointConflictError):
        commit(store, 0)
    assert not store.read_commits("a", "t", "r", after=cursor).events


def test_snapshot_race_has_no_lost_commits(store):
    commit(store, 0)
    with ThreadPoolExecutor(max_workers=1) as pool:

        def produce():
            for revision in range(1, 30):
                commit(store, revision)

        future = pool.submit(produce)
        initial = store.read_commits("a", "t", "r")
        assert initial.snapshot["count"] == initial.cursor.revision
        future.result(timeout=10)
    page = store.read_commits("a", "t", "r", after=initial.cursor)
    assert [e.cursor.revision for e in page.events] == list(
        range(initial.cursor.revision + 1, 31)
    )


@pytest.mark.asyncio
async def test_async_observer_reconnect_and_cancel_do_not_cancel_writer(store):
    commit(store, 0)
    stream = observe_checkpoints(store, "a", "t", "r", poll_interval=0.001)
    initial = await stream.__anext__()
    await stream.aclose()
    commit(store, 1)
    resumed = observe_checkpoints(store, "a", "t", "r", after=initial.cursor)
    page = await resumed.__anext__()
    assert page.cursor.revision == 2
    task = asyncio.create_task(resumed.__anext__())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    commit(store, 2)
    assert (
        await store.aread_commits("a", "t", "r", after=page.cursor)
    ).cursor.revision == 3


def test_sqlite_restart_and_transaction_rollback(tmp_path):
    path = str(tmp_path / "commits.db")
    first = SQLiteCheckpointStore(path)
    commit(first, 0)
    cursor = first.read_commits("a", "t", "r").cursor
    first.close()
    second = SQLiteCheckpointStore(path)
    second._conn.execute(
        "CREATE TRIGGER fail_commit BEFORE INSERT ON checkpoint_commits BEGIN SELECT RAISE(ABORT, 'injected'); END"
    )
    with pytest.raises(Exception, match="injected"):
        commit(second, 1)
    assert second.load_state("a", "t", "r")["count"] == 1
    assert not second.read_commits("a", "t", "r", after=cursor).events
    second._conn.execute("DROP TRIGGER fail_commit")
    commit(second, 1)
    assert second.read_commits("a", "t", "r", after=cursor).cursor.revision == 2
    second._conn.execute("DELETE FROM checkpoint_commits WHERE revision=2")
    second._conn.commit()
    with pytest.raises(CheckpointCursorError, match="unavailable"):
        second.read_commits("a", "t", "r", after=cursor)
    second.close()


def test_sqlite_independent_connection_snapshot_race(tmp_path):
    path = str(tmp_path / "race.db")
    reader, writer = SQLiteCheckpointStore(path), SQLiteCheckpointStore(path)
    try:
        commit(writer, 0)
        with ThreadPoolExecutor(max_workers=1) as pool:

            def produce():
                for revision in range(1, 20):
                    commit(writer, revision)

            future = pool.submit(produce)
            initial = reader.read_commits("a", "t", "r")
            assert initial.snapshot["count"] == initial.cursor.revision
            future.result(timeout=10)
        page = reader.read_commits("a", "t", "r", after=initial.cursor)
        assert [event.cursor.revision for event in page.events] == list(
            range(initial.cursor.revision + 1, 21)
        )
    finally:
        reader.close()
        writer.close()
