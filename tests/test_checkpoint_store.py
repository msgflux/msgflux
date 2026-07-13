from concurrent.futures import ThreadPoolExecutor

import msgspec
import pytest

from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.data.stores import Store


def _message_state(turn_id: str, content: str = "hello"):
    return {
        "items": [{"role": "user", "content": content}],
        "thread_id": "session_1",
        "namespace": "agent:test",
        "turns": [{"turn_id": turn_id, "status": "completed"}],
        "active_turn_index": None,
    }


def test_in_memory_checkpoint_store_state_and_events():
    store = InMemoryCheckpointStore()

    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "running", "step": 1},
    )
    store.append_event(
        "agent:test",
        "session_1",
        "run_1",
        {"event_type": "step_completed", "step": 1},
    )

    state = store.load_state("agent:test", "session_1", "run_1")
    events = store.load_events("agent:test", "session_1", "run_1")
    runs = store.list_runs("agent:test", "session_1")

    assert state == {"status": "running", "step": 1}
    assert events == [{"event_type": "step_completed", "step": 1}]
    assert runs[0]["run_id"] == "run_1"
    assert runs[0]["status"] == "running"


def test_in_memory_checkpoint_store_normalizes_messages():
    store = InMemoryCheckpointStore()
    messages = _message_state("run_1")

    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "messages": messages},
    )

    state = store.load_state("agent:test", "session_1", "run_1")

    assert state == {"status": "completed", "messages": messages}
    raw_state = store._data["agent:test"]["session_1"]["run_1"]["state"]
    assert raw_state == {
        "status": "completed",
        "_messages": {
            "state": {
                "thread_id": "session_1",
                "namespace": "agent:test",
                "turns": [{"turn_id": "run_1", "status": "completed"}],
                "active_turn_index": None,
            },
            "item_refs": [store._item_ref(messages["items"][0])],
        },
    }
    encoded_item = store._message_items["agent:test"]["session_1"][
        store._item_ref(messages["items"][0])
    ]
    assert isinstance(encoded_item, bytes)
    assert msgspec.msgpack.decode(encoded_item) == {
        "role": "user",
        "content": "hello",
    }
    assert len(store._item_ref(messages["items"][0])) == len("item_") + 32


def test_in_memory_checkpoint_store_reuses_thread_items_across_runs():
    store = InMemoryCheckpointStore()
    first = _message_state("run_1", "first")
    second = _message_state("run_2", "second")
    second["items"] = [first["items"][0], second["items"][0]]

    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "messages": first},
    )
    store.save_state(
        "agent:test",
        "session_1",
        "run_2",
        {"status": "completed", "messages": second},
    )

    state = store.load_state("agent:test", "session_1", "run_2")

    assert state == {"status": "completed", "messages": second}
    assert len(store._message_items["agent:test"]["session_1"]) == 2


def test_in_memory_checkpoint_store_preserves_forked_items():
    store = InMemoryCheckpointStore()
    first = _message_state("run_1", "first")
    second = _message_state("run_2", "second")
    fork = _message_state("run_3", "alternate")
    second["items"] = [first["items"][0], second["items"][0]]
    fork["items"] = [first["items"][0], fork["items"][0]]

    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "messages": first},
    )
    store.save_state(
        "agent:test",
        "session_1",
        "run_2",
        {"status": "completed", "messages": second},
    )
    store.save_state(
        "agent:test",
        "session_1",
        "run_3",
        {"status": "completed", "messages": fork},
    )

    assert store.load_state("agent:test", "session_1", "run_2")["messages"] == second
    assert store.load_state("agent:test", "session_1", "run_3")["messages"] == fork
    assert len(store._message_items["agent:test"]["session_1"]) == 3


def test_in_memory_checkpoint_store_raises_on_missing_item_ref():
    store = InMemoryCheckpointStore()
    messages = _message_state("run_1")
    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "messages": messages},
    )
    store._message_items["agent:test"]["session_1"].clear()

    with pytest.raises(ValueError, match="missing or corrupted"):
        store.load_state("agent:test", "session_1", "run_1")


def test_in_memory_checkpoint_store_cleans_orphaned_items():
    store = InMemoryCheckpointStore()
    first = _message_state("run_1", "first")
    second = _message_state("run_2", "second")
    second["items"] = [first["items"][0], second["items"][0]]
    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "messages": first},
    )
    store.save_state(
        "agent:test",
        "session_1",
        "run_2",
        {"status": "completed", "messages": second},
    )

    assert store.delete_run("agent:test", "session_1", "run_2") is True

    assert len(store._message_items["agent:test"]["session_1"]) == 1
    assert store.load_state("agent:test", "session_1", "run_1")["messages"] == first


def test_in_memory_checkpoint_store_forks_run_to_new_thread():
    store = InMemoryCheckpointStore()
    first = _message_state("run_1", "first")
    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "messages": first},
    )

    forked = store.fork_run(
        "agent:test",
        "session_1",
        "run_1",
        target_thread_id="session_fork",
        target_run_id="run_fork",
        status="running",
    )

    assert forked["status"] == "running"
    assert forked["messages"] == first
    assert store.load_state("agent:test", "session_fork", "run_fork") == forked
    assert len(store._message_items["agent:test"]["session_fork"]) == 1


def test_sqlite_checkpoint_store_roundtrip(tmp_path):
    path = tmp_path / "checkpoints.sqlite3"
    store = SQLiteCheckpointStore(path=str(path))

    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "result": 42},
    )
    state = store.load_state("agent:test", "session_1", "run_1")
    runs = store.list_runs("agent:test", "session_1")

    assert state == {"status": "completed", "result": 42}
    assert runs[0]["run_id"] == "run_1"
    assert runs[0]["status"] == "completed"

    store.close()


def test_sqlite_checkpoint_store_supports_worker_threads(tmp_path):
    store = SQLiteCheckpointStore(path=str(tmp_path / "checkpoints.sqlite3"))

    def save(index: int) -> None:
        store.save_state(
            "agent:test",
            "session_1",
            f"run_{index}",
            {"status": "completed", "result": index},
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(save, range(20)))

    assert len(store.list_runs("agent:test", "session_1")) == 20
    assert store.load_state("agent:test", "session_1", "run_7")["result"] == 7
    store.close()


def test_sqlite_checkpoint_store_normalizes_messages(tmp_path):
    path = tmp_path / "checkpoints.sqlite3"
    store = SQLiteCheckpointStore(path=str(path))
    messages = _message_state("run_1")

    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "messages": messages},
    )

    state = store.load_state("agent:test", "session_1", "run_1")
    row = store._conn.execute(
        "SELECT state FROM checkpoints WHERE namespace=? AND thread_id=? AND run_id=?",
        ("agent:test", "session_1", "run_1"),
    ).fetchone()
    item_count = store._conn.execute(
        "SELECT COUNT(*) FROM checkpoint_message_items "
        "WHERE namespace=? AND thread_id=?",
        ("agent:test", "session_1"),
    ).fetchone()[0]

    assert state == {"status": "completed", "messages": messages}
    assert '"messages"' not in row[0]
    assert store._item_ref(messages["items"][0]) in row[0]
    assert item_count == 1
    assert len(store._item_ref(messages["items"][0])) == len("item_") + 32

    store.close()


def test_sqlite_checkpoint_store_reuses_thread_items_across_runs(tmp_path):
    path = tmp_path / "checkpoints.sqlite3"
    store = SQLiteCheckpointStore(path=str(path))
    first = _message_state("run_1", "first")
    second = _message_state("run_2", "second")
    second["items"] = [first["items"][0], second["items"][0]]

    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "messages": first},
    )
    store.save_state(
        "agent:test",
        "session_1",
        "run_2",
        {"status": "completed", "messages": second},
    )

    state = store.load_state("agent:test", "session_1", "run_2")
    item_count = store._conn.execute(
        "SELECT COUNT(*) FROM checkpoint_message_items "
        "WHERE namespace=? AND thread_id=?",
        ("agent:test", "session_1"),
    ).fetchone()[0]

    assert state == {"status": "completed", "messages": second}
    assert item_count == 2

    store.close()


def test_sqlite_checkpoint_store_preserves_forked_items(tmp_path):
    path = tmp_path / "checkpoints.sqlite3"
    store = SQLiteCheckpointStore(path=str(path))
    first = _message_state("run_1", "first")
    second = _message_state("run_2", "second")
    fork = _message_state("run_3", "alternate")
    second["items"] = [first["items"][0], second["items"][0]]
    fork["items"] = [first["items"][0], fork["items"][0]]

    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "messages": first},
    )
    store.save_state(
        "agent:test",
        "session_1",
        "run_2",
        {"status": "completed", "messages": second},
    )
    store.save_state(
        "agent:test",
        "session_1",
        "run_3",
        {"status": "completed", "messages": fork},
    )
    item_count = store._conn.execute(
        "SELECT COUNT(*) FROM checkpoint_message_items "
        "WHERE namespace=? AND thread_id=?",
        ("agent:test", "session_1"),
    ).fetchone()[0]

    assert store.load_state("agent:test", "session_1", "run_2")["messages"] == second
    assert store.load_state("agent:test", "session_1", "run_3")["messages"] == fork
    assert item_count == 3

    store.close()


def test_sqlite_checkpoint_store_raises_on_missing_item_ref(tmp_path):
    path = tmp_path / "checkpoints.sqlite3"
    store = SQLiteCheckpointStore(path=str(path))
    messages = _message_state("run_1")
    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "messages": messages},
    )
    store._conn.execute("DELETE FROM checkpoint_message_items")
    store._conn.commit()

    with pytest.raises(ValueError, match="missing or corrupted"):
        store.load_state("agent:test", "session_1", "run_1")

    store.close()


def test_sqlite_checkpoint_store_cleans_orphaned_items(tmp_path):
    path = tmp_path / "checkpoints.sqlite3"
    store = SQLiteCheckpointStore(path=str(path))
    first = _message_state("run_1", "first")
    second = _message_state("run_2", "second")
    second["items"] = [first["items"][0], second["items"][0]]
    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "messages": first},
    )
    store.save_state(
        "agent:test",
        "session_1",
        "run_2",
        {"status": "completed", "messages": second},
    )

    assert store.delete_run("agent:test", "session_1", "run_2") is True
    item_count = store._conn.execute(
        "SELECT COUNT(*) FROM checkpoint_message_items "
        "WHERE namespace=? AND thread_id=?",
        ("agent:test", "session_1"),
    ).fetchone()[0]

    assert item_count == 1
    assert store.load_state("agent:test", "session_1", "run_1")["messages"] == first

    store.close()


def test_sqlite_checkpoint_store_forks_run_to_new_thread(tmp_path):
    path = tmp_path / "checkpoints.sqlite3"
    store = SQLiteCheckpointStore(path=str(path))
    first = _message_state("run_1", "first")
    store.save_state(
        "agent:test",
        "session_1",
        "run_1",
        {"status": "completed", "messages": first},
    )

    forked = store.fork_run(
        "agent:test",
        "session_1",
        "run_1",
        target_thread_id="session_fork",
        target_run_id="run_fork",
        status="running",
    )
    item_count = store._conn.execute(
        "SELECT COUNT(*) FROM checkpoint_message_items "
        "WHERE namespace=? AND thread_id=?",
        ("agent:test", "session_fork"),
    ).fetchone()[0]

    assert forked["status"] == "running"
    assert forked["messages"] == first
    assert store.load_state("agent:test", "session_fork", "run_fork") == forked
    assert item_count == 1

    store.close()


def test_store_factory_creates_checkpoint_stores(tmp_path):
    memory_store = Store.checkpoint("in_memory")
    sqlite_store = Store.checkpoint(
        "sqlite", path=str(tmp_path / "checkpoints.sqlite3")
    )

    assert isinstance(memory_store, InMemoryCheckpointStore)
    assert isinstance(sqlite_store, SQLiteCheckpointStore)

    sqlite_store.close()
