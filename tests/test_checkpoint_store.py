from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.data.stores import Store


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


def test_store_factory_creates_checkpoint_stores(tmp_path):
    memory_store = Store.checkpoint("in_memory")
    sqlite_store = Store.checkpoint(
        "sqlite", path=str(tmp_path / "checkpoints.sqlite3")
    )

    assert isinstance(memory_store, InMemoryCheckpointStore)
    assert isinstance(sqlite_store, SQLiteCheckpointStore)

    sqlite_store.close()
