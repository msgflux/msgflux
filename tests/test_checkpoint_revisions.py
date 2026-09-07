import pytest

from msgflux.data.stores import (
    CheckpointConflictError,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from msgflux.runtime.agent_run import AgentRun, agent_run_context, get_agent_run


@pytest.mark.parametrize("store_factory", [InMemoryCheckpointStore])
def test_checkpoint_commit_is_revision_checked_and_atomic(store_factory):
    store = store_factory()
    first = store.commit_state(
        "agent",
        "thread",
        "run",
        {"status": "running"},
        expected_revision=0,
        event={"event_type": "scope.open"},
        branch_id="root",
    )
    assert first.revision == 1
    assert store.load_events("agent", "thread", "run") == [{"event_type": "scope.open"}]
    with pytest.raises(CheckpointConflictError):
        store.commit_state(
            "agent",
            "thread",
            "run",
            {"status": "running"},
            expected_revision=0,
            event={"event_type": "should.not.persist"},
        )
    assert len(store.load_events("agent", "thread", "run")) == 1


def test_agent_run_round_trips_extension_and_budget_state():
    run = AgentRun(
        namespace="agent",
        thread_id="thread",
        run_id="run",
        revision=3,
        branch_id="review",
        budgets={"tool_turns": 2},
    )
    run.set_extension("context", {"active": "review"})
    restored = AgentRun.from_durable_state(run.durable_state())
    assert restored.durable_state() == run.durable_state()
    with agent_run_context(restored):
        assert get_agent_run() is restored
    assert get_agent_run() is None


def test_stale_commit_does_not_create_empty_in_memory_run():
    store = InMemoryCheckpointStore()
    with pytest.raises(CheckpointConflictError):
        store.commit_state("agent", "thread", "missing", {}, expected_revision=1)
    assert store.load_state("agent", "thread", "missing") is None


def test_future_agent_run_schema_is_rejected():
    with pytest.raises(ValueError, match="Unsupported AgentRun schema"):
        AgentRun.from_durable_state({"schema_version": 2})


@pytest.mark.parametrize("store_factory", [InMemoryCheckpointStore, SQLiteCheckpointStore])
def test_commit_without_extension_state_preserves_existing_extensions(
    store_factory, tmp_path
):
    kwargs = {"path": str(tmp_path / "checkpoint.sqlite3")} if store_factory is SQLiteCheckpointStore else {}
    store = store_factory(**kwargs)
    try:
        store.commit_state(
            "agent",
            "thread",
            "run",
            {"status": "running"},
            expected_revision=0,
            extension_state={"budget": {"remaining": 2}},
        )
        store.commit_state(
            "agent",
            "thread",
            "run",
            {"status": "running"},
            expected_revision=1,
        )
        assert store.load_state("agent", "thread", "run")["_checkpoint"]["extensions"] == {
            "budget": {"remaining": 2}
        }
    finally:
        if isinstance(store, SQLiteCheckpointStore):
            store.close()


@pytest.mark.parametrize("store_factory", [InMemoryCheckpointStore, SQLiteCheckpointStore])
def test_fork_records_namespace_and_source_branch_for_legacy_and_revisioned_state(
    store_factory, tmp_path
):
    kwargs = {"path": str(tmp_path / "fork.sqlite3")} if store_factory is SQLiteCheckpointStore else {}
    store = store_factory(**kwargs)
    try:
        store.save_state(
            "tenant",
            "source",
            "run",
            {
                "status": "completed",
                "_checkpoint": {
                    "schema_version": 1,
                    "revision": 4,
                    "branch_id": "review",
                    "head_item_id": "head-1",
                },
            },
        )
        forked = store.fork_run(
            "tenant",
            "source",
            "run",
            target_thread_id="target",
            target_run_id="fork",
        )
        origin = forked["_checkpoint"]["fork_of"]
        assert origin == {
            "namespace": "tenant",
            "thread_id": "source",
            "run_id": "run",
            "item_id": None,
            "branch_id": "review",
            "head_item_id": "head-1",
        }
        assert forked["_checkpoint"]["branch_id"] == "root"
    finally:
        if isinstance(store, SQLiteCheckpointStore):
            store.close()


@pytest.mark.parametrize("store_factory", [InMemoryCheckpointStore, SQLiteCheckpointStore])
def test_checkpoint_envelope_rejects_future_or_invalid_metadata(store_factory, tmp_path):
    kwargs = {"path": str(tmp_path / "invalid.sqlite3")} if store_factory is SQLiteCheckpointStore else {}
    store = store_factory(**kwargs)
    try:
        with pytest.raises(ValueError, match="schema version"):
            store.save_state("agent", "thread", "run", {"_checkpoint": {"schema_version": 2}})
        with pytest.raises(ValueError, match="revision"):
            store.save_state("agent", "thread", "run", {"_checkpoint": {"revision": -1}})
        with pytest.raises(ValueError, match="branch_id"):
            store.save_state("agent", "thread", "run", {"_checkpoint": {"branch_id": 4}})
    finally:
        if isinstance(store, SQLiteCheckpointStore):
            store.close()


@pytest.mark.parametrize("store_factory", [InMemoryCheckpointStore, SQLiteCheckpointStore])
def test_commit_keeps_runtime_revision_in_step_with_checkpoint(store_factory, tmp_path):
    kwargs = {"path": str(tmp_path / "runtime.sqlite3")} if store_factory is SQLiteCheckpointStore else {}
    store = store_factory(**kwargs)
    try:
        state = {"runtime": {"revision": 0, "branch_id": "root"}}
        original = {"runtime": dict(state["runtime"])}
        commit = store.commit_state(
            "agent", "thread", "run", state, expected_revision=0, branch_id="review"
        )
        assert commit.state["runtime"] == {"revision": 1, "branch_id": "review"}
        assert state == original
        assert store.load_state("agent", "thread", "run")["runtime"] == commit.state["runtime"]
    finally:
        if isinstance(store, SQLiteCheckpointStore):
            store.close()
