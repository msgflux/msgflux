import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import (
    CheckpointConflictError,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from msgflux.runtime.agent_run import AgentRun, agent_run_context, get_agent_run
from msgflux.runtime.context_scopes import (
    ContextScopeConflictError,
    ContextScopeController,
)


@pytest.mark.parametrize("store_factory", [InMemoryCheckpointStore])
def test_checkpoint_commit_is_revision_checked_and_atomic(store_factory):
    store = store_factory()
    first = store.commit_state(
        "agent", "thread", "run", {"status": "running"},
        expected_revision=0,
        event={"event_type": "scope.open"},
        branch_id="root",
    )
    assert first.revision == 1
    assert store.load_events("agent", "thread", "run") == [
        {"event_type": "scope.open"}
    ]
    with pytest.raises(CheckpointConflictError):
        store.commit_state(
            "agent", "thread", "run", {"status": "running"},
            expected_revision=0,
            event={"event_type": "should.not.persist"},
        )
    assert len(store.load_events("agent", "thread", "run")) == 1


def test_agent_run_round_trips_extension_and_budget_state():
    run = AgentRun(
        namespace="agent", thread_id="thread", run_id="run", revision=3,
        branch_id="review", budgets={"tool_turns": 2},
    )
    run.set_extension("context", {"active": "review"})
    restored = AgentRun.from_durable_state(run.durable_state())
    assert restored.durable_state() == run.durable_state()
    with agent_run_context(restored):
        assert get_agent_run() is restored
    assert get_agent_run() is None


