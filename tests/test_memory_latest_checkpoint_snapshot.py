"""A competing thread cannot split latest-run selection from hydration."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.nn import Agent
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    ServiceRecoveryRequiredError,
)


@pytest.mark.parametrize("consumer", ["snapshot", "service_guard"])
def test_latest_read_keeps_snapshot_across_competing_delete(monkeypatch, consumer):
    store = InMemoryCheckpointStore()
    agent = Agent(
        name="memory-latest",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=store,
    )
    session = AgentSession(agent)
    for run in ["older", "latest"]:
        latest = store.commit_state(
            session.namespace,
            "thread",
            run,
            {
                "status": "paused",
                "scope": {"run_id": run},
                "messages": ChatMessages(
                    [{"role": "user", "content": run}],
                    namespace=session.namespace,
                    thread_id="thread",
                )._to_state(),
            },
            expected_revision=0,
        ).state
    original = store.list_runs
    attempts = []

    def try_delete():
        # A zero-wait writer distinguishes a held snapshot from a read gap.
        # It retries after the read rather than blocking the reader's hook.
        if not store._lock.acquire(blocking=False):
            return False
        try:
            assert store.delete_run(session.namespace, "thread", "latest")
            return True
        finally:
            store._lock.release()

    with ThreadPoolExecutor(max_workers=1) as writer:

        def list_then_delete(*args, **kwargs):
            rows = original(*args, **kwargs)
            assert rows[0]["run_id"] == "latest"
            attempts.append(writer.submit(try_delete).result(timeout=2))
            return rows

        with monkeypatch.context() as patch:
            patch.setattr(store, "list_runs", list_then_delete)
            if consumer == "snapshot":
                assert store.load_latest_run(session.namespace, "thread") == latest
            else:
                with pytest.raises(
                    ServiceRecoveryRequiredError, match="explicit recovery"
                ):
                    AgentService._validate_new_input(session, "thread")
        assert attempts == [False]
        assert writer.submit(
            store.delete_run, session.namespace, "thread", "latest"
        ).result(timeout=2)
    assert (
        store.load_latest_run(session.namespace, "thread")["scope"]["run_id"] == "older"
    )
