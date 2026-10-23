"""Latest-run selection and hydration share one SQLite read snapshot."""

from unittest.mock import Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.nn import Agent
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    ServiceRecoveryRequiredError,
)


@pytest.fixture
def stores(tmp_path):
    path = tmp_path / "checkpoints.sqlite3"
    reader, writer = SQLiteCheckpointStore(path), SQLiteCheckpointStore(path)
    try:
        yield reader, writer
    finally:
        reader.close()
        writer.close()


def _commit(store, namespace, run):
    return store.commit_state(
        namespace,
        "thread",
        run,
        {
            "status": "paused",
            "scope": {"run_id": run},
            "messages": ChatMessages(
                [{"role": "user", "content": run}],
                namespace=namespace,
                thread_id="thread",
            )._to_state(),
        },
        expected_revision=0,
    ).state


def _delete_selected(reader, writer, namespace, monkeypatch):
    original = reader.list_runs

    def list_then_delete(*args, **kwargs):
        rows = original(*args, **kwargs)
        assert rows[0]["run_id"] == "latest"
        assert writer.delete_run(namespace, "thread", "latest")
        return rows

    monkeypatch.setattr(reader, "list_runs", list_then_delete)


def test_latest_snapshot_survives_independent_delete(stores, monkeypatch):
    reader, writer = stores
    older = _commit(reader, "agent", "older")
    latest = _commit(reader, "agent", "latest")
    with monkeypatch.context() as patch:
        _delete_selected(reader, writer, "agent", patch)
        assert reader.load_latest_run("agent", "thread") == latest
    assert reader.load_latest_run("agent", "thread") == older
    assert not reader._conn.in_transaction


def test_latest_delete_cannot_skip_service_recovery_guard(stores, monkeypatch):
    reader, writer = stores
    agent = Agent(
        name="latest-agent",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=reader,
    )
    session = AgentSession(agent)
    _commit(reader, session.namespace, "older")
    _commit(reader, session.namespace, "latest")
    with monkeypatch.context() as patch:
        _delete_selected(reader, writer, session.namespace, patch)
        with pytest.raises(ServiceRecoveryRequiredError, match="explicit recovery"):
            AgentService._validate_new_input(session, "thread")
    assert reader.load_latest_run(session.namespace, "thread")["status"] == "paused"


def test_latest_read_releases_transaction_after_base_exception(stores, monkeypatch):
    class StopRead(BaseException):
        pass

    reader, writer = stores
    _commit(reader, "agent", "latest")
    original = reader.list_runs
    failure = StopRead("selection interrupted")

    def stop_after_selection(*args, **kwargs):
        original(*args, **kwargs)
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(reader, "list_runs", stop_after_selection)
        with pytest.raises(StopRead) as caught:
            reader.load_latest_run("agent", "thread")
        assert caught.value is failure
    assert not reader._conn.in_transaction
    assert writer.delete_run("agent", "thread", "latest")
    assert reader.load_latest_run("agent", "thread") is None
