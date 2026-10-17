"""Recover actual request and conversation reasoning from durable checkpoints."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import msgspec
import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.data.stores.base import AsyncCheckpointStore
from msgflux.models.configuration import ModelConfiguration
from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.nn import Agent
from msgflux.runtime import ExecutionScope
from msgflux.runtime.agent_inbox import AgentInbox, InMemoryAgentInboxStore
from tests.models._chat_transport import EndpointMockTransport
from tests.test_manual_compaction import paused_agent


def _provider(monkeypatch, *, model_id="gpt-6-astra", effort="low"):
    monkeypatch.setenv("OPENAI_API_KEY", "test-model-configuration-key")
    raw = SimpleNamespace(
        id="resp_configuration",
        status="completed",
        incomplete_details=None,
        usage=None,
        output=[
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
    )
    client, async_client = MagicMock(), MagicMock()
    client.responses.create.return_value = raw
    async_client.responses.create = AsyncMock(return_value=raw)
    async_client.close = AsyncMock()
    transport = EndpointMockTransport(client, async_client)
    model = OpenAIChatCompletion(
        model_id=model_id,
        api_mode="responses",
        reasoning_effort=effort,
        chat_transport=transport,
        retry=False,
    )
    return model, client, async_client


def _agent(model, store):
    return Agent(
        name="configuration-agent",
        model=model,
        checkpoint_store=store,
        agent_inbox=AgentInbox(store=InMemoryAgentInboxStore()),
    )


def _scope(run_id="first", thread_id="thread"):
    return ExecutionScope(thread_id=thread_id, run_id=run_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("model_id,native", [("gpt-6-astra", True), ("gpt-5.6", False)])
async def test_sqlite_reopen_distinguishes_request_baseline_from_conversation_update(
    tmp_path, monkeypatch, asynchronous, model_id, native
):
    database = tmp_path / "checkpoints.sqlite3"
    store = SQLiteCheckpointStore(database)
    model, client, async_client = _provider(monkeypatch, model_id=model_id)
    agent = _agent(model, store)
    try:
        if asynchronous:
            assert await agent.acall("first", scope=_scope()) == "done"
        else:
            assert agent("first", scope=_scope()) == "done"
        agent.agent_inbox.set_reasoning_effort("high")
        if asynchronous:
            assert await agent.acall("second", scope=_scope("second")) == "done"
            wire = async_client.responses.create.call_args.kwargs
        else:
            assert agent("second", scope=_scope("second")) == "done"
            wire = client.responses.create.call_args.kwargs
        expected = ModelConfiguration(
            provider="openai",
            model_id=model_id,
            api_mode="responses",
            reasoning_effort="high",
            request_reasoning_effort="low" if native else "high",
            conversation_reasoning_effort="high" if native else None,
        )
        assert wire["reasoning"]["effort"] == ("low" if native else "high")
        assert (
            any(item.get("type") == "configuration_update" for item in wire["input"])
            is native
        )
        assert agent.get_model_configuration(scope=_scope("second")) == expected
        saved = store.load_state("configuration-agent", "thread", "second")
        assert (
            msgspec.convert(saved["model_configuration"], type=ModelConfiguration)
            == expected
        )
    finally:
        await agent.aclose()
        store.close()

    reopened_store = SQLiteCheckpointStore(database)
    fresh_model, _, _ = _provider(monkeypatch, model_id=model_id, effort="medium")
    reopened = _agent(fresh_model, reopened_store)
    try:
        assert reopened.get_model_configuration(scope=_scope("second")) == expected
        assert (
            reopened_store.get_model_configuration("configuration-agent", "thread")
            == expected
        )
        assert (
            reopened_store.get_model_configuration(
                "configuration-agent", "thread", "second"
            )
            == expected
        )
        # Recovery reports history; it does not silently reconfigure the new model.
        assert fresh_model.sampling_run_params["reasoning_effort"] == "medium"
        assert model.sampling_run_params["reasoning_effort"] == "low"
    finally:
        await reopened.aclose()
        reopened_store.close()


@pytest.mark.asyncio
async def test_first_turn_override_is_the_recorded_request_baseline(monkeypatch):
    model, client, _ = _provider(monkeypatch)
    store = InMemoryCheckpointStore()
    agent = _agent(model, store)
    try:
        agent.agent_inbox.set_reasoning_effort("medium")
        agent("first", scope=_scope())
        agent.agent_inbox.set_reasoning_effort("high")
        agent("second", scope=_scope("second"))
        configuration = agent.get_model_configuration(scope=_scope("second"))
        assert configuration.request_reasoning_effort == "medium"
        assert configuration.conversation_reasoning_effort == "high"
        assert configuration.reasoning_effort == "high"
        assert (
            client.responses.create.call_args.kwargs["reasoning"]["effort"] == "medium"
        )
        assert model.sampling_run_params["reasoning_effort"] == "low"
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_known_default_request_reasoning_is_not_unknown_legacy_metadata(
    monkeypatch,
):
    model, _, _ = _provider(monkeypatch, effort=None)
    store = InMemoryCheckpointStore()
    agent = _agent(model, store)
    try:
        agent("first", scope=_scope())
        configuration = agent.get_model_configuration(scope=_scope())
        assert configuration.request_reasoning_effort is None
        assert configuration.conversation_reasoning_effort is None
    finally:
        await agent.aclose()


def _save_response(store, metadata):
    messages = ChatMessages(
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer", "metadata": metadata},
        ]
    )
    state = {"status": "completed", "messages": messages._to_state()}
    store.save_state("configuration-agent", "thread", "first", state)


@pytest.mark.parametrize("sqlite", [False, True])
def test_legacy_model_metadata_does_not_invent_a_request_baseline(tmp_path, sqlite):
    store = (
        SQLiteCheckpointStore(tmp_path / "legacy.sqlite3")
        if sqlite
        else InMemoryCheckpointStore()
    )
    try:
        _save_response(
            store,
            {
                "model": {
                    "provider": "openai",
                    "model_id": "gpt-6-astra",
                    "api_mode": "responses",
                    "reasoning_effort": "high",
                }
            },
        )
        configuration = store.get_model_configuration("configuration-agent", "thread")
        assert configuration.reasoning_effort == "high"
        assert configuration.request_reasoning_effort is msgspec.UNSET
        assert configuration.conversation_reasoning_effort is msgspec.UNSET
        encoded = msgspec.to_builtins(configuration)
        assert "request_reasoning_effort" not in encoded
        assert "conversation_reasoning_effort" not in encoded
    finally:
        if sqlite:
            store.close()


@pytest.mark.parametrize("sqlite", [False, True])
def test_latest_response_without_identity_does_not_suggest_an_older_model(
    tmp_path, sqlite
):
    store = (
        SQLiteCheckpointStore(tmp_path / "unknown.sqlite3")
        if sqlite
        else InMemoryCheckpointStore()
    )
    try:
        messages = ChatMessages(
            [
                {
                    "role": "assistant",
                    "content": "known",
                    "metadata": {"model": {"provider": "openai", "model_id": "old"}},
                },
                {"role": "assistant", "content": "unknown"},
            ]
        )
        store.save_state(
            "configuration-agent",
            "thread",
            "first",
            {"status": "completed", "messages": messages._to_state()},
        )
        assert store.get_model_configuration("configuration-agent", "thread") is None
    finally:
        if sqlite:
            store.close()


@pytest.mark.asyncio
async def test_missing_thread_configuration_queries_do_not_create_resources(tmp_path):
    model = Mock(model_type="chat_completion")
    agent_dir = tmp_path / "not-created"
    agent = Agent(name="configuration-agent", model=model, agent_dir=agent_dir)
    try:
        assert agent.get_model_configuration(scope=_scope()) is None
        assert not agent_dir.exists()
    finally:
        await agent.aclose()
    assert not agent_dir.exists()


def test_manual_compaction_preserves_recorded_model_configuration(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "compaction.sqlite3")
    agent, store, scope = paused_agent(store)
    try:
        configuration = ModelConfiguration(
            provider="openai",
            model_id="gpt-6-astra",
            api_mode="responses",
            reasoning_effort="high",
            request_reasoning_effort="low",
            conversation_reasoning_effort="high",
        )
        state = store.load_state("agent", "thread", "run")
        state["model_configuration"] = msgspec.to_builtins(configuration)
        store.save_state("agent", "thread", "run", state)
        assert agent.compact(scope=scope).status == "completed"
        assert store.get_model_configuration("agent", "thread", "run") == configuration
        assert agent.get_model_configuration(scope=scope) == configuration
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("run_id", [None, "chosen-run"])
async def test_async_store_configuration_uses_native_async_loaders(run_id):
    expected = ModelConfiguration(
        provider="openai",
        model_id="gpt-6-astra",
        reasoning_effort="high",
        request_reasoning_effort="low",
        conversation_reasoning_effort="high",
    )
    state = {"model_configuration": msgspec.to_builtins(expected)}
    store = SimpleNamespace(
        aload_latest_run=AsyncMock(return_value=state),
        aload_state=AsyncMock(return_value=state),
    )
    assert (
        await AsyncCheckpointStore.aget_model_configuration(
            store, "configuration-agent", "thread", run_id
        )
        == expected
    )
    if run_id is None:
        store.aload_latest_run.assert_awaited_once_with("configuration-agent", "thread")
        store.aload_state.assert_not_awaited()
    else:
        store.aload_state.assert_awaited_once_with(
            "configuration-agent", "thread", run_id
        )
        store.aload_latest_run.assert_not_awaited()
