"""Opt-in real Codex reasoning configuration persistence across SQLite reopen.

Run with ``MSGFLUX_LIVE_CODEX_MODEL_CONFIGURATION=1`` and an existing Codex
account. This makes two short requests and never starts a login flow.
"""

import os
from uuid import uuid4

import msgspec
import pytest

import msgflux as mf
from msgflux.accounts import Accounts
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.configuration import ModelConfiguration
from msgflux.models.providers.openai_codex import CodexChatTransport
from msgflux.nn import Agent
from msgflux.runtime import ExecutionScope
from msgflux.runtime.agent_inbox import AgentInbox, InMemoryAgentInboxStore


_OPT_IN = "MSGFLUX_LIVE_CODEX_MODEL_CONFIGURATION"
pytestmark = pytest.mark.skipif(
    os.getenv(_OPT_IN) != "1",
    reason=f"Set {_OPT_IN}=1 to run live Codex model configuration checks",
)


def _model(effort, account_name):
    model_id = os.getenv("MSGFLUX_LIVE_CODEX_MODEL", "gpt-6-luna")
    return mf.Model.chat_completion(
        f"openai-codex/{model_id}",
        api_mode="responses",
        reasoning_effort=effort,
        credential_resolver=Accounts().resolver(
            "openai-codex", account_name=account_name
        ),
        chat_transport=CodexChatTransport(timeout=120, max_retries=0),
        retry=False,
    )


def _agent(model, store, namespace):
    return Agent(
        name=namespace,
        model=model,
        checkpoint_store=store,
        agent_inbox=AgentInbox(store=InMemoryAgentInboxStore()),
        system_prompt="Answer briefly. Do not use tools.",
    )


@pytest.mark.asyncio
async def test_codex_reasoning_configuration_survives_sqlite_reopen(tmp_path):
    account_name = os.getenv("MSGFLUX_LIVE_SERVICE_ACCOUNT", "work")
    namespace = "codex_configuration_live"
    thread_id = f"configuration-{uuid4().hex}"
    database = tmp_path / "model-configuration.sqlite3"
    first_scope = ExecutionScope(
        namespace=namespace,
        thread_id=thread_id,
        run_id="first",
        account_name=account_name,
    )
    second_scope = ExecutionScope(
        namespace=namespace,
        thread_id=thread_id,
        run_id="second",
        account_name=account_name,
    )
    model = _model("low", account_name)
    store = SQLiteCheckpointStore(database)
    agent = _agent(model, store, namespace)
    try:
        first_answer = await agent.acall("Reply with OK.", scope=first_scope)
        assert isinstance(first_answer, str) and first_answer.strip()
        first_configuration = agent.get_model_configuration(scope=first_scope)
        assert first_configuration.request_reasoning_effort == "low"
        assert first_configuration.conversation_reasoning_effort is None
        assert first_configuration.reasoning_effort == "low"

        agent.agent_inbox.set_reasoning_effort("high")
        # Unsupported native updates must fail visibly rather than retrying with
        # another provider or silently changing the request's initial reasoning.
        second_answer = await agent.acall(
            "What is two plus two? Reply with the number only.", scope=second_scope
        )
        assert isinstance(second_answer, str) and second_answer.strip()
        native_update = model.supports_reasoning_update()
        expected = ModelConfiguration(
            provider="openai-codex",
            model_id=model.model_id,
            api_mode="responses",
            reasoning_effort="high",
            request_reasoning_effort="low" if native_update else "high",
            conversation_reasoning_effort="high" if native_update else None,
        )
        assert agent.get_model_configuration(scope=second_scope) == expected
        snapshot = store.load_state(namespace, thread_id, "second")[
            "model_configuration"
        ]
        assert msgspec.convert(snapshot, type=ModelConfiguration) == expected
        assert model.sampling_run_params["reasoning_effort"] == "low"
    finally:
        await agent.aclose()
        store.close()
        await model.aclose()

    fresh_model = _model("medium", account_name)
    reopened_store = SQLiteCheckpointStore(database)
    reopened = _agent(fresh_model, reopened_store, namespace)
    try:
        # Recovery alone makes no model call and does not override a constructor.
        assert reopened.get_model_configuration(scope=second_scope) == expected
        assert reopened_store.get_model_configuration(namespace, thread_id) == expected
        assert (
            reopened_store.get_model_configuration(namespace, thread_id, "second")
            == expected
        )
        assert fresh_model.sampling_run_params["reasoning_effort"] == "medium"
    finally:
        await reopened.aclose()
        reopened_store.close()
        await fresh_model.aclose()
