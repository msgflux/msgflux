"""Durable account selection on admitted runs."""

import asyncio
import sqlite3
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_scope,
)
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.records import ServiceConflictError, ServiceThread


def _response():
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("ok")
    response.reasoning = None
    return response


def _agent(name, seen):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name=name, model=model)

    async def forward(*_args, **_kwargs):
        seen.append(get_execution_scope().account_name)
        await asyncio.sleep(0.01)
        return _response()

    agent.generator.aforward = AsyncMock(side_effect=forward)
    return agent


@pytest.mark.asyncio
async def test_concurrent_threads_keep_their_admitted_accounts():
    seen = []
    service = AgentService(store=SQLiteServiceStore(), default_account="default")
    service.register("agent", lambda _thread: AgentSession(_agent("parallel", seen)))
    first = await service.open_thread("agent", thread_id="account-a")
    second = await service.open_thread("agent", thread_id="account-b")
    try:
        await asyncio.gather(
            service.prompt(
                first.thread_id, "hello", request_id="one", account_name="work"
            ),
            service.prompt(
                second.thread_id, "hello", request_id="two", account_name="personal"
            ),
        )
        await asyncio.gather(
            service.wait(first.thread_id, "one"),
            service.wait(second.thread_id, "two"),
        )
        assert sorted(seen) == ["personal", "work"]
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_each_new_run_selects_account_and_idempotency_preserves_it():
    seen = []
    service = AgentService(store=SQLiteServiceStore(), default_account="default")
    service.register("agent", lambda _thread: AgentSession(_agent("sequential", seen)))
    thread = await service.open_thread("agent", thread_id="account-sequential")
    try:
        first = await service.prompt(thread.thread_id, "one", request_id="same")
        await service.wait(thread.thread_id, "same")
        duplicate = await service.prompt(thread.thread_id, "one", request_id="same")
        assert duplicate.run_id == first.run_id
        with pytest.raises(ServiceConflictError):
            await service.prompt(
                thread.thread_id, "one", request_id="same", account_name="work"
            )

        second = await service.prompt(
            thread.thread_id, "two", request_id="next", account_name="work"
        )
        await service.wait(thread.thread_id, "next")
        assert service.store.get(thread.thread_id, "same").account_name == "default"
        assert service.store.get(thread.thread_id, "next").account_name == "work"
        assert seen == ["default", "work"]
        assert first.run_id != second.run_id
    finally:
        await service.aclose()


def test_execution_scope_round_trip_accepts_legacy_dict():
    scope = ExecutionScope(thread_id="thread", run_id="run", account_name="work")
    assert ExecutionScope.from_dict(scope.to_dict()).account_name == "work"
    legacy = ExecutionScope.from_dict(
        {"thread_id": "thread", "namespace": "legacy", "run_id": "old"}
    )
    assert legacy.account_name is None


def test_legacy_admission_migration_keeps_account_unknown(tmp_path):
    path = tmp_path / "legacy-service.sqlite3"
    store = SQLiteServiceStore(path)
    store.bind_thread(ServiceThread("legacy-account", "agent"))
    store.admit("legacy-account", "old-request", "old", "legacy")
    store.close()

    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE service_admissions DROP COLUMN account_name")

    reopened = SQLiteServiceStore(path)
    try:
        recovered = reopened.get("legacy-account", "old-request")
        assert recovered.account_name is None
        assert (
            reopened.admit(
                "legacy-account",
                "old-request",
                "old",
                "legacy",
                "new-default",
                check_account=False,
            )
            == recovered
        )
    finally:
        reopened.close()


def test_scope_factory_cannot_override_admitted_account():
    session = AgentSession(
        _agent("scope-factory", []),
        scope_factory=lambda scope: scope.with_overrides(account_name="wrong"),
    )
    scope = session.scope("thread", run_id="run", account_name="chosen")
    assert scope.account_name == "chosen"


def test_nested_scope_inherits_account_and_cannot_switch_it():
    with execution_context(scope=ExecutionScope(account_name="work")):
        with execution_context(scope=ExecutionScope(thread_id="child")):
            assert get_execution_scope().account_name == "work"
        with pytest.raises(ValueError, match="cannot change its account"):
            with execution_context(scope=ExecutionScope(account_name="other")):
                pass
