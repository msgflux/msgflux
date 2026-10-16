"""Manual compaction preserves durable paused/failed execution state."""

import asyncio

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.data.stores.base import CheckpointConflictError
from msgflux.nn import Agent, CompactionExtension
from msgflux.nn.hooks import Hook
from msgflux.runtime import ExecutionScope
from msgflux.runtime.compaction import (
    CompactionBusyError,
)
from tests.test_agent_extensions import _CompactingModel


def paused_agent(store=None, *, status="paused", history=True, model=None):
    store = store if store is not None else InMemoryCheckpointStore()
    messages = ChatMessages(thread_id="thread", namespace="agent")
    if history:
        messages.begin_turn(turn_id="previous")
        messages.add_user("An important previous request")
        messages.add_assistant("An important decision")
        messages.end_turn()
    messages.begin_turn(turn_id="run")
    messages.add_user("Unfinished current request")
    scope = ExecutionScope(thread_id="thread", run_id="run", namespace="agent")
    state = {
        "schema_version": 1,
        "status": status,
        "scope": scope.to_dict(),
        "messages": messages._to_state(),
        "model_preference": None,
        "runtime": {
            "schema_version": 1,
            "namespace": "agent",
            "thread_id": "thread",
            "run_id": "run",
            "branch_id": "root",
            "budgets": {"tool_calls": 3},
            "extensions": {"counter": {"requests": 7}},
        },
        "metadata": {"namespace": "agent", "sentinel": "preserved"},
        "task_result": {"value": "preserved"},
    }
    store.save_state("agent", "thread", "run", state)
    agent = Agent(
        name="agent",
        model=model or _CompactingModel(),
        checkpoint_store=store,
        system_prompt="Preserve decisions",
        extensions=[CompactionExtension()],
    )
    return agent, store, scope


@pytest.mark.parametrize("status", ["paused", "failed"])
@pytest.mark.parametrize("sqlite", [False, True])
def test_manual_compaction_preserves_run_state_and_canonical_history(
    tmp_path, status, sqlite
):
    store = (
        SQLiteCheckpointStore(tmp_path / "checkpoint.sqlite3")
        if sqlite
        else InMemoryCheckpointStore()
    )
    agent, store, scope = paused_agent(store, status=status)
    before = store.load_state("agent", "thread", "run")
    try:
        result = agent.compact(scope=scope)
        after = store.load_state("agent", "thread", "run")
        assert result.status == "completed"
        assert result.reason == "manual"
        assert result.run_id == "run" and result.thread_id == "thread"
        assert after["status"] == status
        for key in ("scope", "model_preference", "metadata", "task_result"):
            assert after[key] == before[key]
        assert after["runtime"]["extensions"] == before["runtime"]["extensions"]
        assert after["runtime"]["budgets"] == before["runtime"]["budgets"]
        assert after["messages"]["items"][:-1] == before["messages"]["items"]
        assert after["messages"]["items"][-1]["item_id"] == result.operation_id
        compacted = ChatMessages()
        compacted._hydrate_state(after["messages"])
        assert compacted.to_chatml() == [
            {"role": "system", "content": "SUMMARY"},
            {"role": "user", "content": "Unfinished current request"},
        ]
        assert (
            after["_checkpoint"]["revision"]
            == before.get("_checkpoint", {}).get("revision", 0) + 1
        )
        assert agent.compact(scope=scope).reason == "already_compacted"
        assert len(agent.model.compaction_calls) == 1
    finally:
        if sqlite:
            store.close()


@pytest.mark.asyncio
async def test_async_manual_compaction_without_threshold_or_capacity():
    model = _CompactingModel()
    model.context_capacity = None
    agent, store, scope = paused_agent(model=model)
    result = await agent.acompact(scope=ExecutionScope(thread_id=scope.thread_id))
    assert result.status == "completed"
    assert store.load_state("agent", "thread", "run")["status"] == "paused"


@pytest.mark.parametrize("status", ["completed", "interrupted", "running", "streaming"])
def test_manual_compaction_rejects_ineligible_state(status):
    agent, store, scope = paused_agent(status=status)
    before = store.load_state("agent", "thread", "run")
    with pytest.raises((ValueError, CompactionBusyError)):
        agent.compact(scope=scope)
    assert store.load_state("agent", "thread", "run") == before
    assert not agent.model.compaction_calls


def test_manual_compaction_skips_missing_completed_turn_and_honors_veto():
    agent, _store, scope = paused_agent(history=False)
    assert agent.compact(scope=scope).reason == "no_completed_turn"
    agent, _store, scope = paused_agent()
    from dataclasses import replace

    Hook(
        event="before_compaction", handler=lambda event: replace(event, action="skip")
    ).register(agent)
    assert agent.compact(scope=scope).reason == "policy_skipped"
    assert not agent.model.compaction_calls


def test_scope_and_checkpoint_are_required_and_only_latest_run_is_eligible():
    agent, _store, scope = paused_agent()
    with pytest.raises(ValueError, match="thread_id"):
        agent.compact(scope=ExecutionScope())
    with pytest.raises(TypeError):
        agent.compact(scope=scope, messages=ChatMessages())
    with pytest.raises(ValueError, match="latest"):
        agent.compact(scope=scope.with_overrides(run_id="older"))
    with pytest.raises(ValueError, match="namespace"):
        agent.compact(scope=scope.with_overrides(namespace="other"))


def test_provider_failure_and_revision_conflict_do_not_publish_compaction(monkeypatch):
    agent, store, scope = paused_agent()
    before = store.load_state("agent", "thread", "run")

    def fail(*_a, **_kw):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(agent.model, "compact_context", fail)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        agent.compact(scope=scope)
    assert store.load_state("agent", "thread", "run") == before
    agent, store, scope = paused_agent()
    original = agent.model.compact_context

    def race(*args, **kwargs):
        view = original(*args, **kwargs)
        current = store.load_state("agent", "thread", "run")
        current["metadata"]["concurrent"] = True
        store.commit_state(
            "agent",
            "thread",
            "run",
            current,
            expected_revision=current.get("_checkpoint", {}).get("revision", 0),
        )
        return view

    monkeypatch.setattr(agent.model, "compact_context", race)
    with pytest.raises(CheckpointConflictError):
        agent.compact(scope=scope)
    latest = store.load_state("agent", "thread", "run")
    assert latest["metadata"]["concurrent"] is True
    assert not any(
        item.get("type") == "compaction" for item in latest["messages"]["items"]
    )


@pytest.mark.asyncio
async def test_active_operation_blocks_both_compaction_and_normal_agent_calls(
    monkeypatch,
):
    agent, _store, scope = paused_agent()
    entered, release = asyncio.Event(), asyncio.Event()
    original = agent.model.acompact_context

    async def slow(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(agent.model, "acompact_context", slow)
    task = asyncio.create_task(agent.acompact(scope=scope))
    await asyncio.wait_for(entered.wait(), timeout=2)
    try:
        with pytest.raises(CompactionBusyError):
            await agent.acompact(scope=scope)
        with pytest.raises(CompactionBusyError):
            await agent.acall("Do not run", scope=scope)
    finally:
        release.set()
        assert (await task).status == "completed"


@pytest.mark.asyncio
async def test_cancelled_compaction_preserves_checkpoint_and_releases_fence(
    monkeypatch,
):
    agent, store, scope = paused_agent()
    before = store.load_state("agent", "thread", "run")
    entered = asyncio.Event()
    original = agent.model.acompact_context

    async def slow(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(agent.model, "acompact_context", slow)
    task = asyncio.create_task(agent.acompact(scope=scope))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.load_state("agent", "thread", "run") == before
    monkeypatch.setattr(agent.model, "acompact_context", original)
    assert (await agent.acompact(scope=scope)).status == "completed"


def test_explicit_manual_compaction_does_not_require_automatic_extension():
    agent, _store, scope = paused_agent()
    agent.remove_extension("compaction")
    assert agent.compact(scope=scope).status == "completed"


@pytest.mark.asyncio
async def test_nonexistent_managed_thread_does_not_create_files(tmp_path):
    agent = Agent(name="agent", model=_CompactingModel(), agent_dir=tmp_path / "agent")
    try:
        with pytest.raises(ValueError, match="existing checkpoint"):
            await agent.acompact(scope=ExecutionScope(thread_id="absent"))
        assert not (tmp_path / "agent").exists()
    finally:
        await agent.aclose()


@pytest.mark.parametrize("uncertain", ["executing", "background"])
def test_uncertain_work_blocks_compaction_without_calling_provider(uncertain):
    from msgflux.runtime.compaction import CompactionRecoveryRequiredError

    agent, store, scope = paused_agent()
    state = store.load_state("agent", "thread", "run")
    if uncertain == "executing":
        state["runtime"]["extensions"]["pending_approvals"] = {
            "schema_version": 1,
            "phase": "executing",
            "requests": {"call": "approval"},
        }
        store.save_state("agent", "thread", "run", state)
    else:
        agent.tool_library.get_task_store().create(
            "background", metadata={"thread_id": "thread", "run_id": "run"}
        )
    before = store.load_state("agent", "thread", "run")
    with pytest.raises(CompactionRecoveryRequiredError):
        agent.compact(scope=scope)
    assert store.load_state("agent", "thread", "run") == before
    assert not agent.model.compaction_calls


@pytest.mark.asyncio
async def test_watch_observes_manual_start_and_end_with_saved_run_identity():
    agent, _store, scope = paused_agent()
    async with agent.watch(scope.thread_id) as watcher:
        await agent.acompact(scope=scope)
        start = await asyncio.wait_for(anext(watcher), timeout=2)
        end = await asyncio.wait_for(anext(watcher), timeout=2)
        assert [start.type, end.type] == ["compaction.start", "compaction.end"]
        assert start.run_id == end.run_id == "run"
        assert end.data["status"] == "completed"


def test_manual_compaction_hook_cannot_mutate_saved_canonical_messages():
    agent, store, scope = paused_agent()
    before = store.load_state("agent", "thread", "run")

    def edit_hook(event):
        event.messages.add_user("Hook-local content")
        return event

    Hook(event="before_compaction", handler=edit_hook).register(agent)
    result = agent.compact(scope=scope)
    assert result.status == "completed"
    after = store.load_state("agent", "thread", "run")
    assert after["messages"]["items"][:-1] == before["messages"]["items"]


def test_preaborted_manual_compaction_does_not_contact_provider():
    from msgflux.exceptions import AbortRequestedError
    from msgflux.runtime.abort import AbortSignal

    agent, store, scope = paused_agent()
    before = store.load_state("agent", "thread", "run")
    signal = AbortSignal()
    signal.abort("Operator cancelled")
    with pytest.raises(AbortRequestedError):
        agent.compact(scope=scope.with_overrides(abort_signal=signal))
    assert store.load_state("agent", "thread", "run") == before
    assert not agent.model.compaction_calls


@pytest.mark.asyncio
async def test_managed_agent_reopens_thread_and_compaction_survives_resource_release(
    tmp_path,
):
    from msgflux.runtime.agent_resources import AgentResources

    _seed_agent, seed_store, scope = paused_agent()
    resources = AgentResources(tmp_path / "agent")
    binding = resources.bind("thread", namespace="agent")
    binding.checkpoint_store.save_state(
        "agent", "thread", "run", seed_store.load_state("agent", "thread", "run")
    )
    binding.close()
    agent = Agent(name="agent", model=_CompactingModel(), agent_dir=resources.agent_dir)
    try:
        result = await agent.acompact(scope=scope)
        assert result.status == "completed"
    finally:
        await agent.aclose()
    reopened = Agent(
        name="agent", model=_CompactingModel(), agent_dir=resources.agent_dir
    )
    try:
        result = await reopened.acompact(scope=scope)
        assert result.reason == "already_compacted"
        assert not reopened.model.compaction_calls
    finally:
        await reopened.aclose()
