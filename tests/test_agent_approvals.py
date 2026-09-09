import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import (
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from msgflux.data.stores.base import CheckpointConflictError
from msgflux.exceptions import TaskInterruptRequestedError, TaskPauseRequestedError
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime import (
    AgentApprovals,
    ExecutionScope,
    InMemoryApprovalStore,
    SQLiteApprovalStore,
)


def tool_response(mode="chat_completions"):
    calls = ToolCallAggregator(api_mode=mode)
    calls.process(0, "call_1", "lookup", '{"query":"secret"}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def text_response():
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("done")
    return response


def make_agent(checkpoints, approvals, calls):
    def lookup(query: str) -> str:
        """Look up one entry."""
        calls.append(query)
        return "found"

    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="reviewer",
        model=model,
        tools=[lookup],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(approvals, {"lookup": "v1"}, "p1"),
    )
    agent.generator.forward = Mock(side_effect=[tool_response(), text_response()])
    return agent


def scope():
    return ExecutionScope(
        namespace="reviewer", thread_id="thread", run_id="run", principal="user"
    )


@pytest.mark.parametrize("mode", ["chat_completions", "responses"])
def test_pause_resume_without_another_model_request_or_duplicate_calls(mode):
    checkpoint, journal, calls = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)
    agent.generator.forward = Mock(side_effect=[tool_response(mode), text_response()])
    with pytest.raises(TaskPauseRequestedError, match="waiting"):
        agent("lookup", scope=scope())
    assert calls == []
    assert agent.generator.forward.call_count == 1
    state = checkpoint.load_state("reviewer", "thread", "run")
    assert state["status"] == "paused"
    assert state["runtime"]["extensions"]["pending_approvals"]
    with pytest.raises(TaskPauseRequestedError, match="waiting"):
        agent("ignored", scope=scope())
    assert agent.generator.forward.call_count == 1
    record = journal.pending("reviewer", "thread", "run")[0]
    agent.decide_approval(record.request_id, approved=True, decided_by="human")
    assert agent("ignored", scope=scope()) == "done"
    assert calls == ["secret"]
    state = checkpoint.load_state("reviewer", "thread", "run")
    assert "pending_approvals" not in state["runtime"]["extensions"]
    assert (
        sum(item.get("type") == "function_call" for item in state["messages"]["items"])
        == 1
    )
    assert journal.get("reviewer", record.request_id).status == "consumed"


@pytest.mark.asyncio
async def test_async_restart_watcher_and_denial(tmp_path):
    checkpoint_path, approval_path = str(tmp_path / "cp.db"), str(tmp_path / "ap.db")
    checkpoint, journal = (
        SQLiteCheckpointStore(checkpoint_path),
        SQLiteApprovalStore(approval_path),
    )
    calls = []
    agent = make_agent(checkpoint, journal, calls)
    agent.generator.aforward = AsyncMock(side_effect=[tool_response(), text_response()])
    with pytest.raises(TaskPauseRequestedError):
        await agent.acall("lookup", scope=scope())
    checkpoint.close()
    journal.close()
    checkpoint, journal = (
        SQLiteCheckpointStore(checkpoint_path),
        SQLiteApprovalStore(approval_path),
    )
    restored = make_agent(checkpoint, journal, calls)
    restored.generator.aforward = AsyncMock(return_value=text_response())
    async with restored.watch("thread") as watcher:
        assert len(watcher.snapshot.approvals) == 1
        record = watcher.snapshot.approvals[0]
        await restored.adecide_approval(
            record.request_id, approved=False, decided_by="human"
        )
        event = await asyncio.wait_for(watcher.__anext__(), timeout=2)
        assert event.type == "tool.approval_resolved"
    assert await restored.acall("ignored", scope=scope()) == "done"
    assert calls == []
    assert restored.generator.aforward.call_count == 1
    checkpoint.close()
    journal.close()


@pytest.mark.parametrize("change", ["remove", "version", "expired"])
def test_pending_policy_changes_and_expiry(change):
    checkpoint, journal, calls = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)
    with pytest.raises(TaskPauseRequestedError):
        agent("lookup", scope=scope())
    record = journal.pending("reviewer", "thread", "run")[0]
    if change == "remove":
        agent.approvals = None
    elif change == "version":
        agent.approvals = AgentApprovals(journal, {"lookup": "v2"}, "p1")
    else:
        journal._clock = lambda: record.expires_at
        assert agent("ignored", scope=scope()) == "done"
        assert journal.get("reviewer", record.request_id).status == "expired"
        assert calls == []
        return
    with pytest.raises(TaskPauseRequestedError):
        agent("ignored", scope=scope())
    assert agent.generator.forward.call_count == 1
    assert calls == []


def test_crash_after_tool_execution_never_replays_batch(monkeypatch):
    checkpoint, journal, calls = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)
    with pytest.raises(TaskPauseRequestedError):
        agent("lookup", scope=scope())
    record = journal.pending("reviewer", "thread", "run")[0]
    agent.decide_approval(record.request_id, approved=True, decided_by="human")
    original = agent._process_tool_intents

    def crash(*args):
        original(*args)
        raise SystemExit("simulated process loss before result checkpoint")

    monkeypatch.setattr(agent, "_process_tool_intents", crash)
    with pytest.raises(SystemExit):
        agent("ignored", scope=scope())
    assert calls == ["secret"]
    restored = make_agent(checkpoint, journal, calls)
    with pytest.raises(TaskPauseRequestedError, match="reconciliation"):
        restored("ignored", scope=scope())
    restored.generator.forward.assert_not_called()
    assert calls == ["secret"]


def test_entire_batch_waits_before_unprotected_sibling():
    checkpoint, journal, calls = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)

    def public() -> str:
        """Execute an unprotected sibling."""
        calls.append("public")
        return "public"

    agent.tool_library.add(public)
    response = tool_response()
    response.data.process(1, "call_2", "public", "{}")
    agent.generator.forward = Mock(side_effect=[response, text_response()])
    with pytest.raises(TaskPauseRequestedError):
        agent("lookup", scope=scope())
    assert calls == []
    record = journal.pending("reviewer", "thread", "run")[0]
    agent.decide_approval(record.request_id, approved=True, decided_by="human")
    agent("ignored", scope=scope())
    assert sorted(calls) == ["public", "secret"]


def test_final_hook_arguments_cannot_reuse_approval():
    checkpoint, journal, calls = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)
    with pytest.raises(TaskPauseRequestedError):
        agent("lookup", scope=scope())
    record = journal.pending("reviewer", "thread", "run")[0]
    agent.decide_approval(record.request_id, approved=True, decided_by="human")
    Hook(
        event="before_tool",
        handler=lambda payload: replace(payload, arguments={"query": "changed"}),
    ).register(agent)
    assert agent("ignored", scope=scope()) == "done"
    assert calls == []
    assert journal.get("reviewer", record.request_id).status == "approved"


@pytest.mark.asyncio
async def test_live_approval_events_and_async_allow():
    checkpoint, journal, calls = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)
    streamed = ModelStreamResponse(mode="async")
    streamed.set_response_type("tool_call")
    streamed.data = tool_response().data
    streamed.finish()
    agent.generator.aforward = AsyncMock(side_effect=[streamed, text_response()])
    events = []
    with pytest.raises(TaskPauseRequestedError):
        async for event in agent.stream_events("lookup", scope=scope()):
            events.append(event)
    required = [event for event in events if event.type == "tool.approval_required"]
    assert len(required) == 1
    assert sum(event.type == "run.paused" for event in events) == 1
    assert not any(event.type == "run.error" for event in events)
    assert "secret" not in repr(required[0].data)
    assert calls == []
    await agent.adecide_approval(
        required[0].data["request_id"], approved=True, decided_by="human"
    )
    assert await agent.acall("ignored", scope=scope()) == "done"
    assert calls == ["secret"]


def test_nested_call_cannot_bypass_top_level_approval():
    checkpoint, journal, calls = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)

    def outer() -> str:
        """Try a protected nested invocation."""
        return agent.tool_library.run("lookup", {"query": "nested"})

    agent.tool_library.add(outer)
    response = ModelResponse()
    response.set_response_type("tool_call")
    requests = ToolCallAggregator()
    requests.process(0, "outer_1", "outer", "{}")
    response.add(requests)
    agent.generator.forward = Mock(side_effect=[response, text_response()])
    assert agent("lookup", scope=scope()) == "done"
    assert calls == []


def test_execution_checkpoint_failure_prevents_dispatch(monkeypatch):
    checkpoint, journal, calls = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)
    with pytest.raises(TaskPauseRequestedError):
        agent("lookup", scope=scope())
    record = journal.pending("reviewer", "thread", "run")[0]
    agent.decide_approval(record.request_id, approved=True, decided_by="human")
    original = checkpoint.commit_state

    def fail_execution(*args, **kwargs):
        pending = kwargs.get("extension_state", {}).get("pending_approvals", {})
        if pending.get("phase") == "executing":
            raise OSError("checkpoint unavailable")
        return original(*args, **kwargs)

    monkeypatch.setattr(checkpoint, "commit_state", fail_execution)
    with pytest.raises(OSError):
        agent("ignored", scope=scope())
    assert calls == []
    assert journal.get("reviewer", record.request_id).status == "approved"


def test_inbox_interrupt_is_checked_before_resumed_dispatch():
    checkpoint, journal, calls = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)
    with pytest.raises(TaskPauseRequestedError):
        agent("lookup", scope=scope())
    record = journal.pending("reviewer", "thread", "run")[0]
    agent.decide_approval(record.request_id, approved=True, decided_by="human")
    agent._get_scoped_agent_inbox(scope()).interrupt(reason="operator cancelled")
    with pytest.raises(TaskInterruptRequestedError):
        agent("ignored", scope=scope())
    assert calls == []
    assert journal.get("reviewer", record.request_id).status == "approved"


def test_changed_live_principal_cannot_use_approval():
    checkpoint, journal, calls = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)
    with pytest.raises(TaskPauseRequestedError):
        agent("lookup", scope=scope())
    record = journal.pending("reviewer", "thread", "run")[0]
    agent.decide_approval(record.request_id, approved=True, decided_by="human")
    with pytest.raises(TaskPauseRequestedError, match="binding changed"):
        agent("ignored", scope=replace(scope(), principal="someone_else"))
    assert calls == []
    assert journal.get("reviewer", record.request_id).status == "approved"


def test_competing_resume_cannot_repeat_in_flight_batch():
    checkpoint, journal, calls = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)
    with pytest.raises(TaskPauseRequestedError):
        agent("lookup", scope=scope())
    record = journal.pending("reviewer", "thread", "run")[0]
    agent.decide_approval(record.request_id, approved=True, decided_by="human")
    started, release = Event(), Event()
    tool = agent.tool_library.get_tool_definition("lookup").executor
    original = tool.impl

    def blocked_worker(**kwargs):
        started.set()
        assert release.wait(timeout=5)
        return original(**kwargs)

    tool.impl = blocked_worker
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(agent, "ignored", scope=scope())
        try:
            assert started.wait(timeout=5)
            competitor = make_agent(checkpoint, journal, calls)
            with pytest.raises(TaskPauseRequestedError, match="reconciliation"):
                competitor("ignored", scope=scope())
            competitor.generator.forward.assert_not_called()
        finally:
            release.set()
        assert running.result(timeout=5) == "done"
    assert calls == ["secret"]
    assert checkpoint.load_state("reviewer", "thread", "run")["status"] == "completed"


@pytest.mark.parametrize("sqlite", [False, True])
@pytest.mark.parametrize("abandon", [False, True])
def test_host_reconciliation_is_atomic_and_idempotent(
    tmp_path, monkeypatch, sqlite, abandon
):
    checkpoint = (
        SQLiteCheckpointStore(str(tmp_path / "cp.db"))
        if sqlite
        else InMemoryCheckpointStore()
    )
    journal, calls = InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)
    with pytest.raises(TaskPauseRequestedError):
        agent("lookup", scope=scope())
    record = journal.pending("reviewer", "thread", "run")[0]
    agent.decide_approval(record.request_id, approved=True, decided_by="human")

    def crash(*args):
        raise SystemExit("worker lost")

    monkeypatch.setattr(agent, "_process_tool_intents", crash)
    with pytest.raises(SystemExit):
        agent("", scope=scope())
    if sqlite:
        checkpoint.close()

        checkpoint = SQLiteCheckpointStore(str(tmp_path / "cp.db"))
    restored = make_agent(checkpoint, journal, calls)
    state = restored.inspect_approval_batch("thread", "run")
    decision = {
        "expected_revision": state["_checkpoint"]["revision"],
        "decision_id": "repair:1",
        "decided_by": "operator",
        "reason": "checked external system",
        "worker_stopped": True,
        "abandon": abandon,
        "results": None if abandon else {"call_1": "confirmed result"},
    }
    with pytest.raises(ValueError, match="stopped"):
        restored.reconcile_approval_batch(
            "thread", "run", **{**decision, "worker_stopped": False}
        )
    with pytest.raises(CheckpointConflictError):
        restored.reconcile_approval_batch(
            "thread", "run", **{**decision, "expected_revision": 0}
        )
    if not abandon:
        with pytest.raises(ValueError, match="entire batch"):
            restored.reconcile_approval_batch(
                "thread", "run", **{**decision, "results": {}}
            )
    receipt = restored.reconcile_approval_batch("thread", "run", **decision)
    assert restored.reconcile_approval_batch("thread", "run", **decision) == receipt
    with pytest.raises(CheckpointConflictError):
        restored.reconcile_approval_batch(
            "thread", "run", **{**decision, "reason": "changed"}
        )
    events = checkpoint.load_events("reviewer", "thread", "run")
    assert sum(e["event_type"] == "approval.reconciled" for e in events) == 1
    if abandon:
        with pytest.raises(ValueError, match="terminal"):
            restored("", scope=scope())
    else:
        restored.generator.forward = Mock(return_value=text_response())
        assert restored("", scope=scope()) == "done"
        assert restored.reconcile_approval_batch("thread", "run", **decision) == receipt
    assert calls == []
    if sqlite:
        checkpoint.close()


@pytest.mark.asyncio
async def test_async_reconciliation_rollback_and_stale_writer(monkeypatch):
    checkpoint, journal, calls = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    agent = make_agent(checkpoint, journal, calls)
    with pytest.raises(TaskPauseRequestedError):
        agent("lookup", scope=scope())
    state = checkpoint.load_state("reviewer", "thread", "run")
    state["runtime"]["extensions"]["pending_approvals"]["phase"] = "executing"
    checkpoint.commit_state("reviewer", "thread", "run", state)
    state = await agent.ainspect_approval_batch("thread", "run")
    revision = state["_checkpoint"]["revision"]
    decision = {
        "expected_revision": revision,
        "decision_id": "repair",
        "decided_by": "host",
        "reason": "verified",
        "worker_stopped": True,
        "results": {"call_1": "found"},
    }
    original = checkpoint.commit_state

    def fail(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(checkpoint, "commit_state", fail)
    with pytest.raises(OSError):
        await agent.areconcile_approval_batch("thread", "run", **decision)
    assert checkpoint.load_state("reviewer", "thread", "run") == state
    monkeypatch.setattr(checkpoint, "commit_state", original)
    await agent.areconcile_approval_batch("thread", "run", **decision)
    with pytest.raises(CheckpointConflictError):
        checkpoint.commit_state(
            "reviewer", "thread", "run", state, expected_revision=revision
        )
    assert calls == []
