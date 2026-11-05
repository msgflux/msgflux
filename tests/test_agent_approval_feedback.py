"""Reviewer comments reach inference without wrapping or altering tool results."""

import json
import os
import subprocess
import sys
from pathlib import Path
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import (
    AgentApprovals,
    ExecutionScope,
    InMemoryApprovalStore,
    SQLiteApprovalStore,
)
from msgflux.runtime.approvals.feedback import append_review_feedback


def _response(data="done", *, mode="chat_completions", native=None):
    response = ModelResponse()
    if data == "call":
        calls = ToolCallAggregator(api_mode=mode)
        calls.process(0, "call_1", "lookup", '{"value": 6}')
        if native is not None:
            calls.native_calls["call_1"] = native
        response.set_response_type("tool_call")
        response.add(calls)
    else:
        response.set_response_type("text_generation")
        response.add(data)
    return response


def _agent(
    checkpoints, journal, effects, *, crash_marker=None, hooks=None, tool_result=None
):
    def lookup(value: int) -> dict:
        """Return a structured result."""
        effects.append(value)
        if crash_marker is not None:
            Path(crash_marker).write_text(str(value))
            os._exit(73)
        return (
            tool_result if tool_result is not None else {"value": value * 2, "ok": True}
        )

    return Agent(
        name="reviewer",
        model=Mock(model_type="chat_completion"),
        tools=[lookup],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(journal, {"lookup": "v1"}, "p1"),
        hooks=hooks,
    )


def _scope():
    return ExecutionScope(
        namespace="reviewer", thread_id="thread", run_id="run", principal="user"
    )


def _comments(items):
    return [item for item in items if "approval_feedback" in item.get("metadata", {})]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat_completions", "responses"])
@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_feedback_follows_original_tool_output(mode, approved, asynchronous):
    checkpoint, journal, effects = (
        InMemoryCheckpointStore(),
        InMemoryApprovalStore(),
        [],
    )
    agent = _agent(checkpoint, journal, effects)
    requests = []

    def generate(**kwargs):
        requests.append(deepcopy(kwargs["messages"]._to_state()["items"]))
        return _response("call", mode=mode) if len(requests) == 1 else _response()

    agent.generator.forward = Mock(side_effect=generate)
    agent.generator.aforward = AsyncMock(side_effect=generate)

    async def invoke(prompt):
        return (
            await agent.acall(prompt, scope=_scope())
            if asynchronous
            else agent(prompt, scope=_scope())
        )

    try:
        with pytest.raises(TaskPauseRequestedError):
            await invoke("look up")
        (record,) = journal.pending("reviewer", "thread", "run")
        agent.decide_approval(
            record.request_id,
            approved=approved,
            decided_by="human",
            feedback="  Use the result to prepare a summary.  ",
        )
        assert effects == []
        assert await invoke("ignored on resume") == "done"
        assert len(requests) == 2
        assert effects == ([6] if approved else [])
        items = requests[-1]
        comments = _comments(items)
        assert len(comments) == 1
        comment = comments[0]
        (review,) = comment["metadata"]["approval_feedback"]["reviews"]
        assert review == {
            "request_id": record.request_id,
            "tool_call_id": "call_1",
            "tool_name": "lookup",
            "approved": approved,
            "status": "approved" if approved else "denied",
            "decided_by": "human",
            "feedback": "Use the result to prepare a summary.",
        }
        assert comment["role"] == "user"
        assert review["feedback"] in comment["content"]
        outputs = [item for item in items if item.get("type") == "function_call_output"]
        assert outputs
        output = outputs[-1]
        assert items.index(output) < items.index(comment)
        if approved:
            assert json.loads(output["output"]) == {"value": 12, "ok": True}
        else:
            assert output["output"] == "Tool approval is denied"
        assert review["feedback"] not in output["output"]
        state = checkpoint.load_state("reviewer", "thread", "run")
        assert len(_comments(state["messages"]["items"])) == 1
        assert "pending_approvals" not in state["runtime"]["extensions"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True])
async def test_feedback_survives_reopen_before_resume(tmp_path, approved):
    cp_path, journal_path = tmp_path / "cp.db", tmp_path / "approvals.db"
    checkpoints, journal = (
        SQLiteCheckpointStore(str(cp_path)),
        SQLiteApprovalStore(str(journal_path)),
    )
    effects = []
    agent = _agent(checkpoints, journal, effects)
    agent.generator.forward = Mock(return_value=_response("call"))
    with pytest.raises(TaskPauseRequestedError):
        agent("lookup", scope=_scope())
    (record,) = journal.pending("reviewer", "thread", "run")
    agent.decide_approval(
        record.request_id,
        approved=approved,
        decided_by="host",
        feedback="Continue with a short explanation.",
    )
    await agent.aclose()
    checkpoints.close()
    journal.close()
    checkpoints, journal = (
        SQLiteCheckpointStore(str(cp_path)),
        SQLiteApprovalStore(str(journal_path)),
    )
    restored = _agent(checkpoints, journal, effects)
    seen = []

    def generate(**kwargs):
        seen.extend(deepcopy(kwargs["messages"]._to_state()["items"]))
        return _response()

    restored.generator.aforward = AsyncMock(side_effect=generate)
    try:
        assert await restored.acall("resume", scope=_scope()) == "done"
        assert effects == ([6] if approved else [])
        assert len(_comments(seen)) == 1
        assert restored.generator.aforward.await_count == 1
        state = checkpoints.load_state("reviewer", "thread", "run")
        assert len(_comments(state["messages"]["items"])) == 1
    finally:
        await restored.aclose()
        checkpoints.close()
        journal.close()


@pytest.mark.asyncio
async def test_no_comment_means_no_extra_context():
    checkpoint, journal, effects = (
        InMemoryCheckpointStore(),
        InMemoryApprovalStore(),
        [],
    )
    agent = _agent(checkpoint, journal, effects)
    agent.generator.aforward = AsyncMock(side_effect=[_response("call"), _response()])
    try:
        with pytest.raises(TaskPauseRequestedError):
            await agent.acall("lookup", scope=_scope())
        (record,) = journal.pending("reviewer", "thread", "run")
        agent.decide_approval(record.request_id, approved=True, decided_by="human")
        assert await agent.acall("resume", scope=_scope()) == "done"
        state = checkpoint.load_state("reviewer", "thread", "run")
        assert not _comments(state["messages"]["items"])
    finally:
        await agent.aclose()


def test_delivery_receipt_survives_history_serialization():
    pending = {
        "review_feedback": [
            {
                "request_id": "request",
                "tool_call_id": "call",
                "tool_name": "lookup",
                "approved": True,
                "decided_by": "host",
                "feedback": "Good, continue.",
            }
        ]
    }
    messages = ChatMessages()
    messages.add_user("Original task")
    append_review_feedback(messages, pending)
    restored = ChatMessages()
    restored._hydrate_state(messages._to_state())
    append_review_feedback(restored, json.loads(json.dumps(pending)))
    assert len(_comments(list(restored))) == 1


@pytest.mark.asyncio
async def test_real_process_crash_keeps_feedback_for_host_reconciliation(tmp_path):
    cp_path, ap_path, marker = (
        tmp_path / "cp.db",
        tmp_path / "ap.db",
        tmp_path / "effect",
    )
    cp, journal = SQLiteCheckpointStore(str(cp_path)), SQLiteApprovalStore(str(ap_path))
    agent = _agent(cp, journal, [], crash_marker=marker)
    agent.generator.forward = Mock(return_value=_response("call"))
    with pytest.raises(TaskPauseRequestedError):
        agent("lookup", scope=_scope())
    (record,) = journal.pending("reviewer", "thread", "run")
    agent.decide_approval(
        record.request_id,
        approved=True,
        decided_by="host",
        feedback="Explain the confirmed result.",
    )
    await agent.aclose()
    cp.close()
    journal.close()
    script = """
import sys
from unittest.mock import Mock
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.runtime import SQLiteApprovalStore
from tests.test_agent_approval_feedback import _agent, _scope, _response
cp = SQLiteCheckpointStore(sys.argv[1])
ap = SQLiteApprovalStore(sys.argv[2])
agent = _agent(cp, ap, [], crash_marker=sys.argv[3])
agent.generator.forward = Mock(side_effect=AssertionError('Unexpected inference before tool replay'))
agent('resume', scope=_scope())
"""
    result = subprocess.run(  # noqa: S603 - trusted fixed worker script, local temp paths
        [sys.executable, "-c", script, str(cp_path), str(ap_path), str(marker)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 73, result.stderr
    assert marker.read_text() == "6"
    cp, journal = SQLiteCheckpointStore(str(cp_path)), SQLiteApprovalStore(str(ap_path))
    effects = []
    restored = _agent(cp, journal, effects)
    try:
        state = restored.inspect_approval_batch("thread", "run")
        pending = state["runtime"]["extensions"]["pending_approvals"]
        assert pending["phase"] == "executing"
        assert not _comments(state["messages"]["items"])
        assert (
            pending["review_feedback"][0]["feedback"] == "Explain the confirmed result."
        )
        decision = {
            "expected_revision": state["_checkpoint"]["revision"],
            "decision_id": "repair",
            "decided_by": "operator",
            "reason": "Verified the worker effect",
            "worker_stopped": True,
            "results": {"call_1": '{"value":12,"ok":true}'},
        }
        restored.reconcile_approval_batch("thread", "run", **decision)
        restored.reconcile_approval_batch("thread", "run", **decision)
        seen = []

        def generate(**kwargs):
            seen.extend(deepcopy(kwargs["messages"]._to_state()["items"]))
            return _response()

        restored.generator.aforward = AsyncMock(side_effect=generate)
        assert await restored.acall("resume", scope=_scope()) == "done"
        assert len(_comments(seen)) == 1
        assert effects == []
        assert marker.read_text() == "6"
    finally:
        await restored.aclose()
        cp.close()
        journal.close()


@pytest.mark.asyncio
async def test_partial_batch_control_and_reconciliation_do_not_repeat_comment():
    from msgflux.nn.hooks import Hook

    cp, journal, effects = InMemoryCheckpointStore(), InMemoryApprovalStore(), []

    def after_tool(context):
        raise TaskPauseRequestedError("run", "host pause after execution")

    agent = _agent(
        cp, journal, effects, hooks=[Hook(event="after_tool", handler=after_tool)]
    )
    agent.generator.forward = Mock(return_value=_response("call"))
    try:
        with pytest.raises(TaskPauseRequestedError, match="waiting"):
            agent("lookup", scope=_scope())
        (record,) = journal.pending("reviewer", "thread", "run")
        agent.decide_approval(
            record.request_id,
            approved=True,
            decided_by="host",
            feedback="Next, summarize the result.",
        )
        with pytest.raises(TaskPauseRequestedError, match="host pause"):
            agent("resume", scope=_scope())
        state = agent.inspect_approval_batch("thread", "run")
        assert effects == [6]
        assert len(_comments(state["messages"]["items"])) == 1
        agent.reconcile_approval_batch(
            "thread",
            "run",
            expected_revision=state["_checkpoint"]["revision"],
            decision_id="repair",
            decided_by="operator",
            reason="Verified result",
            worker_stopped=True,
            results={"call_1": '{"value":12,"ok":true}'},
        )
        state = agent.inspect_approval_batch("thread", "run")
        assert len(_comments(state["messages"]["items"])) == 1
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("kind", ["apply_patch", "shell"])
async def test_native_tool_wire_result_is_unchanged(approved, kind):
    from msgflux.tools.shell import ShellCommandResult, ShellResult

    cp, journal, effects = InMemoryCheckpointStore(), InMemoryApprovalStore(), []
    native = {"codec": f"openai.responses.{kind}", "version": 1, "name": "lookup"}
    tool_result = {"status": "completed", "output": "patched"}
    if kind == "shell":
        native["command_count"] = 1
        native["max_output_length"] = None
        tool_result = ShellResult(
            results=(
                ShellCommandResult(
                    status="exited", returncode=0, stdout="12", stderr=""
                ),
            )
        )
    agent = _agent(cp, journal, effects, tool_result=tool_result)
    seen = []

    def generate(**kwargs):
        seen.append(deepcopy(kwargs["messages"]._to_state()["items"]))
        return (
            _response("call", mode="responses", native=native)
            if len(seen) == 1
            else _response()
        )

    agent.generator.aforward = AsyncMock(side_effect=generate)
    try:
        with pytest.raises(TaskPauseRequestedError):
            await agent.acall("do it", scope=_scope())
        (record,) = journal.pending("reviewer", "thread", "run")
        agent.decide_approval(
            record.request_id,
            approved=approved,
            decided_by="host",
            feedback="Report what happened.",
        )
        assert await agent.acall("resume", scope=_scope()) == "done"
        (native_output,) = [
            item for item in seen[-1] if item.get("type") == f"{kind}_call_output"
        ]
        if kind == "apply_patch":
            assert native_output["status"] == ("completed" if approved else "failed")
            assert native_output["output"] == (
                "patched" if approved else "Tool approval is denied"
            )
        else:
            (part,) = native_output["output"]
            assert part["stdout"] == ("12" if approved else "")
            assert part["stderr"] == ("" if approved else "Tool approval is denied")
        assert "Report what happened." not in str(native_output)
        assert len(_comments(seen[-1])) == 1
        assert effects == ([6] if approved else [])
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "provider", ["openai-chat", "openai-responses", "anthropic", "gemini"]
)
def test_provider_payload_keeps_feedback_outside_tool_result(monkeypatch, provider):
    from msgflux.models import Model
    from msgflux.models.openai_compatible import OpenAICompatibleModel

    monkeypatch.setattr(OpenAICompatibleModel, "profile", property(lambda _owner: None))
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(name, "test-only-key")
    messages = ChatMessages()
    messages.add_user("Look up six")
    messages.append(
        {
            "type": "function_call",
            "call_id": "call",
            "name": "lookup",
            "arguments": '{"value":6}',
        }
    )
    messages.append(
        {
            "type": "function_call_output",
            "call_id": "call",
            "output": '{"value":12,"ok":true}',
        }
    )
    append_review_feedback(
        messages,
        {
            "review_feedback": [
                {
                    "request_id": "request",
                    "tool_call_id": "call",
                    "tool_name": "lookup",
                    "approved": True,
                    "decided_by": "host",
                    "feedback": "Now summarize the result.",
                }
            ]
        },
    )
    base = "openai" if provider.startswith("openai") else provider
    options = (
        {"api_mode": "chat_completions" if provider.endswith("chat") else "responses"}
        if base == "openai"
        else {}
    )
    model = Model.chat_completion(f"{base}/test-model", **options)
    try:
        params = model._build_generation_params(
            messages, system_prompt=None, prefilling=None, tool_catalog=None
        )
        if provider == "anthropic":
            body = model._to_anthropic_body({**model.sampling_run_params, **params})
            parts = [
                part
                for message in body["messages"]
                if isinstance(message["content"], list)
                for part in message["content"]
            ]
            (output,) = [part for part in parts if part.get("type") == "tool_result"]
            assert output["content"] == '{"value":12,"ok":true}'
        elif provider == "gemini":
            body = model._adapt_interactions_params(
                {**model.sampling_run_params, **params}, stream=False
            )
            assert [part["type"] for part in body["input"]] == [
                "user_input",
                "function_call",
                "function_result",
                "user_input",
            ]
            assert "Now summarize the result." not in str(body["input"][2])
        else:
            body = params
            if provider.endswith("chat"):
                (output,) = [
                    part for part in body["messages"] if part.get("role") == "tool"
                ]
                assert output["content"] == '{"value":12,"ok":true}'
            else:
                (output,) = [
                    part
                    for part in body["input"]
                    if part.get("type") == "function_call_output"
                ]
                assert output["output"] == '{"value":12,"ok":true}'
        assert "Now summarize the result." in json.dumps(body)
        assert "approval_feedback" not in json.dumps(body)
    finally:
        model.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("blocked", ["filter", "expired"])
async def test_comment_survives_when_approved_call_is_blocked(blocked, asynchronous):
    checkpoints, journal, effects = (
        InMemoryCheckpointStore(),
        InMemoryApprovalStore(),
        [],
    )
    agent = _agent(checkpoints, journal, effects)
    agent.generator.forward = Mock(side_effect=[_response("call"), _response()])
    agent.generator.aforward = AsyncMock(side_effect=[_response("call"), _response()])

    async def invoke(**kwargs):
        if asynchronous:
            return await agent.acall("lookup", scope=_scope(), **kwargs)
        return agent("lookup", scope=_scope(), **kwargs)

    try:
        with pytest.raises(TaskPauseRequestedError):
            await invoke()
        (record,) = journal.pending("reviewer", "thread", "run")
        agent.decide_approval(
            record.request_id,
            approved=True,
            decided_by="human",
            feedback="Summarize only the data actually available.",
        )
        if blocked == "expired":
            journal._clock = lambda: record.expires_at
        options = {"tool_filter": {"block": "lookup"}} if blocked == "filter" else {}
        assert await invoke(**options) == "done"
        assert effects == []
        state = checkpoints.load_state("reviewer", "thread", "run")
        items = state["messages"]["items"]
        (comment,) = _comments(items)
        (review,) = comment["metadata"]["approval_feedback"]["reviews"]
        assert review["approved"] is True
        assert review["status"] == ("expired" if blocked == "expired" else "approved")
        assert review["feedback"] == "Summarize only the data actually available."
        (output,) = [
            item for item in items if item.get("type") == "function_call_output"
        ]
        assert items.index(output) < items.index(comment)
        assert review["feedback"] not in output["output"]
        assert "pending_approvals" not in state["runtime"]["extensions"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_batch_comments_follow_all_results_and_keep_call_identity():
    checkpoints, journal, effects = (
        InMemoryCheckpointStore(),
        InMemoryApprovalStore(),
        [],
    )
    agent = _agent(checkpoints, journal, effects)
    calls = ToolCallAggregator()
    calls.process(0, "first", "lookup", '{"value": 3}')
    calls.process(1, "second", "lookup", '{"value": 6}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    agent.generator.aforward = AsyncMock(side_effect=[response, _response()])
    try:
        with pytest.raises(TaskPauseRequestedError):
            await agent.acall("lookup both", scope=_scope())
        records = journal.pending("reviewer", "thread", "run")
        assert len(records) == 2
        for record in records:
            accepted = record.binding.tool_call_id == "first"
            agent.decide_approval(
                record.request_id,
                approved=accepted,
                decided_by="human",
                feedback="Explain the first result."
                if accepted
                else "Skip the second lookup.",
            )
        assert await agent.acall("resume", scope=_scope()) == "done"
        assert effects == [3]
        items = checkpoints.load_state("reviewer", "thread", "run")["messages"]["items"]
        (comment,) = _comments(items)
        reviews = comment["metadata"]["approval_feedback"]["reviews"]
        assert [(r["tool_call_id"], r["approved"], r["feedback"]) for r in reviews] == [
            ("first", True, "Explain the first result."),
            ("second", False, "Skip the second lookup."),
        ]
        outputs = [item for item in items if item.get("type") == "function_call_output"]
        assert len(outputs) == 2
        assert all(items.index(output) < items.index(comment) for output in outputs)
        assert json.loads(outputs[0]["output"]) == {"value": 6, "ok": True}
        assert outputs[1]["output"] == "Tool approval is denied"
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_feedback_remains_visible_after_approved_scope_close():
    from msgflux.tools.builtin.context_scope import (
        close_context_scope,
        open_context_scope,
    )

    checkpoints, journal = InMemoryCheckpointStore(), InMemoryApprovalStore()
    agent = Agent(
        name="reviewer",
        model=Mock(model_type="chat_completion"),
        tools=[open_context_scope, close_context_scope],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(journal, {"close_context_scope": "v1"}, "p1"),
    )

    def transition(name, arguments):
        calls = ToolCallAggregator()
        calls.process(0, name, name, json.dumps(arguments))
        response = ModelResponse()
        response.set_response_type("tool_call")
        response.add(calls)
        return response

    seen = []
    responses = iter(
        [
            transition("open_context_scope", {"name": "research"}),
            transition("close_context_scope", {"summary": "Research complete."}),
            _response(),
        ]
    )

    def generate(**kwargs):
        seen.append(deepcopy(kwargs["messages"]._to_state()["items"]))
        return next(responses)

    agent.generator.aforward = AsyncMock(side_effect=generate)
    try:
        with pytest.raises(TaskPauseRequestedError):
            await agent.acall("research", scope=_scope())
        (record,) = journal.pending("reviewer", "thread", "run")
        agent.decide_approval(
            record.request_id,
            approved=True,
            decided_by="human",
            feedback="Include a short conclusion after returning.",
        )
        assert await agent.acall("resume", scope=_scope()) == "done"
        (comment,) = _comments(seen[-1])
        assert "Include a short conclusion after returning." in comment["content"]
    finally:
        await agent.aclose()
