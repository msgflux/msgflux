"""Workspace approval failures remain visible without applying changes."""

from unittest.mock import AsyncMock, Mock

import msgspec
import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import (
    AgentApprovals,
    AgentWorkspace,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryApprovalStore,
    InMemoryWorkspace,
    PermissionSet,
    execution_context,
)
import msgflux.runtime.approvals.agent as approval_module
from msgflux.tools.builtin import WriteTool


def _tool_call_response():
    calls = ToolCallAggregator()
    calls.process(0, "write-call", "write", '{"path":"a","content":"new"}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _text_response():
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("done")
    return response


@pytest.mark.asyncio
async def test_final_workspace_approval_guard_failure_is_visible_and_non_mutating(
    monkeypatch,
):
    filesystem = InMemoryWorkspace("files", {"/a": b"old"})
    environment = ExecutionEnvironment(filesystem, write_guarantee="atomic_compare")
    workspace = AgentWorkspace.from_environment(
        environment,
        permissions=PermissionSet(
            resources=[
                filesystem.permission("/a", "filesystem.read"),
                filesystem.permission("/a", "filesystem.write"),
            ]
        ),
    )
    scope = ExecutionScope(
        namespace="editor",
        thread_id="thread",
        run_id="run",
        principal="user",
        workspace=workspace,
    )
    checkpoint = InMemoryCheckpointStore()
    approvals = InMemoryApprovalStore()
    agent = Agent(
        name="editor",
        model=Mock(model_type="chat_completion"),
        tools=[WriteTool()],
        checkpoint_store=checkpoint,
        approvals=AgentApprovals(approvals, {"write": "v1"}, "p1"),
    )
    agent.generator.aforward = AsyncMock(
        side_effect=[_tool_call_response(), _text_response()]
    )

    initial_events = []
    with pytest.raises(TaskPauseRequestedError):
        async for event in agent.stream_events("start", scope=scope):
            initial_events.append(event)
    record = approvals.pending("editor", "thread", "run")[0]
    agent.decide_approval(record.request_id, approved=True, decided_by="host")

    original_prepare = approval_module.prepare_workspace_change
    prepare_calls = 0

    def fail_during_consuming_guard(definition, arguments):
        nonlocal prepare_calls
        prepare_calls += 1
        # Resume preparation, guard preflight, then the final guard immediately
        # before consume and dispatch.
        if prepare_calls == 3:
            raise IsADirectoryError(21, "Is a directory", "/a")
        return original_prepare(definition, arguments)

    monkeypatch.setattr(
        approval_module, "prepare_workspace_change", fail_during_consuming_guard
    )
    resumed_events = [
        event async for event in agent.stream_events("resume", scope=scope)
    ]

    assert prepare_calls == 3
    blocked = [event for event in resumed_events if event.type == "tool.blocked"]
    assert len(blocked) == 1
    payload = blocked[0].data
    assert payload["tool_call_id"] == "write-call"
    assert payload["tool_name"] == "write"
    assert payload["code"] == "tool_approval_blocked"
    assert payload["reason"] == payload["error"].message
    error = msgspec.to_builtins(payload["error"])
    assert error["details"]["exception_type"] == "IsADirectoryError"
    assert "Is a directory" in error["message"]
    assert "/a" in error["message"]
    assert not any(event.type == "tool.end" for event in resumed_events)

    model_messages = agent.generator.aforward.call_args_list[-1].kwargs["messages"]
    call_output = next(
        item
        for item in model_messages
        if item.get("type") == "function_call_output"
        and item.get("call_id") == "write-call"
    )
    assert "Is a directory" in call_output["output"]
    assert "/a" in call_output["output"]

    state = checkpoint.load_state("editor", "thread", "run")
    history_output = next(
        item
        for item in state["messages"]["items"]
        if item.get("type") == "function_call_output"
        and item.get("call_id") == "write-call"
    )
    assert history_output["output"] == call_output["output"]
    with execution_context(scope=scope):
        assert filesystem.read_text("/a") == "old"
    assert approvals.get("editor", record.request_id).status == "approved"
