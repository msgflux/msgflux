"""Public Agent and Bash trajectories persist workspace command evidence."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import (
    AgentWorkspace,
    ExecutionScope,
    LocalWorkspaceBackend,
    PermissionSet,
    SandboxRequirements,
)
from msgflux.runtime.workspace.receipts import decode_command_receipt
from msgflux.runtime.workspace.references import encode_workspace_reference
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.tools.builtin import BashTool


def _tool_response(calls):
    aggregate = ToolCallAggregator()
    for index, call_id, command in calls:
        aggregate.process(
            index,
            call_id,
            "bash",
            json.dumps({"command": command}),
        )
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(aggregate)
    response.reasoning = None
    response.metadata = {}
    return response


def _final_response():
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("Finished the workspace commands.")
    response.reasoning = None
    response.metadata = {}
    return response


async def _open_workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    registry = SQLiteWorkspaceRegistry(tmp_path / "workspace-registry.sqlite")
    backend = LocalWorkspaceBackend(root, registry=registry, allow_processes=True)
    workspace = await AgentWorkspace.open(
        backend,
        "agent-command-workspace",
        permissions=PermissionSet(
            {"filesystem.read", "filesystem.write", "process.execute"}
        ),
        requirements=SandboxRequirements(),
    )
    return workspace, registry


def _agent(*, workspace, checkpoint_store, responses):
    agent = Agent(
        name="command_agent",
        model=Mock(model_type="chat_completion"),
        workspace=workspace,
        checkpoint_store=checkpoint_store,
        tools=[BashTool()],
    )
    agent.generator.aforward = AsyncMock(side_effect=responses)
    return agent


def _scope(workspace):
    return ExecutionScope(
        namespace="command_agent",
        thread_id="command-thread",
        run_id="command-run",
        principal="test-host",
        workspace=workspace,
    )


@pytest.mark.asyncio
async def test_agent_bash_persists_completed_bounded_command_receipt(tmp_path):
    workspace, registry = await _open_workspace(tmp_path)
    checkpoints = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite")
    agent = _agent(
        workspace=workspace,
        checkpoint_store=checkpoints,
        responses=[
            _tool_response(
                [
                    (
                        0,
                        "large-output-call",
                        "printf 'written by Agent Bash' > result.txt && "
                        "python -c \"print('x' * 20000)\"",
                    )
                ]
            ),
            _final_response(),
        ],
    )
    try:
        result = await agent.acall("Create the result file", scope=_scope(workspace))
        assert result == "Finished the workspace commands."
        assert agent.generator.aforward.await_count == 2
        assert workspace.read_text("/result.txt") == "written by Agent Bash"

        state = checkpoints.load_state("command_agent", "command-thread", "command-run")
        assert state["runtime"]["extensions"]["workspace_reference"] == (
            encode_workspace_reference(workspace)
        )
        extensions = state["runtime"]["extensions"]
        assert len(extensions["command_receipts"]) == 1
        receipt = decode_command_receipt(extensions["command_receipts"][0])
        assert receipt.state == "completed"
        assert receipt.tool_output_recorded is True
        assert receipt.tool_call_id == "large-output-call"
        assert receipt.workspace_reference == encode_workspace_reference(workspace)
        assert receipt.returncode == 0
        assert receipt.stdout is not None
        assert len(receipt.stdout.encode("utf-8")) == 8192

        items = state["messages"]["items"]
        call = next(
            item
            for item in items
            if item.get("type") == "function_call"
            and item.get("call_id") == "large-output-call"
        )
        output = next(
            item
            for item in items
            if item.get("type") == "function_call_output"
            and item.get("call_id") == "large-output-call"
        )
        assert items.index(call) < items.index(output)
    finally:
        checkpoints.close()
        await workspace.aclose()
        registry.close()


@pytest.mark.asyncio
async def test_agent_bash_without_checkpoint_store_still_executes(tmp_path):
    workspace, registry = await _open_workspace(tmp_path)
    agent = _agent(
        workspace=workspace,
        checkpoint_store=None,
        responses=[
            _tool_response([(0, "uncached-call", "printf 'ok' > no-checkpoint.txt")]),
            _final_response(),
        ],
    )
    try:
        result = await agent.acall("Create a file", scope=_scope(workspace))
        assert result == "Finished the workspace commands."
        assert agent.generator.aforward.await_count == 2
        assert workspace.read_text("/no-checkpoint.txt") == "ok"
    finally:
        await workspace.aclose()
        registry.close()


@pytest.mark.asyncio
async def test_parallel_agent_bash_calls_share_receipt_checkpoint_lock(tmp_path):
    workspace, registry = await _open_workspace(tmp_path)
    checkpoints = SQLiteCheckpointStore(tmp_path / "parallel-checkpoints.sqlite")
    agent = _agent(
        workspace=workspace,
        checkpoint_store=checkpoints,
        responses=[
            _tool_response(
                [
                    (0, "parallel-a", "printf 'a' > a.txt"),
                    (1, "parallel-b", "printf 'b' > b.txt"),
                ]
            ),
            _final_response(),
        ],
    )
    try:
        result = await agent.acall("Write both files", scope=_scope(workspace))
        assert result == "Finished the workspace commands."
        assert agent.generator.aforward.await_count == 2
        assert workspace.read_text("/a.txt") == "a"
        assert workspace.read_text("/b.txt") == "b"

        state = checkpoints.load_state("command_agent", "command-thread", "command-run")
        receipts = [
            decode_command_receipt(value)
            for value in state["runtime"]["extensions"]["command_receipts"]
        ]
        assert {receipt.tool_call_id for receipt in receipts} == {
            "parallel-a",
            "parallel-b",
        }
        assert all(receipt.state == "completed" for receipt in receipts)
        assert all(receipt.tool_output_recorded for receipt in receipts)
    finally:
        checkpoints.close()
        await workspace.aclose()
        registry.close()
