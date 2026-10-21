"""Approved tool controls keep an executing batch behind reconciliation."""

import json
from unittest.mock import Mock

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime import (
    AgentApprovals,
    ApprovalReconciliationRequiredError,
    ExecutionScope,
    SQLiteApprovalStore,
)


def _tool_response():
    calls = ToolCallAggregator(api_mode="responses")
    history = []
    for index, (call_id, name, arguments) in enumerate(
        [
            ("approved-call", "approved_effect", {"value": "A"}),
            ("sibling-call", "sibling_effect", {"value": "B"}),
        ]
    ):
        serialized = json.dumps(arguments, separators=(",", ":"))
        calls.process(index, call_id, name, serialized)
        history.append(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": serialized,
            }
        )
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    response.history_items = history
    return response


def _make_agent(checkpoints, approvals, after_tool, effects):
    def approved_effect(value: str) -> str:
        """Record the externally visible approved action."""
        effects.append(f"approved:{value}")
        return "approved result"

    def sibling_effect(value: str) -> str:
        """Record the sibling action in the same tool batch."""
        effects.append(f"sibling:{value}")
        return "sibling result"

    agent = Agent(
        name="approval-agent",
        model=Mock(model_type="chat_completion"),
        tools=[approved_effect, sibling_effect],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(approvals, {"approved_effect": "v1"}, "policy-v1"),
        hooks=[Hook(event="after_tool", handler=after_tool)],
    )
    return agent


@pytest.mark.asyncio
async def test_approved_controlled_batch_survives_reopen_as_uncertain(tmp_path):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    approval_path = tmp_path / "approvals.sqlite3"
    checkpoints = SQLiteCheckpointStore(checkpoint_path)
    approvals = SQLiteApprovalStore(approval_path)
    effects = []
    control = TaskPauseRequestedError("run", "pause after approved effect")
    state = {"raised": False}

    def after_tool(context):
        if context.tool_name == "approved_effect" and not state["raised"]:
            state["raised"] = True
            raise control

    agent = _make_agent(checkpoints, approvals, after_tool, effects)
    agent.generator.forward = Mock(return_value=_tool_response())
    scope = ExecutionScope(
        namespace="approval-agent",
        thread_id="thread",
        run_id="run",
        principal="user",
    )
    reopened_agent = None

    try:
        with pytest.raises(TaskPauseRequestedError, match="waiting"):
            agent("start", scope=scope)
        assert effects == []
        (record,) = approvals.pending("approval-agent", "thread", "run")
        agent.decide_approval(record.request_id, approved=True, decided_by="host")

        with pytest.raises(TaskPauseRequestedError) as raised:
            agent("resume", scope=scope)
        assert raised.value is control
        assert sorted(effects) == ["approved:A", "sibling:B"]
        agent_state = checkpoints.load_state("approval-agent", "thread", "run")
        assert agent_state["status"] == "paused"
        pending = agent_state["runtime"]["extensions"]["pending_approvals"]
        assert pending["phase"] == "executing"
        outputs = {
            item["call_id"]: item
            for item in agent_state["messages"]["items"]
            if item.get("type") == "function_call_output"
        }
        assert set(outputs) == {"approved-call", "sibling-call"}
        assert "sibling result" in str(outputs["sibling-call"]["output"])
        assert "pause after approved effect" in str(outputs["approved-call"]["output"])
        agent.generator.forward.assert_called_once()

        await agent.aclose()
        agent = None
        checkpoints.close()
        approvals.close()
        checkpoints = SQLiteCheckpointStore(checkpoint_path)
        approvals = SQLiteApprovalStore(approval_path)
        reopened_agent = _make_agent(checkpoints, approvals, after_tool, effects)
        reopened_agent.generator.forward = Mock(return_value=_tool_response())

        with pytest.raises(ApprovalReconciliationRequiredError, match="reconciliation"):
            reopened_agent("resume", scope=scope)
        reopened_agent.generator.forward.assert_not_called()
        assert sorted(effects) == ["approved:A", "sibling:B"]
        reopened_state = checkpoints.load_state("approval-agent", "thread", "run")
        assert (
            reopened_state["runtime"]["extensions"]["pending_approvals"]["phase"]
            == "executing"
        )
    finally:
        if agent is not None:
            await agent.aclose()
        if reopened_agent is not None:
            await reopened_agent.aclose()
        checkpoints.close()
        approvals.close()
