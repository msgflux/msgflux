"""Scope transitions retain investigation and budgets through real Agent resume."""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.generation.reasoning.react import ReAct
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent, ToolTurnLimitExtension
from msgflux.runtime.context import ExecutionScope
from msgflux.tools.builtin import close_context_scope, open_context_scope


def scope_response(name, args, flow):
    result = ModelResponse()
    if flow:
        result.set_response_type("structured")
        result.add(
            {"thought": "transition", "actions": [{"name": name, "arguments": args}]}
        )
    else:
        calls = ToolCallAggregator()
        calls.process(0, name, name, json.dumps(args))
        result.set_response_type("tool_call")
        result.add(calls)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("flow", [False, True])
async def test_resume_closes_scope_without_resetting_budget(asynchronous, flow):
    store = InMemoryCheckpointStore()
    scope = ExecutionScope(namespace="scopes", thread_id="thread", run_id="run")

    def make(responses):
        agent = Agent(
            name="scopes",
            model=Mock(model_type="chat_completion"),
            tools=[open_context_scope, close_context_scope],
            checkpoint_store=store,
            generation_schema=ReAct if flow else None,
            extensions=[ToolTurnLimitExtension(2)],
        )
        agent.generator.forward = Mock(side_effect=responses)
        agent.generator.aforward = AsyncMock(side_effect=responses)
        return agent

    first = make(
        [
            scope_response(
                "open_context_scope",
                {"name": "research", "summary": "Start research"},
                flow,
            ),
            RuntimeError("provider unavailable"),
        ]
    )
    with pytest.raises(RuntimeError, match="provider unavailable"):
        if asynchronous:
            await first.acall("Investigate", scope=scope)
        else:
            first("Investigate", scope=scope)
    checkpoint = store.load_state("scopes", "thread", "run")
    assert checkpoint["runtime"]["branch_id"] == "research"
    second = make(
        [scope_response("close_context_scope", {"summary": "Research result"}, flow)]
    )
    output = await second.acall(scope=scope) if asynchronous else second(scope=scope)
    assert output["completed_tool_turns"] == 2
    checkpoint = store.load_state("scopes", "thread", "run")
    assert checkpoint["runtime"]["branch_id"] == "root"
    assert checkpoint["status"] == "completed"
    state = checkpoint["messages"]
    assert "Start research" in str(state["items"])
    assert "Research result" in str(state["items"])
    branches = state["metadata"]["runtime"]["context_scopes"]["branches"]
    assert branches["research"]["closed"] is True
    assert "close_context_scope" in str(branches["research"]["items"])
