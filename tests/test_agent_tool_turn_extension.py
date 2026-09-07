"""Budget enforcement is identical across transports, async and resume."""

from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.generation.reasoning.react import ReAct
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import ToolTurnLimitExtension
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.context import ExecutionScope
from msgflux.nn.hooks.events import ContinuationContext


def lookup() -> str:
    """Return the answer."""
    return "found"


def tool_response(flow=False, index=0):
    response = ModelResponse()
    if flow:
        response.set_response_type("structured")
        response.add(
            {"thought": "look", "actions": [{"name": "lookup", "arguments": {}}]}
        )
    else:
        calls = ToolCallAggregator()
        calls.process(0, f"call_{index}", "lookup", "{}")
        response.set_response_type("tool_call")
        response.add(calls)
    return response


def make_agent(limit, *, flow=False, store=None):
    agent = Agent(
        name="budget",
        model=Mock(model_type="chat_completion"),
        tools=[lookup],
        generation_schema=ReAct if flow else None,
        checkpoint_store=store,
        extensions=[ToolTurnLimitExtension(limit)],
    )
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("flow", [False, True])
@pytest.mark.parametrize("limit", [1, 2])
async def test_limit_stops_after_last_settled_batch(asynchronous, flow, limit):
    agent = make_agent(limit, flow=flow)
    responses = [tool_response(flow, i) for i in range(limit)]
    mock = (
        AsyncMock(side_effect=responses)
        if asynchronous
        else Mock(side_effect=responses)
    )
    if asynchronous:
        agent.generator.aforward = mock
    else:
        agent.generator.forward = mock
    history = ChatMessages()
    output = (
        await agent.acall("start", messages=history)
        if asynchronous
        else agent("start", messages=history)
    )
    assert output["stop_reason"] == "tool_turn_limit"
    assert output["completed_tool_turns"] == limit
    assert output["tool_responses"]["tool_calls"][0]["result"] == "found"
    assert mock.call_count == limit
    assert "Tool budget: 1 round(s) remaining" in mock.call_args.kwargs["system_prompt"]
    assert "Tool budget:" not in (agent.system_prompt.data or "")
    assert history.get_active_turn() is None
    if not flow:
        assert (
            len([item for item in history if item["type"] == "function_call_output"])
            == limit
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_budget_survives_failed_run_resume(asynchronous):
    store = InMemoryCheckpointStore()
    agent = make_agent(2, store=store)
    scope = ExecutionScope(thread_id="thread", run_id="run", namespace="budget")
    responses = [tool_response(index=0), RuntimeError("provider unavailable")]
    if asynchronous:
        agent.generator.aforward = AsyncMock(side_effect=responses)
    else:
        agent.generator.forward = Mock(side_effect=responses)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        if asynchronous:
            await agent.acall("start", scope=scope)
        else:
            agent("start", scope=scope)
    restored = make_agent(2, store=store)
    mock = (
        AsyncMock(return_value=tool_response(index=1))
        if asynchronous
        else Mock(return_value=tool_response(index=1))
    )
    if asynchronous:
        restored.generator.aforward = mock
        result = await restored.acall(scope=scope)
    else:
        restored.generator.forward = mock
        result = restored(scope=scope)
    assert result["completed_tool_turns"] == 2
    assert mock.call_count == 1


@pytest.mark.parametrize("limit", [True, False, 0, -1, "2"])
def test_limit_rejects_invalid_values(limit):
    with pytest.raises(ValueError, match="positive integer"):
        ToolTurnLimitExtension(limit)


def test_empty_tool_batch_does_not_consume_budget():
    extension = ToolTurnLimitExtension(1)
    state = {}
    extension.durable_state = lambda: state
    context = ContinuationContext(
        phase="after_tools",
        action="continue",
        scope=ExecutionScope(),
        messages=None,
    )
    result = extension._decide(context)
    assert result.action == "continue"
    assert state == {}
