"""return_messages must accept ordinary feedback-hook results."""

from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.generation.reasoning.react import ReAct
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime import ExecutionScope


def _tool_call(flow):
    response = ModelResponse()
    if flow:
        response.set_response_type("structured")
        response.add(
            {"thought": "use tool", "actions": [{"name": "lookup", "arguments": {}}]}
        )
    else:
        response.set_response_type("tool_call")
        calls = ToolCallAggregator()
        calls.process(0, "call_1", "lookup", "{}")
        response.add(calls)
    response.metadata = {}
    return response


def lookup() -> str:
    """Return a stable tool result."""
    return "found"


def _agent(output, *, flow=False):
    def feedback(ctx):
        return replace(ctx, action="return", output=output)

    agent = Agent(
        name="feedback",
        model=Mock(model_type="chat_completion"),
        tools=[lookup],
        generation_schema=ReAct if flow else None,
        checkpoint_store=InMemoryCheckpointStore(),
        hooks=[Hook(event="resolve_tool_feedback", handler=feedback)],
        config={"return_messages": True},
    )
    agent.generator.forward = Mock(return_value=_tool_call(flow))
    agent.generator.aforward = AsyncMock(return_value=_tool_call(flow))
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("flow", [False, True], ids=["native", "react"])
@pytest.mark.parametrize(
    ("output", "field", "expected"),
    [
        ({"status": "awaiting_approval"}, "status", "awaiting_approval"),
        ("manual result", "response", "manual result"),
    ],
    ids=["mapping", "scalar"],
)
async def test_return_messages_preserves_feedback_output(
    asynchronous, flow, output, field, expected
):
    agent = _agent(output, flow=flow)
    history = ChatMessages()
    scope = ExecutionScope(namespace="feedback", thread_id="thread", run_id="run")
    try:
        result = (
            await agent.acall("start", messages=history, scope=scope)
            if asynchronous
            else agent("start", messages=history, scope=scope)
        )
        assert result[field] == expected
        assert result["messages"] is history
        assert history.turns[-1]["status"] == "completed"
        state = agent.checkpoint_store.load_state("feedback", "thread", "run")
        assert state["status"] == "completed"
        assert state["messages"]["items"] == history._to_state()["items"]
    finally:
        await agent.aclose()
