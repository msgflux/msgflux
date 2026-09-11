"""End-to-end invariants shared by native and structured tool loops."""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.generation.reasoning.react import ReAct
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.context import ExecutionScope
from msgflux.tools.config import tool_config


def response(kind, data):
    result = ModelResponse()
    result.set_response_type(kind)
    result.add(data)
    return result


def make_agent(**kwargs):
    model = Mock(model_type="chat_completion")
    return Agent(name="integrity", model=model, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["request", "tool"])
async def test_external_cancellation_persists_interruption(phase):
    entered = asyncio.Event()
    hooks = []

    async def pending():
        """Wait for external cancellation."""
        entered.set()
        await asyncio.Future()

    store = InMemoryCheckpointStore()
    agent = make_agent(
        checkpoint_store=store,
        tools=[pending],
        hooks=[
            Hook(event="after_run_end", handler=lambda ctx: hooks.append(ctx.outcome))
        ],
    )
    if phase == "request":

        async def request(**kwargs):
            return await pending()

        agent.generator.aforward = request
    else:
        calls = ToolCallAggregator()
        calls.process(0, "call_pending", "pending", "{}")
        agent.generator.aforward = AsyncMock(return_value=response("tool_call", calls))
    scope = ExecutionScope(thread_id="thread", run_id="run", namespace="integrity")
    task = asyncio.create_task(agent.acall("start", scope=scope))
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    state = store.load_state("integrity", "thread", "run")
    assert state["status"] == "interrupted"
    assert hooks == ["interrupted"]
    history = ChatMessages()
    history._hydrate_state(state["messages"])
    assert history.get_active_turn() is None
    if phase == "tool":
        assert any(
            item.get("call_id") == "call_pending"
            and item["type"] == "function_call_output"
            for item in history
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
async def test_direct_return_persists_paired_calls(asynchronous, api_mode):
    @tool_config(return_direct=True)
    def lookup() -> str:
        """Return a stable answer."""
        return "found"

    store = InMemoryCheckpointStore()
    agent = make_agent(tools=[lookup], checkpoint_store=store)
    calls = ToolCallAggregator(api_mode=api_mode)
    calls.process(0, "call_lookup", "lookup", "{}")
    agent.generator.forward = Mock(return_value=response("tool_call", calls))
    agent.generator.aforward = AsyncMock(return_value=response("tool_call", calls))
    scope = ExecutionScope(thread_id="thread", run_id="run", namespace="integrity")
    if asynchronous:
        await agent.acall("start", scope=scope)
    else:
        agent("start", scope=scope)
    state = store.load_state("integrity", "thread", "run")
    items = state["messages"]["items"]
    assert state["status"] == "completed"
    assert [item["type"] for item in items if item.get("call_id") == "call_lookup"] == [
        "function_call",
        "function_call_output",
    ]
    assert (
        next(item for item in items if item["type"] == "function_call_output")["output"]
        == "found"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_flow_control_uses_feedback_after_recording_observations(asynchronous):
    def lookup() -> str:
        """Return a stable answer."""
        return "found"

    def feedback(ctx):
        assert ctx.intents[0].name == "lookup"
        assert ctx.outcomes[0].result == "found"
        assert "found" in str(list(ctx.messages))
        return replace(ctx, action="return", output={"answer": "custom"})

    agent = make_agent(
        tools=[lookup],
        generation_schema=ReAct,
        hooks=[Hook(event="resolve_tool_feedback", handler=feedback)],
    )
    data = {"thought": "look", "actions": [{"name": "lookup", "arguments": {}}]}
    agent.generator.forward = Mock(return_value=response("structured", data))
    agent.generator.aforward = AsyncMock(return_value=response("structured", data))
    result = await agent.acall("start") if asynchronous else agent("start")
    assert result == {"answer": "custom"}
