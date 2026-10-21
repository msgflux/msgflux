"""Output hook failures must not hide an interruption from the agent."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.exceptions import (
    AbortRequestedError,
    TaskInterruptRequestedError,
    TaskPauseRequestedError,
)
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime.events import EventType, _EventSink, _capture_events


def _tool_call_response():
    calls = ToolCallAggregator()
    calls.process(0, "call-1", "interrupt", "{}")
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _text_response():
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("unexpected continuation")
    return response


def _control_error(control):
    if control == "abort":
        return AbortRequestedError("control marker")
    if control == "interrupt":
        return TaskInterruptRequestedError("task", "control marker")
    if control == "pause":
        return TaskPauseRequestedError("task", "control marker")
    return asyncio.CancelledError("control marker")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("control", "expected_type"),
    [
        ("abort", TaskInterruptRequestedError),
        ("interrupt", TaskInterruptRequestedError),
        ("pause", TaskPauseRequestedError),
        ("cancel", asyncio.CancelledError),
    ],
)
async def test_transform_failure_preserves_control_outcome(
    asynchronous, control, expected_type
):
    error = _control_error(control)

    def interrupt():
        raise error

    async def fail_transform(_outcome):
        raise ValueError("transform failure")

    model = Mock(model_type="chat_completion")
    agent = Agent(name="control-transform", model=model, tools=[interrupt])
    if asynchronous:
        agent.generator.aforward = AsyncMock(
            side_effect=[_tool_call_response(), _text_response()]
        )
    else:
        agent.generator.forward = Mock(
            side_effect=[_tool_call_response(), _text_response()]
        )
    agent.register_lifecycle_hook(
        "transform_tool_output",
        Hook(event="transform_tool_output", handler=fail_transform),
    )
    events = []

    try:
        with _capture_events(_EventSink(events.append)):
            expected_error = pytest.raises(expected_type)
            if control != "cancel":
                expected_error = pytest.raises(expected_type, match="control marker")
            with expected_error as raised:
                if asynchronous:
                    await agent.acall("run")
                else:
                    agent("run")

        if control == "abort":
            assert raised.value.__cause__ is error
        elif control != "cancel":
            assert raised.value is error

        model_call = (
            agent.generator.aforward if asynchronous else agent.generator.forward
        )
        assert model_call.call_count == 1
        failures = [event for event in events if event.type == EventType.HANDLER_ERROR]
        assert len(failures) == 1
        assert failures[0].data["hook"] == "transform_tool_output"
        assert "Tool output processing failed (ValueError)" in failures[0].data["error"]
    finally:
        await agent.aclose()
