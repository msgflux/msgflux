"""Runtime after-tool policy failures stay visible without replaying effects."""

from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent, ToolPolicy
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.events import EventType, _EventSink, _capture_events


def _tool_response():
    calls = ToolCallAggregator()
    calls.process(0, "call-policy", "effect", "{}")
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _text_response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_after_tool_policy_failure_is_reported_without_repeating_effect(
    tmp_path, asynchronous
):
    effects = []
    policy_error = RuntimeError("after policy failed")

    def effect() -> str:
        effects.append("completed")
        return "effect result"

    class BrokenPolicy(ToolPolicy):
        def __init__(self):
            super().__init__("broken_after_tool")

        async def after_tool(self, payload):
            del payload
            raise policy_error

    checkpoint = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    agent = Agent(
        name="after_tool_policy_failure",
        model=Mock(model_type="chat_completion"),
        tools=[effect],
        checkpoint_store=checkpoint,
    )
    agent.tool_library.runtime_extensions.register(BrokenPolicy())
    model_call = AsyncMock(side_effect=[_tool_response(), _text_response("answer")])
    if asynchronous:
        agent.generator.aforward = model_call
    else:
        agent.generator.forward = Mock(
            side_effect=[_tool_response(), _text_response("answer")]
        )

    events = []
    try:
        with _capture_events(_EventSink(events.append)):
            if asynchronous:
                result = await agent.acall(
                    "run", scope=ExecutionScope(thread_id="thread", run_id="run")
                )
            else:
                result = agent(
                    "run", scope=ExecutionScope(thread_id="thread", run_id="run")
                )

        assert result == "answer"
        assert effects == ["completed"]
        policy_events = [
            event for event in events if event.type == EventType.HANDLER_ERROR
        ]
        assert len(policy_events) == 1
        assert policy_events[0].data == {
            "hook": "after_tool",
            "error": str(policy_error),
            "tool_name": "effect",
            "tool_call_id": "call-policy",
        }
        tool_end = next(
            event
            for event in events
            if event.type == EventType.TOOL_END
            and event.data["tool_call_id"] == "call-policy"
        )
        assert tool_end.data["result"] == "effect result"
        assert tool_end.data["error"] is None

        saved = checkpoint.load_state("after_tool_policy_failure", "thread", "run")
        assert saved["status"] == "completed"
        tool_messages = [
            item
            for item in saved["messages"]["items"]
            if item.get("type") == "function_call_output"
        ]
        assert len(tool_messages) == 1
        assert tool_messages[0]["output"] == "effect result"
    finally:
        await agent.aclose()
        checkpoint.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_after_tool_policy_control_still_propagates(asynchronous):
    effects = []
    control = TaskPauseRequestedError("task", "pause after effect")

    def effect() -> str:
        effects.append("completed")
        return "effect result"

    class PausingPolicy(ToolPolicy):
        def __init__(self):
            super().__init__("pausing_after_tool")

        async def after_tool(self, payload):
            del payload
            raise control

    agent = Agent(
        name="after_tool_policy_control",
        model=Mock(model_type="chat_completion"),
        tools=[effect],
    )
    agent.tool_library.runtime_extensions.register(PausingPolicy())
    if asynchronous:
        agent.generator.aforward = AsyncMock(return_value=_tool_response())
    else:
        agent.generator.forward = Mock(return_value=_tool_response())

    try:
        with pytest.raises(TaskPauseRequestedError) as raised:
            if asynchronous:
                await agent.acall("run")
            else:
                agent("run")
        assert raised.value is control
        assert effects == ["completed"]
    finally:
        await agent.aclose()
