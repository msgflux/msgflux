"""ToolFlowControl history keeps settled siblings when a batch pauses."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import msgspec

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.generation.control_flow import ToolFlowControl, ToolFlowResult
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime.context import ExecutionScope
from msgflux.utils.chat import ChatBlock


class _BatchFlow(msgspec.Struct, ToolFlowControl):
    tools_template = "Tools: {{ tool_schemas|length }}"
    injected = []
    histories = []

    @classmethod
    def extract_flow_result(cls, raw_response):
        if raw_response.get("done"):
            return ToolFlowResult(
                is_complete=True,
                tool_calls=None,
                reasoning=None,
                final_response=raw_response.get("answer"),
            )
        return ToolFlowResult(
            is_complete=False,
            tool_calls=[
                (call["id"], call["name"], call.get("arguments", {}))
                for call in raw_response["calls"]
            ],
            reasoning=None,
            final_response=None,
        )

    @classmethod
    def inject_results(cls, raw_response, tool_results):
        observations = []
        for call in raw_response["calls"]:
            outcome = tool_results.get_by_id(call["id"])
            observations.append(
                {
                    "id": call["id"],
                    "result": outcome.result,
                    "error": outcome.error,
                }
            )
        raw_response["observations"] = observations
        cls.injected.append(deepcopy(observations))
        return raw_response

    @classmethod
    def build_history(cls, raw_response, messages):
        cls.histories.append(deepcopy(raw_response))
        updated = messages.copy()
        updated.append(ChatBlock.assist(raw_response))
        return updated


def _response(data):
    response = ModelResponse()
    response.set_response_type("structured")
    response.add(data)
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_flow_batch_pause_records_completed_sibling_before_resuming(
    tmp_path: Path, asynchronous: bool
):
    effect_file = tmp_path / "effect.log"
    effect_calls = []
    sibling_calls = []
    pause = TaskPauseRequestedError("run", "post-effect hold")
    should_pause = True
    _BatchFlow.injected = []
    _BatchFlow.histories = []

    def write_effect(value: str):
        with effect_file.open("a") as stream:
            stream.write(f"{value}\n")
        return "written"

    def read_effect():
        sibling_calls.append("read")
        return "read result"

    def after_tool(context):
        nonlocal should_pause
        if context.tool_name == "write_effect" and should_pause:
            should_pause = False
            effect_calls.append("pause-after-write")
            raise pause

    agent = Agent(
        name="flow-batch",
        model=Mock(model_type="chat_completion"),
        tools=[write_effect, read_effect],
        generation_schema=_BatchFlow,
        checkpoint_store=InMemoryCheckpointStore(),
        hooks=[Hook(event="after_tool", handler=after_tool)],
    )
    tool_batch = _response(
        {
            "calls": [
                {
                    "id": "write-call",
                    "name": "write_effect",
                    "arguments": {"value": "once"},
                },
                {"id": "read-call", "name": "read_effect", "arguments": {}},
            ]
        }
    )
    final_response = _response({"done": True, "answer": "resumed"})
    if asynchronous:
        agent.generator.aforward = AsyncMock(side_effect=[tool_batch, final_response])
    else:
        agent.generator.forward = Mock(side_effect=[tool_batch, final_response])

    scope = ExecutionScope(thread_id="thread", namespace="flow-batch", run_id="run")
    try:
        with pytest.raises(TaskPauseRequestedError) as raised:
            if asynchronous:
                await agent.acall("start", scope=scope)
            else:
                agent("start", scope=scope)
        assert raised.value is pause

        # The whole batch settles before cooperative pause escapes. Its sibling
        # result is available through ToolFlowControl's regular result/history path.
        assert sibling_calls == ["read"]
        assert effect_calls == ["pause-after-write"]
        assert effect_file.read_text().splitlines() == ["once"]
        assert len(_BatchFlow.injected) == len(_BatchFlow.histories) == 1
        observations = _BatchFlow.injected[0]
        write_observation, read_observation = observations
        assert write_observation["id"] == "write-call"
        assert write_observation["result"] is None
        assert write_observation["error"] is not None
        assert read_observation == {
            "id": "read-call",
            "result": "read result",
            "error": None,
        }

        checkpoint = agent.checkpoint_store.load_state("flow-batch", "thread", "run")
        assert checkpoint["status"] == "paused"
        saved_history = checkpoint["messages"]["items"]
        flow_message = next(
            item
            for item in saved_history
            if item.get("role") == "assistant" and item.get("content")
        )
        saved_flow = json.loads(flow_message["content"])
        assert saved_flow["observations"][1]["result"] == "read result"

        # A later final model response does not cause the runtime to replay either
        # tool. A model may still choose to issue another call after resumption.
        if asynchronous:
            assert await agent.acall("resume", scope=scope) == "resumed"
        else:
            assert agent("resume", scope=scope) == "resumed"
        assert effect_file.read_text().splitlines() == ["once"]
        assert sibling_calls == ["read"]
    finally:
        await agent.aclose()
