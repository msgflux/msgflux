"""Implicit ``None`` results must remain paired in tool history."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import ExecutionScope


def _response(kind, data):
    response = ModelResponse()
    response.set_response_type(kind)
    response.add(data)
    return response


async def _agent_call(agent, message, scope, asynchronous):
    if asynchronous:
        return await agent.acall(message, scope=scope)
    return agent(message, scope=scope)


def _response_mock(asynchronous, **kwargs):
    return (AsyncMock if asynchronous else Mock)(**kwargs)


def _set_model_forward(agent, asynchronous, response_mock):
    method = "aforward" if asynchronous else "forward"
    setattr(agent.generator, method, response_mock)


def _tool_response(api_mode):
    call_id = "call-record"
    calls = ToolCallAggregator(api_mode=api_mode)
    calls.process(0, call_id, "record_effect", "{}")
    response = _response("tool_call", calls)
    if api_mode == "responses":
        response.history_items = [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "record_effect",
                "arguments": "{}",
            }
        ]
    return response


def _history_for_request(call):
    return call.kwargs["messages"]


def _assert_null_tool_output(history, api_mode):
    if api_mode == "responses":
        items = history.to_responses_input()
        outputs = [
            item
            for item in items
            if item.get("type") == "function_call_output"
            and item.get("call_id") == "call-record"
        ]
        assert len(outputs) == 1
        assert outputs[0]["output"] == "null"
    else:
        items = history.to_chatml()
        outputs = [
            item
            for item in items
            if item.get("role") == "tool" and item.get("tool_call_id") == "call-record"
        ]
        assert len(outputs) == 1
        assert outputs[0]["content"] == "null"


def _assert_calls_are_paired(items):
    calls = {item["call_id"] for item in items if item.get("type") == "function_call"}
    outputs = {
        item["call_id"] for item in items if item.get("type") == "function_call_output"
    }
    outputs.update(item["tool_call_id"] for item in items if item.get("role") == "tool")
    assert calls == outputs


@pytest.mark.asyncio
@pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_none_tool_result_survives_checkpoint_reopen_and_new_run(
    tmp_path, api_mode, asynchronous
):
    effects = []

    def record_effect() -> None:
        """Record one effect and intentionally return None."""
        effects.append("done")

    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    checkpoints = SQLiteCheckpointStore(checkpoint_path)

    def make_agent(model):
        return Agent(
            name="agent",
            model=model,
            tools=[record_effect],
            checkpoint_store=checkpoints,
        )

    first_model = Mock(model_type="chat_completion")
    first_model_responses = _response_mock(
        asynchronous,
        side_effect=[
            _tool_response(api_mode),
            _response("text_generation", "first run complete"),
        ],
    )
    agent = make_agent(first_model)
    _set_model_forward(agent, asynchronous, first_model_responses)

    scope = ExecutionScope(thread_id="thread", run_id="run-one")
    continued_agent = None
    try:
        output = await _agent_call(agent, "record this", scope, asynchronous)
        assert output == "first run complete"
        assert effects == ["done"]

        first_followup_history = _history_for_request(
            first_model_responses.call_args_list[1]
        )
        _assert_null_tool_output(first_followup_history, api_mode)

        first_state = checkpoints.load_state("agent", "thread", "run-one")
        assert first_state["status"] == "completed"
        _assert_calls_are_paired(first_state["messages"]["items"])

        await agent.aclose()
        agent = None
        checkpoints.close()
        checkpoints = SQLiteCheckpointStore(checkpoint_path)

        continued_model = Mock(model_type="chat_completion")
        continued_model_response = _response_mock(
            asynchronous,
            return_value=_response("text_generation", "continued"),
        )
        continued_agent = make_agent(continued_model)
        _set_model_forward(continued_agent, asynchronous, continued_model_response)

        continued_scope = ExecutionScope(thread_id="thread", run_id="run-two")
        continued_output = await _agent_call(
            continued_agent, "continue", continued_scope, asynchronous
        )
        assert continued_output == "continued"
        assert effects == ["done"]

        continued_request_history = _history_for_request(
            continued_model_response.call_args_list[0]
        )
        _assert_null_tool_output(continued_request_history, api_mode)
        continued_state = checkpoints.load_state("agent", "thread", "run-two")
        assert continued_state["status"] == "completed"
        _assert_calls_are_paired(continued_state["messages"]["items"])
        assert (
            sum(
                item.get("type") == "function_call"
                and item.get("call_id") == "call-record"
                for item in continued_state["messages"]["items"]
            )
            == 1
        )
    finally:
        if agent is not None:
            await agent.aclose()
        if continued_agent is not None:
            await continued_agent.aclose()
        checkpoints.close()
