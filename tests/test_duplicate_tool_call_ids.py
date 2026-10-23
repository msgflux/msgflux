"""Reject ambiguous call identities before performing a batch's effects."""

import asyncio

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskInterruptRequestedError
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime import ExecutionScope


def _tool_response(call_id: str, value: str) -> ModelResponse:
    calls = ToolCallAggregator(api_mode="chat_completions")
    calls.process(0, call_id, "record", f'{{"value":"{value}"}}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    # Chat-completions providers omit function_call history unless commentary
    # is present; the agent reconstructs calls from the decoded intents.
    response.history_items = []
    return response


def _text_response(value: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(value)
    return response


def _call_and_output_ids(items):
    calls = [
        item.get("call_id") or item.get("id")
        for item in items
        if item.get("type") == "function_call"
    ]
    outputs = [
        item.get("call_id")
        for item in items
        if item.get("type") == "function_call_output"
    ]
    return calls, outputs


class _HistoryModel:
    model_type = "chat_completion"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.history = []

    def __call__(self, **kwargs):
        self.history.append(kwargs["messages"].to_items())
        return next(self.responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_duplicate_ids_reject_entire_tool_batch_before_effects(asynchronous):
    effects = []

    def record(value: str) -> str:
        """Record a visible effect."""
        effects.append(value)
        return value

    library = ToolLibrary("tools", [record])
    calls = [
        ("same-id", "record", {"value": "A"}),
        ("same-id", "record", {"value": "B"}),
    ]
    with pytest.raises(ValueError, match=r"Duplicate tool call IDs.*same-id"):
        if asynchronous:
            await library.acall(calls)
        else:
            library(calls)
    assert effects == []

    unique_calls = [
        ("first", "record", {"value": "A"}),
        ("second", "record", {"value": "B"}),
    ]
    result = (
        await library.acall(unique_calls) if asynchronous else library(unique_calls)
    )
    assert sorted(effects) == ["A", "B"]
    assert [call.result for call in result.tool_calls] == ["A", "B"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_agent_rejects_duplicate_ids_before_history_and_can_resume(
    asynchronous, streaming
):
    effects = []

    def record(value: str) -> str:
        """Record a visible effect."""
        effects.append(value)
        return value

    calls = ToolCallAggregator()
    calls.process(0, "same-id", "record", '{"value":"A"}')
    calls.process(1, "same-id", "record", '{"value":"B"}')
    invalid = ModelResponse()
    invalid.set_response_type("tool_call")
    invalid.add(calls)
    invalid.history_items = [
        {
            "type": "function_call",
            "call_id": "same-id",
            "name": "record",
            "arguments": arguments,
        }
        for arguments in ('{"value":"A"}', '{"value":"B"}')
    ]
    if streaming:
        invalid = ModelStreamResponse(mode="async" if asynchronous else "sync")
        invalid.set_response_type("tool_call")
        invalid.data = calls
        invalid.finish()
    valid = ModelResponse()
    valid.set_response_type("text_generation")
    valid.add("recovered")
    responses = iter([invalid, valid])

    class Model:
        model_type = "chat_completion"

        def __init__(self):
            self.history = []

        def __call__(self, **kwargs):
            self.history.append(kwargs["messages"].to_items())
            return next(responses)

        async def acall(self, **kwargs):
            return self(**kwargs)

    model = Model()
    checkpoints = InMemoryCheckpointStore()
    agent = Agent(
        name="main", model=model, tools=[record], checkpoint_store=checkpoints
    )
    scope = ExecutionScope(thread_id="thread", run_id="run")
    try:
        with pytest.raises(ValueError, match=r"Duplicate tool call IDs.*same-id"):
            if asynchronous:
                await agent.acall("work", scope=scope)
            else:
                agent("work", scope=scope)
        assert effects == []
        failed = checkpoints.load_state("main", "thread", "run")
        assert failed["status"] == "failed"
        assert not any(
            item.get("type") == "function_call" for item in failed["messages"]["items"]
        )

        result = (
            await agent.acall("retry", scope=scope)
            if asynchronous
            else agent("retry", scope=scope)
        )
        assert result == "recovered"
        assert len(model.history) == 2
        assert not any(
            item.get("type") == "function_call" for item in model.history[-1]
        )
        assert effects == []
        assert checkpoints.load_state("main", "thread", "run")["status"] == "completed"
    finally:
        await agent.aclose()


def test_reused_call_id_across_rounds_keeps_each_call_paired():
    effects = []

    def record(value: str) -> str:
        """Record a visible effect."""
        effects.append(value)
        return value

    model = _HistoryModel(
        [
            _tool_response("reused-id", "A"),
            _tool_response("reused-id", "B"),
            _text_response("done"),
        ]
    )
    checkpoints = InMemoryCheckpointStore()

    def make_agent(model):
        return Agent(
            name="main", model=model, tools=[record], checkpoint_store=checkpoints
        )

    agent = make_agent(model)
    continued_agent = None
    try:
        assert (
            agent("work", scope=ExecutionScope(thread_id="thread", run_id="run"))
            == "done"
        )
        assert effects == ["A", "B"]

        state = checkpoints.load_state("main", "thread", "run")
        assert state["status"] == "completed"
        calls, outputs = _call_and_output_ids(state["messages"]["items"])
        assert calls == ["reused-id", "reused-id"]
        assert outputs == ["reused-id", "reused-id"]
        assert [
            item["type"]
            for item in state["messages"]["items"]
            if item.get("type") in {"function_call", "function_call_output"}
        ] == [
            "function_call",
            "function_call_output",
            "function_call",
            "function_call_output",
        ]

        continued_model = _HistoryModel([_text_response("continued")])
        continued_agent = make_agent(continued_model)
        assert (
            continued_agent(
                "continue", scope=ExecutionScope(thread_id="thread", run_id="run-two")
            )
            == "continued"
        )
        replayed_calls, replayed_outputs = _call_and_output_ids(
            continued_model.history[0]
        )
        assert replayed_calls == ["reused-id", "reused-id"]
        assert replayed_outputs == ["reused-id", "reused-id"]
        continued_state = checkpoints.load_state("main", "thread", "run-two")
        assert continued_state["status"] == "completed"
        assert _call_and_output_ids(continued_state["messages"]["items"]) == (
            ["reused-id", "reused-id"],
            ["reused-id", "reused-id"],
        )
    finally:
        asyncio.run(agent.aclose())
        if continued_agent is not None:
            asyncio.run(continued_agent.aclose())


def test_orphan_output_does_not_cancel_reused_call_pairing():
    effects = []

    def record(value: str) -> str:
        """Record a visible effect."""
        effects.append(value)
        return value

    store = InMemoryCheckpointStore()

    def make_agent(model):
        return Agent(name="main", model=model, tools=[record], checkpoint_store=store)

    first = make_agent(
        _HistoryModel([_tool_response("reused-id", "A"), _text_response("done")])
    )
    second = None
    try:
        first("work", scope=ExecutionScope(thread_id="thread", run_id="run"))
        state = store.load_state("main", "thread", "run")
        state["messages"]["items"].append(
            {
                "type": "function_call_output",
                "call_id": "reused-id",
                "output": "orphan-output-from-original-bug",
                "item_id": "itm_persisted_orphan_output",
            }
        )
        store.save_state("main", "thread", "run", state)

        second_model = _HistoryModel(
            [_tool_response("reused-id", "B"), _text_response("continued")]
        )
        second = make_agent(second_model)
        second("continue", scope=ExecutionScope(thread_id="thread", run_id="run-two"))

        assert effects == ["A", "B"]
        calls, outputs = _call_and_output_ids(second_model.history[1])
        assert calls == ["reused-id", "reused-id"]
        assert outputs == ["reused-id", "reused-id", "reused-id"]
        assert [
            item["type"]
            for item in second_model.history[1]
            if item.get("type") in {"function_call", "function_call_output"}
        ] == [
            "function_call",
            "function_call_output",
            "function_call_output",
            "function_call",
            "function_call_output",
        ]

        final = store.load_state("main", "thread", "run-two")["messages"]["items"]
        assert _call_and_output_ids(final) == (calls, outputs)
    finally:
        asyncio.run(first.aclose())
        if second is not None:
            asyncio.run(second.aclose())


def test_agent_interrupt_closes_reused_id_and_continues():
    effects = []

    def record(value: str) -> str:
        """Record a visible effect or interrupt the second round."""
        effects.append(value)
        if value == "B":
            raise TaskInterruptRequestedError("run-two", "operator interrupted")
        return value

    store = InMemoryCheckpointStore()

    def make_agent(model):
        return Agent(name="main", model=model, tools=[record], checkpoint_store=store)

    first = make_agent(
        _HistoryModel([_tool_response("reused-id", "A"), _text_response("done")])
    )
    second = None
    third = None
    try:
        first("work", scope=ExecutionScope(thread_id="thread", run_id="run-one"))

        second = make_agent(_HistoryModel([_tool_response("reused-id", "B")]))
        with pytest.raises(TaskInterruptRequestedError):
            second(
                "interrupt", scope=ExecutionScope(thread_id="thread", run_id="run-two")
            )

        interrupted_state = store.load_state("main", "thread", "run-two")
        assert interrupted_state["status"] == "interrupted"
        calls, outputs = _call_and_output_ids(interrupted_state["messages"]["items"])
        assert calls == ["reused-id", "reused-id"]
        assert outputs == ["reused-id", "reused-id"]
        interrupted = [
            item
            for item in interrupted_state["messages"]["items"]
            if item.get("type") == "function_call_output"
            and item.get("call_id") == "reused-id"
        ][-1]
        assert "interrupted" in interrupted["output"]

        third_model = _HistoryModel([_text_response("continued")])
        third = make_agent(third_model)
        assert (
            third(
                "continue", scope=ExecutionScope(thread_id="thread", run_id="run-three")
            )
            == "continued"
        )
        assert _call_and_output_ids(third_model.history[0]) == (
            ["reused-id", "reused-id"],
            ["reused-id", "reused-id"],
        )
        continued = store.load_state("main", "thread", "run-three")
        assert continued["status"] == "completed"
        assert _call_and_output_ids(continued["messages"]["items"]) == (
            ["reused-id", "reused-id"],
            ["reused-id", "reused-id"],
        )
    finally:
        asyncio.run(first.aclose())
        if second is not None:
            asyncio.run(second.aclose())
        if third is not None:
            asyncio.run(third.aclose())
