"""Reject ambiguous call identities before performing a batch's effects."""

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime import ExecutionScope


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
