"""ReAct forwards actual tool outcomes to the next request and checkpoint."""

from unittest.mock import AsyncMock, Mock

import msgspec
import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.generation.reasoning.react import ReAct
from msgflux.runtime import ExecutionScope
from tests.test_agent_structured_history import _agent, _response


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("value", "raises"),
    [
        pytest.param(0, False, id="zero"),
        pytest.param(False, False, id="false"),
        pytest.param("", False, id="empty-string"),
        pytest.param([], False, id="empty-list"),
        pytest.param({}, False, id="empty-mapping"),
        pytest.param(None, False, id="none"),
        pytest.param("found", False, id="truthy"),
        pytest.param(None, True, id="error"),
    ],
)
async def test_react_observations_preserve_tool_outcomes(asynchronous, value, raises):
    def lookup():
        """Look up a value for this task."""
        if raises:
            raise RuntimeError("lookup failed")
        return value

    responses = [
        _response(
            {"thought": "look up", "actions": [{"name": "lookup", "arguments": {}}]}
        ),
        _response({"thought": "done", "final_answer": "ok"}),
    ]
    store = InMemoryCheckpointStore()
    agent = _agent(responses[0], store=store, generation_schema=ReAct, tools=[lookup])
    requests = []

    def forward(*args, **kwargs):
        messages = next(
            item for item in [*args, *kwargs.values()] if isinstance(item, ChatMessages)
        )
        requests.append(messages.to_chatml())
        return responses[len(requests) - 1]

    agent.generator.forward = Mock(side_effect=forward)
    agent.generator.aforward = AsyncMock(side_effect=forward)
    history = ChatMessages()
    scope = ExecutionScope(thread_id="thread", run_id="run")
    expected = "lookup failed" if raises else value
    try:
        result = (
            await agent.acall("go", messages=history, scope=scope)
            if asynchronous
            else agent("go", messages=history, scope=scope)
        )
        assert result == {"thought": "done", "final_answer": "ok"}
        assert len(requests) == 2
        steps = msgspec.json.decode(
            next(
                item["content"]
                for item in requests[1]
                if item.get("role") == "assistant"
            )
        )
        observations = steps[-1]["observations"]
        assert observations == [{"tool": "lookup", "result": expected}]
        assert type(observations[0]["result"]) is type(expected)
        checkpoint = store.load_state("structured", "thread", "run")
        assert checkpoint["status"] == "completed"
        saved_steps = msgspec.json.decode(
            next(
                item["content"]
                for item in checkpoint["messages"]["items"]
                if item.get("role") == "assistant"
            )
        )
        assert saved_steps[-1]["observations"] == observations
        assert type(saved_steps[-1]["observations"][0]["result"]) is type(expected)
    finally:
        await agent.aclose()
