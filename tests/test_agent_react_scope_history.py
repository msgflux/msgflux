"""ReAct history survives ordinary assistant summaries from scope closes."""

from unittest.mock import AsyncMock, Mock

import msgspec
import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.generation.reasoning.react import ReAct
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.tools.builtin import close_context_scope, open_context_scope


def _scope_step(name, arguments):
    response = ModelResponse()
    response.set_response_type("structured")
    response.add(
        {
            "thought": f"Call {name}",
            "actions": [{"name": name, "arguments": arguments}],
        }
    )
    return response


def _final_response():
    response = ModelResponse()
    response.set_response_type("structured")
    response.add({"thought": "Done", "final_answer": "done"})
    return response


def _responses():
    return [
        _scope_step("open_context_scope", {"name": "research"}),
        _scope_step("open_context_scope", {"name": "source"}),
        _scope_step(
            "close_context_scope", {"name": "source", "summary": "Source summary"}
        ),
        _scope_step("close_context_scope", {"summary": "Research summary"}),
        _final_response(),
    ]


def _scope_agent(requests):
    responses = _responses()

    def forward(*args, **kwargs):
        messages = next(
            item for item in [*args, *kwargs.values()] if isinstance(item, ChatMessages)
        )
        requests.append(messages.to_chatml())
        return responses[len(requests) - 1]

    agent = Agent(
        name="react_scope_history",
        model=Mock(model_type="chat_completion"),
        tools=[open_context_scope, close_context_scope],
        generation_schema=ReAct,
    )
    agent.generator.forward = Mock(side_effect=forward)
    agent.generator.aforward = AsyncMock(side_effect=forward)
    return agent


def _react_steps(messages):
    steps = []
    for item in messages:
        content = item.get("content")
        if not isinstance(content, str):
            continue
        try:
            decoded = msgspec.json.decode(content)
        except msgspec.DecodeError:
            continue
        if (
            isinstance(decoded, list)
            and decoded
            and all(
                isinstance(step, dict)
                and {"thought", "actions", "observations"} <= step.keys()
                for step in decoded
            )
        ):
            steps.extend(decoded)
    return steps


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_nested_scope_summaries_do_not_break_react_history(asynchronous):
    """A single run can open/close nested scopes and still reach the final answer."""
    requests = []
    agent = _scope_agent(requests)
    messages = ChatMessages()
    try:
        if asynchronous:
            result = await agent.acall("Investigate", messages=messages)
        else:
            result = agent("Investigate", messages=messages)

        assert result == {"thought": "Done", "final_answer": "done"}
        assert len(requests) == 5
        assert [step["actions"][0]["name"] for step in _react_steps(requests[2])] == [
            "open_context_scope",
            "open_context_scope",
        ]
        # The request after closing the inner scope ends with an ordinary
        # assistant summary; build_history must append instead of decoding it.
        outer_close_request = requests[3]
        outer_close_assistant = [
            item.get("content")
            for item in outer_close_request
            if item.get("role") == "assistant"
        ]
        assert outer_close_assistant[-1] == "Source summary"
        final_request_assistant = [
            item.get("content")
            for item in requests[4]
            if item.get("role") == "assistant"
        ]
        assert final_request_assistant[-1] == "Research summary"
        assert _react_steps(requests[4])
        contents = [
            item.get("content") for item in messages if item.get("role") == "assistant"
        ]
        assert "Research summary" in contents
        assert _react_steps(messages)
        branches = messages.metadata["runtime"]["context_scopes"]["branches"]
        assert "Source summary" in str(branches["research"]["items"])
    finally:
        await agent.aclose()
