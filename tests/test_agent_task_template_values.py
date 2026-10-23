"""Provided task values must not be mistaken for an absent task."""

from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime import ExecutionScope


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize(
    ("task", "template", "variables", "expected"),
    [
        (0, "Task: {}", {}, "0"),
        (False, "Task: {}", {}, "False"),
        ("", "Task: {}", {}, "Task:"),
        ({}, "Hello {{ name | default('friend') }}", {}, "Hello friend"),
        (None, "Task: {}", {}, "Task: {}"),
        (None, "Hello {{ name }}", {"name": "reader"}, "Hello reader"),
        (42, "Task: {}", {}, "42"),
    ],
    ids=[
        "zero",
        "false",
        "empty-string",
        "empty-mapping",
        "absent",
        "vars-only",
        "ordinary-scalar",
    ],
)
async def test_task_template_preserves_provided_values(
    asynchronous, keyword, task, template, variables, expected
):
    model = Mock(model_type="chat_completion")
    model.aclose = AsyncMock()
    store = InMemoryCheckpointStore()
    agent = Agent(
        name="task_template_values",
        model=model,
        checkpoint_store=store,
        templates={"task": template},
    )
    requests = []

    def respond(**kwargs):
        requests.append(kwargs["messages"].to_chatml())
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("ok")
        return response

    agent.generator.forward = Mock(side_effect=respond)
    agent.generator.aforward = AsyncMock(side_effect=respond)
    scope = ExecutionScope(
        namespace="task_template_values", thread_id="thread", run_id="run"
    )
    kwargs = {"scope": scope, "vars": variables}
    args = () if keyword else (task,)
    if keyword:
        kwargs["task"] = task
    try:
        result = (
            await agent.acall(*args, **kwargs)
            if asynchronous
            else agent(*args, **kwargs)
        )
        assert result == "ok"
        assert [m["content"] for m in requests[0] if m["role"] == "user"] == [expected]
        state = store.load_state("task_template_values", "thread", "run")
        assert state["status"] == "completed"
        restored = ChatMessages()
        restored._hydrate_state(state["messages"])
        assert [m["content"] for m in restored.to_chatml() if m["role"] == "user"] == [
            expected
        ]
    finally:
        await agent.aclose()
