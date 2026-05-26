"""Live E2E test for tools with msgspec.Struct parameters.

Requires: OPENAI_API_KEY in environment or .env.
"""

import os

import msgspec
import pytest

import msgflux as mf
from msgflux import nn

mf.load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required for live OpenAI integration tests.",
)


class TodoItem(msgspec.Struct):
    content: str
    active_form: str
    status: str


_stored_todos: list[TodoItem] = []


def store_todos(todos: list[TodoItem]) -> dict:
    """Store TODO items.

    Args:
        todos: TODO items to store. Each item must include content,
            active_form, and status.
    """
    assert all(isinstance(todo, TodoItem) for todo in todos)
    _stored_todos.extend(todos)
    return {"stored": len(todos)}


class TodoAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini", max_tokens=300)
    system_message = "You are a TODO extraction assistant."
    instructions = (
        "Extract TODO items from the user message and call store_todos exactly once. "
        "Use status='pending' for new TODOs and active_form as the action phrase."
    )
    message_fields = {"task": "user.text"}
    tools = [store_todos]
    response_mode = "response"


def test_openai_agent_calls_tool_with_msgspec_struct_list_param():
    _stored_todos.clear()
    agent = TodoAgent()
    msg = mf.Message()
    msg.set(
        "user.text",
        "Add two todos: write unit tests and update the documentation.",
    )

    agent(msg, messages=[])

    assert len(_stored_todos) == 2
    assert all(isinstance(todo, TodoItem) for todo in _stored_todos)
    assert {todo.status for todo in _stored_todos} == {"pending"}
    assert any("test" in todo.content.lower() for todo in _stored_todos)
    assert any("documentation" in todo.content.lower() for todo in _stored_todos)
