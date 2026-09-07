import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.runtime.context_scopes import (
    ContextScopeCommand,
    ContextScopeConflictError,
    ContextScopeController,
)


def test_nested_context_scopes_restore_parent_and_close_idempotently():
    messages = ChatMessages(thread_id="thread", namespace="agent")
    messages.add_user("root")
    controller = ContextScopeController()
    opened = controller.open(messages, "research", summary="Research context")
    assert opened.branch_id == "research"
    messages.add_user("inside")
    nested = controller.open(messages, "source")
    assert nested.branch_id == "research/source"
    messages.add_user("nested")
    closed = controller.close(messages, "source", summary="Source summary")
    assert closed.branch_id == "research"
    assert any(
        item.get("role") == "assistant" and item.get("content") == "Source summary"
        for item in messages
    )
    duplicate = controller.close(messages, "source")
    assert duplicate.changed is False
    controller.close(messages, "research", summary="Research summary")
    again = controller.close(messages)
    assert again.changed is False
    assert ContextScopeController.active_scope(messages) == "root"


def test_context_scope_revision_conflict_does_not_mutate_history():
    messages = ChatMessages()
    messages.add_user("root")
    controller = ContextScopeController()
    controller.open(messages, "one")
    before = messages.copy()
    with pytest.raises(ContextScopeConflictError):
        controller.close(messages, expected_revision=0)
    assert list(messages) == list(before)


def test_scope_command_is_applied_only_when_caller_reaches_boundary():
    messages = ChatMessages()
    messages.add_user("root")
    controller = ContextScopeController()
    command = ContextScopeCommand(action="open", name="research")
    result = controller.apply_command(messages, command.as_dict())
    assert result.branch_id == "research"
    assert ContextScopeController.active_scope(messages) == "research"


from unittest.mock import Mock

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.context import ExecutionScope


def _scope_response(call_id="scope_call"):
    calls = ToolCallAggregator()
    calls.process(0, call_id, "open_context_scope", '{"name":"research"}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    response.reasoning = None
    return response


def _text_response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    return response


def test_agent_applies_scope_command_after_tool_history_is_settled():
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="scope_agent",
        model=model,
        tools=[
            __import__(
                "msgflux.tools.builtin.context_scope", fromlist=["open_context_scope"]
            ).open_context_scope
        ],
    )
    agent.generator.forward = Mock(
        side_effect=[_scope_response(), _text_response("inside")]
    )
    messages = ChatMessages()
    result = agent(
        "Investigate",
        messages=messages,
        scope=ExecutionScope(namespace="scope_agent", thread_id="thread", run_id="run"),
    )
    assert result == "inside"
    assert ContextScopeController.active_scope(messages) == "research"
    assert (
        messages.metadata["runtime"]["context_scopes"]["stack"][-1]["name"]
        == "research"
    )
    assert any(item.get("type") == "function_call_output" for item in messages)


@pytest.mark.asyncio
async def test_async_agent_applies_scope_command_after_settled_history():
    from unittest.mock import AsyncMock

    model = Mock()
    model.model_type = "chat_completion"
    from msgflux.tools.builtin.context_scope import open_context_scope

    agent = Agent(name="async_scope_agent", model=model, tools=[open_context_scope])
    agent.generator.aforward = AsyncMock(
        side_effect=[_scope_response("async_scope"), _text_response("inside")]
    )
    messages = ChatMessages()
    result = await agent.acall(
        "Investigate",
        messages=messages,
        scope=ExecutionScope(
            namespace="async_scope_agent", thread_id="thread", run_id="run"
        ),
    )
    assert result == "inside"
    assert ContextScopeController.active_scope(messages) == "research"


def test_scope_transition_is_exclusive_before_any_tool_executes():
    from msgflux.models.tool_call_agg import ToolCallAggregator

    calls = []

    def other(value: str) -> str:
        calls.append(value)
        return value

    model = Mock()
    model.model_type = "chat_completion"
    from msgflux.tools.builtin.context_scope import open_context_scope

    agent = Agent(
        name="exclusive_scope_agent", model=model, tools=[open_context_scope, other]
    )
    aggregate = ToolCallAggregator()
    aggregate.process(0, "scope_call", "open_context_scope", '{"name":"research"}')
    aggregate.process(1, "other_call", "other", '{"value":"must_not_run"}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(aggregate)
    response.reasoning = None
    agent.generator.forward = Mock(return_value=response)

    with pytest.raises(ValueError, match="exclusive tool calls"):
        agent("Investigate")
    assert calls == []


def test_scope_stack_and_head_survive_checkpoint_resume():
    from msgflux.data.stores import InMemoryCheckpointStore
    from msgflux.tools.builtin.context_scope import open_context_scope

    store = InMemoryCheckpointStore()
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="durable_scope_agent",
        model=model,
        tools=[open_context_scope],
        checkpoint_store=store,
    )
    agent.generator.forward = Mock(
        side_effect=[_scope_response(), _text_response("inside")]
    )
    messages = ChatMessages()
    scope = ExecutionScope(
        namespace="durable_scope_agent", thread_id="thread", run_id="run"
    )
    assert agent("Investigate", messages=messages, scope=scope) == "inside"
    state = store.load_state("durable_scope_agent", "thread", "run")
    runtime = state["messages"]["metadata"]["runtime"]["context_scopes"]
    assert runtime["active"] == "research"
    assert runtime["branches"]["research"]["parent"] == "root"
    assert runtime["revision"] == 1
