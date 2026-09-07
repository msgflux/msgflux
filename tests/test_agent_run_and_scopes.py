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
    assert any(item.get("content") == "Source summary" for item in messages)
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
