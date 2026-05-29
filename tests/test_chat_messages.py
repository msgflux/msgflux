from msgflux.chat_messages import ChatMessages
from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_scope,
    session_context,
)


def test_default_execution_scope_is_available():
    scope = get_execution_scope()

    assert scope.session_id == "default"
    assert scope.namespace == "default"


def test_execution_context_accepts_scope_and_explicit_overrides():
    base_scope = ExecutionScope(
        session_id="session_1",
        namespace="root",
        run_id="run_1",
    )

    with execution_context(scope=base_scope, namespace="agent"):
        scope = get_execution_scope()

    assert scope.session_id == "session_1"
    assert scope.namespace == "agent"
    assert scope.run_id == "run_1"
    assert scope.root_run_id == "run_1"


def test_chat_messages_inherit_session_context():
    with session_context(session_id="user_42", namespace="support"):
        chat = ChatMessages()

    assert chat.session_id == "user_42"
    assert chat.namespace == "support"


def test_chat_messages_state_roundtrip():
    chat = ChatMessages(session_id="session_1", namespace="agent:test")
    turn_id = chat.begin_turn(inputs="What is 2+2?", vars={"temperature": 0.2})
    chat.add_user("What is 2+2?")
    chat.add_assistant_response("4", reasoning_content="Simple arithmetic.")
    chat.end_turn(assistant_output="4", response_type="text_generation")

    state = chat._to_state()

    restored = ChatMessages()
    restored._hydrate_state(state)

    assert restored.session_id == "session_1"
    assert restored.namespace == "agent:test"
    assert restored.turns[0]["turn_id"] == turn_id
    assert restored.to_chatml()[0]["content"] == "What is 2+2?"
    assert restored.to_chatml()[-1]["content"] == "4"


def test_disabled_chat_message_is_persisted_but_not_rendered():
    chat = ChatMessages(session_id="session_1", namespace="agent:test")
    chat.add_user("old context")
    chat.add_user("current context")
    chat.set_item_disabled(0, reason="compacted")

    state = chat._to_state()
    restored = ChatMessages()
    restored._hydrate_state(state)

    assert restored.to_items()[0]["disabled"] is True
    assert restored.to_items()[0]["metadata"]["disabled_reason"] == "compacted"
    assert [item["content"] for item in restored.to_chatml()] == ["current context"]
