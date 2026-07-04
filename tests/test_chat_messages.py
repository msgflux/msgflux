from msgflux.chat_messages import ChatMessages
from msgflux.runtime import AbortSignal
from msgflux.runtime.context import (
    DEFAULT_NAMESPACE,
    ExecutionScope,
    execution_context,
    get_execution_context,
    get_execution_scope,
    thread_context,
)


def test_default_execution_scope_is_available():
    scope = get_execution_scope()

    assert scope.thread_id is None
    assert scope.namespace == DEFAULT_NAMESPACE


def test_execution_context_accepts_scope_and_explicit_overrides():
    base_scope = ExecutionScope(
        thread_id="thread_1",
        namespace="root",
        run_id="run_1",
    )

    with execution_context(scope=base_scope, namespace="agent"):
        scope = get_execution_scope()

    assert scope.thread_id == "thread_1"
    assert scope.namespace == "agent"
    assert scope.run_id == "run_1"
    assert scope.root_run_id == "run_1"


def test_execution_context_exposes_abort_signal():
    abort_signal = AbortSignal()
    scope = ExecutionScope(
        thread_id="thread_1",
        namespace="root",
        run_id="run_1",
        abort_signal=abort_signal,
    )

    with execution_context(scope=scope):
        context = get_execution_context()

    assert context["abort_signal"] is abort_signal
    assert context["scope"].abort_signal is abort_signal
    assert "abort_signal" not in scope.to_dict()


def test_chat_messages_inherit_thread_context():
    with thread_context(thread_id="user_42", namespace="support"):
        chat = ChatMessages()

    assert chat.thread_id == "user_42"
    assert chat.namespace == "support"


def test_chat_messages_state_roundtrip():
    chat = ChatMessages(thread_id="thread_1", namespace="agent:test")
    turn_id = chat.begin_turn(inputs="What is 2+2?", vars={"temperature": 0.2})
    chat.add_user("What is 2+2?")
    chat.add_assistant_response("4", reasoning_content="Simple arithmetic.")
    chat.end_turn(assistant_output="4", response_type="text_generation")

    state = chat._to_state()

    restored = ChatMessages()
    restored._hydrate_state(state)

    assert restored.thread_id == "thread_1"
    assert restored.namespace == "agent:test"
    assert restored.turns[0]["turn_id"] == turn_id
    assert restored.to_chatml()[0]["content"] == "What is 2+2?"
    assert restored.to_chatml()[-1]["content"] == "4"


def test_disabled_chat_message_is_persisted_but_not_rendered():
    chat = ChatMessages(thread_id="thread_1", namespace="agent:test")
    chat.add_user("old context")
    chat.add_user("current context")
    chat.set_item_disabled(0, reason="compacted")

    state = chat._to_state()
    restored = ChatMessages()
    restored._hydrate_state(state)

    assert restored.to_items()[0]["disabled"] is True
    assert restored.to_items()[0]["metadata"]["disabled_reason"] == "compacted"
    assert [item["content"] for item in restored.to_chatml()] == ["current context"]


def test_chat_messages_close_interrupted_tool_calls():
    chat = ChatMessages(thread_id="thread_1", namespace="agent:test")
    chat.append(
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
        }
    )

    closed = chat.close_interrupted_tool_calls(reason="user pressed interrupt")
    closed_again = chat.close_interrupted_tool_calls(reason="user pressed interrupt")

    items = chat.to_items()
    assert closed == 1
    assert closed_again == 0
    assert items[-1]["type"] == "function_call_output"
    assert items[-1]["call_id"] == "call_1"
    assert items[-1]["status"] == "interrupted"
    assert items[-1]["output"]["status"] == "interrupted"
    assert items[-1]["output"]["details"] == "user pressed interrupt"
