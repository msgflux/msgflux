from copy import deepcopy

from msgflux.chat_messages import ChatMessages
from msgflux.models.reasoning import OpenRouterReasoningCodec
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


def test_chat_messages_deepcopy_does_not_copy_runtime_stream_lock():
    messages = ChatMessages([{"role": "user", "content": "hello"}])

    copied = deepcopy(messages)

    assert copied is not messages
    assert copied.to_items() == messages.to_items()


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
    turn_id = chat.begin_turn()
    chat.add_user("What is 2+2?")
    chat.add_assistant_response("4", reasoning_content="Simple arithmetic.")
    chat.end_turn()

    state = chat._to_state()

    restored = ChatMessages()
    restored._hydrate_state(state)

    assert restored.thread_id == "thread_1"
    assert restored.namespace == "agent:test"
    assert restored.turns[0]["turn_id"] == turn_id
    assert set(state) == {"items", "metadata", "thread_id", "namespace"}
    assert [item["event"] for item in state["items"] if item.get("type") == "turn"] == [
        "start",
        "complete",
    ]
    assert restored.to_chatml()[0]["content"] == "What is 2+2?"
    assert restored.to_chatml()[-1]["content"] == "4"


def test_chat_messages_assign_unique_stable_item_ids_without_schema_version():
    chat = ChatMessages(thread_id="thread_1", namespace="agent:test")
    chat.add_user("same")
    chat.add_user("same")

    state = chat._to_state()
    restored = ChatMessages()
    restored._hydrate_state(state)

    item_ids = [item["item_id"] for item in state["items"]]
    assert len(set(item_ids)) == 2
    assert [item["item_id"] for item in restored] == item_ids
    assert "schema_version" not in state


def test_legacy_chat_messages_get_deterministic_item_ids_on_hydration():
    state = {
        "items": [{"type": "message", "role": "user", "content": "legacy"}],
        "thread_id": "thread_1",
        "namespace": "agent:test",
    }
    first = ChatMessages()
    second = ChatMessages()

    first._hydrate_state(state)
    second._hydrate_state(state)

    assert first[0]["item_id"] == second[0]["item_id"]


def test_loaded_tools_survive_copy_and_state_roundtrip():
    chat = ChatMessages(thread_id="thread_1", namespace="agent:test")
    chat.load_tools("agent_tools", ["lookup", "search"])

    copied = chat.copy()
    restored = ChatMessages()
    restored._hydrate_state(chat._to_state())

    assert copied.get_loaded_tools("agent_tools") == {"lookup", "search"}
    assert restored.get_loaded_tools("agent_tools") == {"lookup", "search"}


def test_turns_and_examples_are_derived_from_the_timeline():
    chat = ChatMessages(thread_id="thread_1", namespace="agent:test")
    chat.begin_turn(turn_id="turn_1")
    chat.add_user("Question")
    chat.add_reasoning("Reason")
    chat.add_assistant("Answer")
    chat.end_turn(event="complete")

    example = chat.to_examples()[0]

    assert example.inputs == {
        "trajectory": [{"type": "message", "role": "user", "content": "Question"}]
    }
    assert example.labels["trajectory"][0]["type"] == "reasoning"
    assert example.labels["trajectory"][0]["text"] == "Reason"
    assert example.labels["trajectory"][1] == {
        "type": "message",
        "role": "assistant",
        "content": "Answer",
    }


def test_failed_turn_can_resume_without_parallel_turn_state():
    chat = ChatMessages(thread_id="thread_1", namespace="agent:test")
    chat.begin_turn(turn_id="turn_1")
    chat.add_user("Question")
    chat.end_turn(event="fail", metadata={"error": "timeout"})
    chat.resume_turn("turn_1")
    chat.add_assistant("Answer")
    chat.end_turn()

    state = chat._to_state()

    assert "turns" not in state
    assert chat.turns[0]["status"] == "completed"
    assert [item["event"] for item in state["items"] if item.get("type") == "turn"] == [
        "start",
        "fail",
        "resume",
        "complete",
    ]


def test_reasoning_state_only_roundtrips_to_its_provider():
    chat = ChatMessages()
    details = [{"type": "reasoning.encrypted", "data": "opaque"}]
    chat.add_reasoning(
        "Visible summary",
        provider="openrouter",
        provider_state=details,
    )
    chat.add_assistant("Answer")

    same_provider = chat.to_chatml(
        provider="openrouter",
        reasoning_codec=OpenRouterReasoningCodec(),
    )
    other_provider = chat.to_chatml(provider="openai")

    assert same_provider == [
        {
            "role": "assistant",
            "content": "Answer",
            "reasoning_content": "Visible summary",
            "reasoning_details": details,
        }
    ]
    assert other_provider == [
        {
            "role": "assistant",
            "content": "Answer",
            "reasoning_content": "Visible summary",
        }
    ]


def test_openai_reasoning_state_converts_back_to_responses_item():
    raw_item = {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "opaque",
        "summary": [],
    }
    chat = ChatMessages()
    chat.add_reasoning(provider="openai", provider_state=raw_item)

    assert chat.to_responses_input() == [raw_item]


def test_chatml_multimodal_content_survives_canonical_roundtrip():
    content = [
        {"type": "text", "text": "Describe this image"},
        {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png", "detail": "low"},
        },
        {
            "type": "video_url",
            "video_url": {"url": "https://example.com/video.mp4"},
        },
    ]

    chat = ChatMessages.from_chatml([{"role": "user", "content": content}])

    assert chat.to_chatml() == [{"role": "user", "content": content}]


def test_consecutive_reasoning_items_are_combined_for_chatml():
    chat = ChatMessages()
    chat.add_reasoning("first ")
    chat.add_reasoning("second")
    chat.add_assistant("answer")

    assert chat.to_chatml() == [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "first second",
        }
    ]


def test_function_call_provider_state_only_roundtrips_to_its_provider():
    chat = ChatMessages(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"sku":"1842"}',
                "provider_state": {
                    "provider": "openrouter",
                    "data": {"thought_signature": "opaque"},
                },
            }
        ]
    )

    same_provider_call = chat.to_chatml(provider="openrouter")[0]["tool_calls"][0]
    other_provider_call = chat.to_chatml(provider="openai")[0]["tool_calls"][0]

    assert same_provider_call["thought_signature"] == "opaque"
    assert "thought_signature" not in other_provider_call


def test_function_call_provider_state_does_not_cross_api_modes():
    chat = ChatMessages(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
                "provider_state": {
                    "provider": "openai",
                    "api_mode": "responses",
                    "data": {"id": "fc_1", "status": "completed"},
                },
            }
        ]
    )

    chat_call = chat.to_chatml(provider="openai")[0]["tool_calls"][0]
    responses_call = chat.to_responses_input()[0]

    assert "status" not in chat_call
    assert responses_call["status"] == "completed"


def test_openai_function_call_provider_state_converts_to_responses_item():
    chat = ChatMessages(
        [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{}",
                "provider_state": {
                    "provider": "openai",
                    "data": {"id": "fc_1", "status": "completed"},
                },
            }
        ]
    )

    assert chat.to_responses_input() == [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
            "status": "completed",
        }
    ]


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
    assert chat.to_responses_input()[-1]["status"] == "incomplete"


def _completed_turn(chat: ChatMessages, user: str, assistant: str) -> str:
    chat.begin_turn()
    chat.add_user(user)
    chat.add_assistant(assistant)
    chat.end_turn()
    return chat.latest_completed_turn_boundary()["item_id"]


def test_compaction_materializes_portable_view_and_preserves_canonical_history():
    chat = ChatMessages(thread_id="thread_1", namespace="agent:test")
    boundary = _completed_turn(chat, "old question", "old answer")
    canonical_before = chat.to_items()
    chat.begin_turn()
    chat.add_user("new question")
    chat.add_compaction(
        compacted_through_item_id=boundary,
        views=[
            {
                "format": "messages",
                "items": [
                    {
                        "role": "system",
                        "content": "<conversation_summary>old facts</conversation_summary>",
                    }
                ],
            }
        ],
    )

    projected = chat.to_chatml(provider="anthropic", api_mode="messages")

    assert projected == [
        {
            "role": "system",
            "content": "<conversation_summary>old facts</conversation_summary>",
        },
        {"role": "user", "content": "new question"},
    ]
    assert chat.to_items()[: len(canonical_before)] == canonical_before
    assert chat.to_items()[-1]["type"] == "compaction"


def test_newest_compaction_view_wins_without_stacking_summaries():
    chat = ChatMessages()
    first_boundary = _completed_turn(chat, "first", "one")
    first_operation = chat.add_compaction(
        compacted_through_item_id=first_boundary,
        views=[{"format": "messages", "items": [{"role": "system", "content": "S1"}]}],
    )
    second_boundary = _completed_turn(chat, "second", "two")
    second_operation = chat.add_compaction(
        compacted_through_item_id=second_boundary,
        views=[{"format": "messages", "items": [{"role": "system", "content": "S2"}]}],
    )
    chat.begin_turn()
    chat.add_user("third")

    assert chat.to_chatml() == [
        {"role": "system", "content": "S2"},
        {"role": "user", "content": "third"},
    ]
    assert "parent_compaction_id" not in first_operation
    assert second_operation["parent_compaction_id"] == first_operation["item_id"]

    restored = ChatMessages()
    restored._hydrate_state(chat._to_state())
    assert (
        restored.latest_compaction()["parent_compaction_id"]
        == first_operation["item_id"]
    )


def test_compaction_prefers_exact_provider_view_and_keeps_wire_item_opaque():
    chat = ChatMessages()
    boundary = _completed_turn(chat, "question", "answer")
    chat.add_compaction(
        compacted_through_item_id=boundary,
        views=[
            {
                "format": "messages",
                "items": [{"role": "system", "content": "portable"}],
            },
            {
                "format": "provider",
                "provider": "openai",
                "api_mode": "responses",
                "items": [{"type": "compaction", "encrypted_content": "opaque"}],
            },
        ],
    )
    chat.begin_turn()
    chat.add_user("next")

    assert chat.to_responses_input() == [
        {"type": "compaction", "encrypted_content": "opaque"},
        {"type": "message", "role": "user", "content": "next"},
    ]
    assert chat.to_chatml(provider="anthropic", api_mode="messages")[0] == {
        "role": "system",
        "content": "portable",
    }


def test_provider_only_compaction_rejects_incompatible_projection():
    chat = ChatMessages()
    boundary = _completed_turn(chat, "question", "answer")
    chat.add_compaction(
        compacted_through_item_id=boundary,
        views=[
            {
                "format": "provider",
                "provider": "openai",
                "api_mode": "responses",
                "items": [{"type": "compaction", "encrypted_content": "opaque"}],
            }
        ],
    )

    try:
        chat.to_chatml(provider="openrouter", api_mode="chat_completions")
    except ValueError as error:
        assert "no portable view" in str(error)
    else:
        raise AssertionError("Expected an incompatible compaction error")


def test_compaction_state_roundtrip_and_examples_ignore_operation():
    chat = ChatMessages(thread_id="thread_1", namespace="agent:test")
    boundary = _completed_turn(chat, "question", "answer")
    chat.add_compaction(
        compacted_through_item_id=boundary,
        views=[{"format": "messages", "items": [{"role": "system", "content": "S"}]}],
    )
    restored = ChatMessages()
    restored._hydrate_state(chat._to_state())

    assert restored.to_chatml() == [{"role": "system", "content": "S"}]
    assert restored.to_examples()[0].labels == {
        "trajectory": [{"type": "message", "role": "assistant", "content": "answer"}]
    }
