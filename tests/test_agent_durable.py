from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.exceptions import AbortRequestedError, TaskInterruptRequestedError
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.hooks import BeforeResume, Guard, Hook
from msgflux.nn.modules.agent import Agent
from msgflux.runtime import AbortSignal
from msgflux.runtime.agent_inbox import AgentInbox, InMemoryAgentInboxStore
from msgflux.runtime.context import ExecutionScope, execution_context


def _mock_model():
    model = Mock()
    model.model_type = "chat_completion"
    return model


def _text_response(text="Hello!"):
    response = Mock(spec=ModelResponse)
    response.response_type = "text_generation"
    response.data = text
    response.reasoning = None
    response.metadata = {"model": "test"}
    response.consume.return_value = text
    return response


def _tool_call_response():
    tool_calls = ToolCallAggregator()
    tool_calls.process(0, "call_lookup", "lookup", '{"query":"status"}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(tool_calls)
    response.reasoning = None
    response.metadata = {"model": "test"}
    return response


def _make_agent(checkpoint_store=None, **kwargs):
    return Agent(
        name="test_agent",
        model=_mock_model(),
        checkpoint_store=checkpoint_store,
        **kwargs,
    )


def _without_item_ids(items):
    return [
        {key: value for key, value in item.items() if key != "item_id"}
        for item in items
    ]


def test_agent_accepts_checkpoint_store_with_stream():
    store = InMemoryCheckpointStore()

    agent = _make_agent(checkpoint_store=store, config={"stream": True})

    assert agent.checkpoint_store is store


def test_agent_saves_completed_checkpoint():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)
    agent.generator.forward = Mock(return_value=_text_response("42"))

    scope = ExecutionScope(
        thread_id="user_42", namespace="test_agent", run_id="run_math"
    )
    result = agent("What is 6*7?", scope=scope)

    assert result == "42"
    state = store.load_state("test_agent", "user_42", "run_math")
    assert state is not None
    assert state["status"] == "completed"
    assert state["messages"]["thread_id"] == "user_42"
    assert "turns" not in state["messages"]
    assert "vars" not in state
    assert _without_item_ids(
        [
            item
            for item in state["messages"]["items"]
            if item.get("role") == "assistant" and item.get("content") == "42"
        ]
    ) == [{"type": "message", "role": "assistant", "content": "42"}]


def test_guard_response_completes_turn_and_checkpoint():
    store = InMemoryCheckpointStore()
    history = ChatMessages()
    agent = _make_agent(
        checkpoint_store=store,
        hooks=[
            Guard(
                validator=lambda _data: {"safe": False},
                on="pre",
                message="Blocked.",
            )
        ],
    )
    agent.generator.forward = Mock(side_effect=AssertionError("model called"))
    scope = ExecutionScope(
        thread_id="guard_thread",
        namespace="test_agent",
        run_id="guard_run",
    )

    assert agent("unsafe", messages=history, scope=scope) == "Blocked."

    state = store.load_state("test_agent", "guard_thread", "guard_run")
    assert state["status"] == "completed"
    assert history.get_active_turn() is None
    assert history.turns[-1]["status"] == "completed"
    assert history.to_chatml()[-1] == {"role": "assistant", "content": "Blocked."}


@pytest.mark.asyncio
async def test_async_guard_response_completes_turn_and_checkpoint():
    store = InMemoryCheckpointStore()
    history = ChatMessages()
    agent = _make_agent(
        checkpoint_store=store,
        hooks=[
            Guard(
                validator=lambda _data: {"safe": False},
                on="pre",
                message="Blocked.",
            )
        ],
    )
    agent.generator.aforward = AsyncMock(side_effect=AssertionError("model called"))
    scope = ExecutionScope(
        thread_id="guard_thread_async",
        namespace="test_agent",
        run_id="guard_run_async",
    )

    assert await agent.acall("unsafe", messages=history, scope=scope) == "Blocked."

    state = store.load_state(
        "test_agent",
        "guard_thread_async",
        "guard_run_async",
    )
    assert state["status"] == "completed"
    assert history.get_active_turn() is None
    assert history.turns[-1]["status"] == "completed"


def test_agent_persists_only_minimal_usage_on_each_model_response():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    def lookup(query: str) -> str:
        return f"status for {query}"

    agent.tool_library.add(lookup)
    tool_response = _tool_call_response()
    tool_response.metadata = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "cache_hit_percentage": 80.0,
            "input_tokens_details": {"cached_tokens": 80},
            "raw": {"provider_field": "discarded"},
        }
    }
    final_response = _text_response("All systems operational.")
    final_response.metadata = {
        "usage": {
            "input_tokens": 140,
            "output_tokens": 10,
            "total_tokens": 150,
            "cache_hit_percentage": 50.0,
            "input_tokens_details": {"cached_tokens": 70},
        }
    }
    agent.generator.forward = Mock(side_effect=[tool_response, final_response])

    agent(
        "Check status",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_usage",
        ),
    )

    state = store.load_state("test_agent", "user_42", "run_usage")
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    model_outputs = [
        item
        for item in restored
        if item.get("type") in {"function_call", "message"}
        and isinstance(item.get("metadata", {}).get("usage"), dict)
    ]

    assert [item["metadata"]["usage"] for item in model_outputs] == [
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "cached_input_tokens": 80,
        },
        {
            "input_tokens": 140,
            "output_tokens": 10,
            "cached_input_tokens": 70,
        },
    ]
    assert all("metadata" not in message for message in restored.to_chatml())


def test_agent_persists_model_audit_metadata_from_lm_response():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)
    response = _text_response("Audited")
    response.metadata = {
        "model": {
            "provider": "openai",
            "model_id": "gpt-5.6-luna",
            "api_mode": "responses",
            "reasoning_effort": "medium",
            "untrusted_provider_field": "discarded",
        }
    }
    agent.generator.forward = Mock(return_value=response)

    agent(
        "Answer",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_audit",
        ),
    )

    state = store.load_state("test_agent", "user_42", "run_audit")
    assistant = next(
        item
        for item in state["messages"]["items"]
        if item.get("role") == "assistant" and item.get("content") == "Audited"
    )
    assert assistant["metadata"]["model"] == {
        "provider": "openai",
        "model_id": "gpt-5.6-luna",
        "api_mode": "responses",
        "reasoning_effort": "medium",
    }


def test_agent_persists_tool_call_usage_before_return_direct():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    def lookup(query: str) -> str:
        """Look up the current status."""
        return f"status for {query}"

    lookup.tool_config = {"return_direct": True}
    agent.tool_library.add(lookup)
    tool_response = _tool_call_response()
    tool_response.metadata = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "total_tokens": 120,
            "input_tokens_details": {"cached_tokens": 80},
        }
    }
    agent.generator.forward = Mock(return_value=tool_response)

    agent(
        "Check status",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_return_direct_usage",
        ),
    )

    state = store.load_state(
        "test_agent",
        "user_42",
        "run_return_direct_usage",
    )
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    calls = [item for item in restored if item.get("type") == "function_call"]

    assert len(calls) == 1
    assert calls[0]["call_id"] == "call_lookup"
    assert calls[0]["metadata"]["usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cached_input_tokens": 80,
    }


def test_agent_accepts_execution_scope_for_checkpoint_identity():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)
    agent.generator.forward = Mock(return_value=_text_response("scoped"))

    scope = ExecutionScope(thread_id="user_42", namespace="outer", run_id="run_scope")
    result = agent("Use the scoped identity", scope=scope)

    assert result == "scoped"
    state = store.load_state("test_agent", "user_42", "run_scope")
    assert state is not None
    assert state["status"] == "completed"
    assert state["messages"]["namespace"] == "test_agent"


def test_agent_resumes_exact_run_id():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(turn_id="run_resume")
    chat.add_user("What is 2+2?")
    store.save_state(
        "test_agent",
        "user_42",
        "run_resume",
        {
            "status": "running",
            "messages": chat._to_state(),
        },
    )

    agent.generator.forward = Mock(return_value=_text_response("4"))
    result = agent(
        "this input should be ignored on resume",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_resume",
        ),
    )

    assert result == "4"
    state = store.load_state("test_agent", "user_42", "run_resume")
    assert state is not None
    assert state["status"] == "completed"

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    assert restored.to_chatml()[0]["content"] == "What is 2+2?"
    assert restored.to_chatml()[-1]["content"] == "4"


def test_before_resume_transforms_restored_state():
    store = InMemoryCheckpointStore()
    observed = []

    def observe(event):
        observed.append(event)
        transformed = ChatMessages(
            thread_id=event.messages.thread_id,
            namespace=event.messages.namespace,
        )
        transformed.begin_turn(turn_id=event.scope.run_id)
        transformed.add_user("Transformed input")
        return BeforeResume(
            scope=event.scope,
            messages=transformed,
            model_preference="strong",
        )

    agent = _make_agent(
        checkpoint_store=store,
        hooks=[Hook(event="before_resume", handler=observe)],
    )
    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(turn_id="run_resume_hook")
    chat.add_user("Preserve this input")
    store.save_state(
        "test_agent",
        "user_42",
        "run_resume_hook",
        {"status": "running", "messages": chat._to_state()},
    )
    agent.generator.forward = Mock(return_value=_text_response("preserved"))

    agent(
        "ignored",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_resume_hook",
        ),
    )

    assert len(observed) == 1
    assert isinstance(observed[0], BeforeResume)
    sent_messages = agent.generator.forward.call_args.kwargs["messages"]
    assert sent_messages.to_chatml()[0]["content"] == "Transformed input"
    assert agent.generator.forward.call_args.kwargs["model_preference"] == "strong"


def test_before_resume_rejects_checkpoint_identity_changes():
    store = InMemoryCheckpointStore()

    def change_thread(event):
        return BeforeResume(
            scope=event.scope.with_overrides(thread_id="another_thread"),
            messages=event.messages,
            model_preference=event.model_preference,
        )

    agent = _make_agent(
        checkpoint_store=store,
        hooks=[Hook(event="before_resume", handler=change_thread)],
    )
    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(turn_id="run_resume_identity")
    chat.add_user("Preserve checkpoint identity")
    store.save_state(
        "test_agent",
        "user_42",
        "run_resume_identity",
        {"status": "running", "messages": chat._to_state()},
    )

    with pytest.raises(ValueError, match=r"cannot change.*thread_id"):
        agent(
            "ignored",
            scope=ExecutionScope(
                thread_id="user_42",
                namespace="test_agent",
                run_id="run_resume_identity",
            ),
        )


@pytest.mark.asyncio
async def test_async_before_resume_applies_async_transformation():
    store = InMemoryCheckpointStore()

    async def transform(event):
        transformed = ChatMessages(
            thread_id=event.messages.thread_id,
            namespace=event.messages.namespace,
        )
        transformed.begin_turn(turn_id=event.scope.run_id)
        transformed.add_user("Async transformed input")
        return BeforeResume(
            scope=event.scope,
            messages=transformed,
            model_preference=event.model_preference,
        )

    agent = _make_agent(
        checkpoint_store=store,
        hooks=[Hook(event="before_resume", handler=transform)],
    )
    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(turn_id="run_async_resume_hook")
    chat.add_user("Original input")
    store.save_state(
        "test_agent",
        "user_42",
        "run_async_resume_hook",
        {"status": "running", "messages": chat._to_state()},
    )
    agent.generator.aforward = AsyncMock(return_value=_text_response("preserved"))

    await agent.acall(
        "ignored",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_async_resume_hook",
        ),
    )

    sent_messages = agent.generator.aforward.call_args.kwargs["messages"]
    assert sent_messages.to_chatml()[0]["content"] == "Async transformed input"


def test_agent_resumes_failed_run_id():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(turn_id="run_failed")
    chat.add_user("Call flaky backend")
    store.save_state(
        "test_agent",
        "user_42",
        "run_failed",
        {
            "status": "failed",
            "messages": chat._to_state(),
        },
    )

    agent.generator.forward = Mock(return_value=_text_response("recovered"))
    result = agent(
        "this retry input should be ignored on resume",
        vars={"attempt": 2},
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_failed",
        ),
    )

    assert result == "recovered"
    state = store.load_state("test_agent", "user_42", "run_failed")
    assert state is not None
    assert state["status"] == "completed"
    assert "vars" not in state

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    assert restored.to_chatml()[0]["content"] == "Call flaky backend"
    assert restored.to_chatml()[-1]["content"] == "recovered"


def test_agent_interrupt_closes_active_tool_calls_in_checkpoint():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    def lookup(query: str) -> str:
        raise TaskInterruptRequestedError("task_1", f"user pressed interrupt: {query}")

    agent.tool_library.add(lookup)
    agent.generator.forward = Mock(return_value=_tool_call_response())

    with pytest.raises(TaskInterruptRequestedError):
        agent(
            "Check status",
            scope=ExecutionScope(
                thread_id="user_42",
                namespace="test_agent",
                run_id="run_interrupt",
            ),
        )

    state = store.load_state("test_agent", "user_42", "run_interrupt")
    assert state is not None
    assert state["status"] == "interrupted"

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    chatml = restored.to_chatml()
    assert any(
        item.get("role") == "assistant"
        and item.get("tool_calls", [{}])[0].get("id") == "call_lookup"
        for item in chatml
    )
    interrupted_outputs = [
        item
        for item in chatml
        if item.get("role") == "tool" and item.get("tool_call_id") == "call_lookup"
    ]
    assert len(interrupted_outputs) == 1
    assert "interrupted" in interrupted_outputs[0]["content"]
    assert restored.turns[-1]["status"] == "interrupted"


def test_agent_persists_tool_reasoning_once_in_canonical_history():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    def lookup(query: str) -> str:
        return f"status for {query}"

    agent.tool_library.add(lookup)
    tool_response = _tool_call_response()
    tool_response.reasoning = "Check the current status."
    tool_response.history_items = [
        {
            "type": "reasoning",
            "role": "assistant",
            "text": "Check the current status.",
            "provider_state": {
                "provider": "openrouter",
                "data": [{"type": "reasoning.text", "text": "opaque order"}],
            },
        }
    ]
    agent.generator.forward = Mock(
        side_effect=[tool_response, _text_response("All systems operational.")]
    )

    result = agent(
        "Check status",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_tool_reasoning",
        ),
    )

    state = store.load_state("test_agent", "user_42", "run_tool_reasoning")
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    reasoning_items = [item for item in restored if item.get("type") == "reasoning"]

    assert result == "All systems operational."
    assert len(reasoning_items) == 1
    assert reasoning_items[0]["text"] == "Check the current status."
    assert all("<think>" not in str(item.get("content", "")) for item in restored)


def test_agent_preserves_responses_function_call_without_duplicate():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    def lookup(query: str) -> str:
        return f"status for {query}"

    agent.tool_library.add(lookup)
    tool_response = _tool_call_response()
    tool_response.history_items = [
        {
            "type": "tool_search_call",
            "provider_state": {
                "provider": "openai",
                "api_mode": "responses",
                "data": {"type": "tool_search_call", "id": "ts_lookup"},
            },
        },
        {
            "type": "tool_search_output",
            "provider_state": {
                "provider": "openai",
                "api_mode": "responses",
                "data": {
                    "type": "tool_search_output",
                    "tool_search_call_id": "ts_lookup",
                },
            },
        },
        {
            "type": "function_call",
            "id": "fc_lookup",
            "status": "completed",
            "call_id": "call_lookup",
            "name": "lookup",
            "arguments": '{"query":"status"}',
        },
    ]
    agent.generator.forward = Mock(
        side_effect=[tool_response, _text_response("All systems operational.")]
    )

    result = agent(
        "Check status",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_responses_tool",
        ),
    )

    state = store.load_state("test_agent", "user_42", "run_responses_tool")
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    calls = [item for item in restored if item.get("type") == "function_call"]
    searches = [
        item
        for item in restored
        if item.get("type") in {"tool_search_call", "tool_search_output"}
    ]
    outputs = [item for item in restored if item.get("type") == "function_call_output"]

    assert result == "All systems operational."
    assert _without_item_ids(searches) == tool_response.history_items[:2]
    assert _without_item_ids(calls) == [tool_response.history_items[2]]
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "call_lookup"


def test_agent_preserves_responses_message_without_synthetic_duplicate():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("Final answer")
    response.history_items = [
        {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": "Final answer"}],
            "provider_state": {
                "provider": "openai",
                "api_mode": "responses",
                "data": {"id": "msg_1", "status": "completed"},
            },
        }
    ]
    agent.generator.forward = Mock(return_value=response)

    agent(
        "Answer once",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_message_once",
        ),
    )

    state = store.load_state("test_agent", "user_42", "run_message_once")
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    assistant_messages = [
        item
        for item in restored
        if item.get("type") == "message" and item.get("role") == "assistant"
    ]
    assert _without_item_ids(assistant_messages) == response.history_items


def test_agent_abort_signal_saves_interrupted_checkpoint():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)
    agent.generator.forward = Mock(side_effect=AbortRequestedError("user pressed esc"))
    abort_signal = AbortSignal()
    abort_signal.abort("user pressed esc")

    with pytest.raises(TaskInterruptRequestedError, match="user pressed esc"):
        agent(
            "Check status",
            scope=ExecutionScope(
                thread_id="user_42",
                namespace="test_agent",
                run_id="run_abort",
                abort_signal=abort_signal,
            ),
        )

    state = store.load_state("test_agent", "user_42", "run_abort")
    assert state is not None
    assert state["status"] == "interrupted"

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    assert restored.turns[-1]["status"] == "interrupted"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_final_stream_after_tools_checkpoints_on_completion(asynchronous):
    def lookup(query: str) -> str:
        return f"Found {query}"

    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store, tools=[lookup], config={"stream": True})
    stream = ModelStreamResponse(mode="async" if asynchronous else "sync")
    stream.set_response_type("text_generation")
    agent.generator.forward = Mock(side_effect=[_tool_call_response(), stream])
    agent.generator.aforward = AsyncMock(side_effect=[_tool_call_response(), stream])
    scope = ExecutionScope(namespace="test_agent", thread_id="t", run_id="r")
    result = (
        await agent.acall("Lookup", scope=scope)
        if asynchronous
        else agent("Lookup", scope=scope)
    )
    assert result is stream
    assert store.load_state("test_agent", "t", "r")["status"] == "streaming"
    stream.add("Final answer after tools")
    stream.finish()
    # Consumption selects async iteration while this test's event loop is active,
    # even when Agent.forward itself used the synchronous execution path.
    assert [chunk async for chunk in stream.consume()] == ["Final answer after tools"]
    state = store.load_state("test_agent", "t", "r")
    assert state["status"] == "completed"
    items = state["messages"]["items"]
    assert (
        sum(
            item.get("role") == "assistant"
            and item.get("content") == "Final answer after tools"
            for item in items
        )
        == 1
    )
    assert any(item.get("type") == "function_call_output" for item in items)


def test_agent_stream_checkpoint_completes_when_stream_finishes():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store, config={"stream": True})
    stream_response = ModelStreamResponse(mode="sync")
    stream_response.set_response_type("text_generation")
    stream_response.reasoning = "stream reasoning"
    agent.generator.forward = Mock(return_value=stream_response)

    result = agent(
        "Stream status",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_stream",
        ),
    )

    assert result is stream_response
    streaming_state = store.load_state("test_agent", "user_42", "run_stream")
    assert streaming_state is not None
    assert streaming_state["status"] == "streaming"

    stream_response.add("hello")
    stream_response.add(" world")
    stream_response.set_metadata(
        {
            "timing": {
                "source": "provider",
                "latency_ms": 1200.0,
                "ttft_ms": 300.0,
            },
            "usage": {
                "input_tokens": 90,
                "output_tokens": 12,
                "total_tokens": 102,
                "cache_hit_percentage": 50.0,
                "input_tokens_details": {"cached_tokens": 45},
            },
        }
    )
    stream_response.finish()

    completed_state = store.load_state("test_agent", "user_42", "run_stream")
    assert completed_state is not None
    assert completed_state["status"] == "completed"

    restored = ChatMessages()
    restored._hydrate_state(completed_state["messages"])
    chatml = restored.to_chatml()
    assert chatml[-1]["content"] == "hello world"
    assert restored.turns[-1]["status"] == "completed"
    assistant_item = next(
        item for item in reversed(list(restored)) if item.get("role") == "assistant"
    )
    assert assistant_item["metadata"]["usage"] == {
        "input_tokens": 90,
        "output_tokens": 12,
        "cached_input_tokens": 45,
    }
    assert "timing" not in assistant_item["metadata"]
    assert "assistant_output" not in restored.turns[-1]
    assert [item["type"] for item in restored if item.get("type") == "reasoning"] == [
        "reasoning"
    ]
    assert (
        sum(
            item.get("role") == "assistant" and item.get("content") == "hello world"
            for item in restored
        )
        == 1
    )


def test_agent_stream_updates_the_caller_chat_messages_when_finished():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store, config={"stream": True})
    stream_response = ModelStreamResponse(mode="sync")
    stream_response.set_response_type("text_generation")
    agent.generator.forward = Mock(return_value=stream_response)
    messages = ChatMessages(thread_id="user_42", namespace="test_agent")

    result = agent(
        "Stream status",
        messages=messages,
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_visible_stream",
        ),
    )
    result.add("visible")
    result.finish()

    assert messages.to_chatml()[-1]["content"] == "visible"
    assert messages.turns[-1]["status"] == "completed"


def test_agent_rejects_reusing_history_owned_by_unfinished_stream():
    agent = _make_agent(config={"stream": True})
    stream_response = ModelStreamResponse(mode="sync")
    stream_response.set_response_type("text_generation")
    agent.generator.forward = Mock(return_value=stream_response)
    messages = ChatMessages(thread_id="user_42", namespace="test_agent")

    result = agent(
        "First request",
        messages=messages,
        scope=ExecutionScope(thread_id="user_42", run_id="run_stream_owner"),
    )

    with pytest.raises(RuntimeError, match="unfinished Agent stream"):
        agent(
            "Second request",
            messages=messages,
            scope=ExecutionScope(thread_id="user_42", run_id="run_concurrent"),
        )

    result.add("first answer")
    result.finish()
    assert messages.to_chatml() == [
        {"role": "user", "content": "First request"},
        {"role": "assistant", "content": "first answer"},
    ]


def test_stream_finalizer_never_overwrites_externally_changed_history():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store, config={"stream": True})
    stream_response = ModelStreamResponse(mode="sync")
    stream_response.set_response_type("text_generation")
    agent.generator.forward = Mock(return_value=stream_response)
    messages = ChatMessages(thread_id="user_42", namespace="test_agent")

    result = agent(
        "Stream request",
        messages=messages,
        scope=ExecutionScope(thread_id="user_42", run_id="run_stream_conflict"),
    )
    messages.add_system("External update")
    result.add("stream answer")

    with pytest.raises(RuntimeError, match="changed while its Agent stream was active"):
        result.finish()

    assert messages.to_chatml()[-1] == {
        "role": "system",
        "content": "External update",
    }
    state = store.load_state("test_agent", "user_42", "run_stream_conflict")
    assert state["status"] == "completed"


def test_agent_stream_finalizer_saves_sqlite_checkpoint_from_worker(tmp_path):
    store = SQLiteCheckpointStore(path=str(tmp_path / "checkpoints.sqlite3"))
    agent = _make_agent(checkpoint_store=store, config={"stream": True})
    stream_response = ModelStreamResponse(mode="sync")
    stream_response.set_response_type("text_generation")
    agent.generator.forward = Mock(return_value=stream_response)

    result = agent(
        "Stream to SQLite",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_sqlite_stream",
        ),
    )

    def finish_from_worker():
        result.add("persisted")
        result.finish()

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(finish_from_worker).result()

    state = store.load_state("test_agent", "user_42", "run_sqlite_stream")
    assert state["status"] == "completed"
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    assert restored.to_chatml()[-1]["content"] == "persisted"
    store.close()


def test_agent_stream_checkpoint_preserves_partial_output_on_failure():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store, config={"stream": True})
    stream_response = ModelStreamResponse(mode="sync")
    stream_response.set_response_type("text_generation")
    agent.generator.forward = Mock(return_value=stream_response)

    result = agent(
        "Stream status",
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_stream_failed",
        ),
    )

    assert result is stream_response
    stream_response.add("partial")
    stream_response.finish(error=RuntimeError("stream disconnected"), status="failed")

    state = store.load_state("test_agent", "user_42", "run_stream_failed")
    assert state is not None
    assert state["status"] == "failed"

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    chatml = restored.to_chatml()
    assert chatml[-1]["content"] == "partial"
    assert restored.turns[-1]["status"] == "failed"
    assert "assistant_output" not in restored.turns[-1]


def test_agent_stream_checkpoint_marks_pre_output_abort_interrupted():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store, config={"stream": True})
    stream_response = ModelStreamResponse(mode="sync")
    stream_response.finish(
        error=AbortRequestedError("user pressed esc"),
        status="interrupted",
    )
    agent.generator.forward = Mock(return_value=stream_response)

    with pytest.raises(TaskInterruptRequestedError, match="user pressed esc"):
        agent(
            "Stream status",
            scope=ExecutionScope(
                thread_id="user_42",
                namespace="test_agent",
                run_id="run_stream_abort",
            ),
        )

    state = store.load_state("test_agent", "user_42", "run_stream_abort")
    assert state is not None
    assert state["status"] == "interrupted"


def test_agent_continues_latest_thread_with_new_run_id():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(turn_id="run_1")
    chat.add_user("Open a support ticket")
    chat.add_assistant("Ticket opened.")
    chat.end_turn()
    store.save_state(
        "test_agent",
        "user_42",
        "run_1",
        {
            "status": "completed",
            "messages": chat._to_state(),
        },
    )

    agent.generator.forward = Mock(return_value=_text_response("Added note."))
    result = agent(
        "Add note: customer called back",
        vars={"customer": "current"},
        scope=ExecutionScope(
            thread_id="user_42",
            namespace="test_agent",
            run_id="run_2",
        ),
    )

    assert result == "Added note."
    state = store.load_state("test_agent", "user_42", "run_2")
    assert state is not None
    assert state["status"] == "completed"
    assert "vars" not in state

    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    chatml = restored.to_chatml()
    assert [item["content"] for item in chatml] == [
        "Open a support ticket",
        "Ticket opened.",
        "Add note: customer called back",
        "Added note.",
    ]


def test_agent_rejects_completed_run_id_retry():
    store = InMemoryCheckpointStore()
    agent = _make_agent(checkpoint_store=store)

    chat = ChatMessages(thread_id="user_42", namespace="test_agent")
    chat.begin_turn(turn_id="run_done")
    chat.add_user("Original")
    chat.add_assistant("Done.")
    chat.end_turn()
    store.save_state(
        "test_agent",
        "user_42",
        "run_done",
        {
            "status": "completed",
            "messages": chat._to_state(),
        },
    )

    agent.generator.forward = Mock(return_value=_text_response("Fresh result."))
    with pytest.raises(ValueError, match="Use a new run_id"):
        agent(
            "Fresh input",
            scope=ExecutionScope(
                thread_id="user_42",
                namespace="test_agent",
                run_id="run_done",
            ),
        )

    agent.generator.forward.assert_not_called()
    state = store.load_state("test_agent", "user_42", "run_done")
    restored = ChatMessages()
    restored._hydrate_state(state["messages"])
    chatml = restored.to_chatml()
    assert [item["content"] for item in chatml] == [
        "Original",
        "Done.",
    ]


def test_checkpoint_failure_releases_inbox_claim():
    class FailingStore(InMemoryCheckpointStore):
        def save_state(self, *args, **kwargs):
            raise RuntimeError("checkpoint unavailable")

    checkpoint = FailingStore()
    inbox = AgentInbox(store=InMemoryAgentInboxStore())
    agent = _make_agent(checkpoint_store=checkpoint, agent_inbox=inbox)
    scope = ExecutionScope(
        namespace="test_agent", thread_id="user_42", run_id="run_claim"
    )
    scoped = agent._get_scoped_agent_inbox(scope)
    scoped.user_message("retry after checkpoint failure")
    messages = ChatMessages(thread_id="user_42", namespace="test_agent")
    messages.begin_turn(turn_id="run_claim")
    with execution_context(
        scope=scope, checkpoint_store=checkpoint, agent_inbox=scoped
    ):
        agent._drain_inbox_into_messages(messages)
        with pytest.raises(RuntimeError, match="checkpoint unavailable"):
            agent._checkpoint_save(messages, {}, status="running")
        retry = scoped.claim()

    assert [item.metadata["content"] for item in retry] == [
        "retry after checkpoint failure"
    ]


def test_checkpoint_receipts_deduplicate_redelivery_after_ack_failure():
    checkpoint = InMemoryCheckpointStore()
    inbox = AgentInbox(store=InMemoryAgentInboxStore())
    agent = _make_agent(checkpoint_store=checkpoint, agent_inbox=inbox)
    scope = ExecutionScope(
        namespace="test_agent", thread_id="user_42", run_id="receipt"
    )
    scoped = agent._get_scoped_agent_inbox(scope)
    scoped.user_message("deliver this once")
    messages = ChatMessages(thread_id="user_42", namespace="test_agent")
    messages.begin_turn(turn_id="receipt")
    with execution_context(
        scope=scope, checkpoint_store=checkpoint, agent_inbox=scoped
    ):
        agent._drain_inbox_into_messages(messages)
        actual_ack = scoped.ack
        scoped.ack = Mock(side_effect=RuntimeError("inbox offline"))
        with pytest.warns(RuntimeWarning, match="acknowledgement failed"):
            agent._checkpoint_save(messages, {}, status="running")
        scoped.ack = actual_ack
        restored = ChatMessages()
        restored._hydrate_state(
            checkpoint.load_state("test_agent", "user_42", "receipt")["messages"]
        )
        agent._drain_inbox_into_messages(restored)
        assert str(restored.to_chatml()).count("deliver this once") == 1
        agent._checkpoint_save(restored, {})
        assert scoped.peek() == []
