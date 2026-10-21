"""Structured final responses remain available to subsequent durable turns."""

from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.generation.reasoning.cot import ChainOfThought
from msgflux.generation.reasoning.react import ReAct
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime import ExecutionScope
from msgflux.utils.msgspec import msgspec_loads


def _response(payload, *, response_type="structured", history_items=()):
    response = ModelResponse()
    response.set_response_type(response_type)
    response.add(deepcopy(payload))
    response.history_items = deepcopy(list(history_items))
    response.metadata = {
        "usage": {"input_tokens": 12, "output_tokens": 4},
    }
    return response


def _agent(response, *, store=None, **kwargs):
    agent = Agent(
        name="structured",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=store,
        **kwargs,
    )
    agent.generator.forward = Mock(return_value=response)
    agent.generator.aforward = AsyncMock(return_value=response)
    return agent


def _assistant_items(messages):
    return [
        item
        for item in messages
        if item.get("type") == "message" and item.get("role") == "assistant"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    ("payload", "options"),
    [
        ({"label": "positive"}, {"signature": "text: str -> label: str"}),
        (
            {"reasoning": "2 plus 2", "final_answer": "4"},
            {"generation_schema": ChainOfThought},
        ),
        (
            {"thought": "done", "final_answer": "Paris"},
            {"generation_schema": ReAct},
        ),
        ({"count": 0, "ok": False, "items": [], "missing": None}, {}),
        ({}, {}),
        ({"answer": None}, {}),
        ({"answer": None, "text": "also declared"}, {}),
    ],
)
async def test_structured_output_is_saved_without_changing_return_value(
    asynchronous, payload, options
):
    store = InMemoryCheckpointStore()
    response = _response(payload)
    agent = _agent(response, store=store, **options)
    history = ChatMessages()
    scope = ExecutionScope(namespace="structured", thread_id="thread", run_id="run")
    try:
        message = {"text": "great"} if "signature" in options else "Answer"
        result = (
            await agent.acall(message, messages=history, scope=scope)
            if asynchronous
            else agent(message, messages=history, scope=scope)
        )
        assert result == payload
        assert response.data == payload
        assistant = _assistant_items(history)
        assert len(assistant) == 1
        assert msgspec_loads(assistant[0]["content"]) == payload
        assert not any(item.get("type") == "reasoning" for item in history)
        assert assistant[0]["metadata"]["usage"]["output_tokens"] == 4
        checkpoint = store.load_state("structured", "thread", "run")
        assert _assistant_items(checkpoint["messages"]["items"]) == assistant
        assert checkpoint["status"] == "completed"
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("payload", [{"answer": "yes"}, {"text": "yes"}])
async def test_existing_answer_text_history_is_unchanged(asynchronous, payload):
    agent = _agent(_response(payload))
    history = ChatMessages()
    try:
        if asynchronous:
            await agent.acall("Answer", messages=history)
        else:
            agent("Answer", messages=history)
        assert [item["content"] for item in _assistant_items(history)] == ["yes"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native_history", [False, True])
async def test_provider_structured_message_is_not_duplicated(
    asynchronous, native_history
):
    payload = {"label": "positive"}
    native_item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": '{"label":"positive"}'}],
        "phase": "final_answer",
        "provider_state": {"provider": "openai", "data": {"id": "msg_1"}},
    }
    native_reasoning = "Native provider reasoning"
    items = (
        [{"type": "reasoning", "role": "assistant", "text": native_reasoning}]
        if native_history
        else []
    )
    response = _response(payload, history_items=[*items, native_item])
    response.reasoning = native_reasoning
    agent = _agent(response)
    history = ChatMessages()
    try:
        if asynchronous:
            await agent.acall("Answer", messages=history)
        else:
            agent("Answer", messages=history)
        assistant = _assistant_items(history)
        assert len(assistant) == 1
        assert assistant[0]["content"] == native_item["content"]
        assert assistant[0]["provider_state"] == native_item["provider_state"]
        reasoning = [item for item in history if item.get("type") == "reasoning"]
        assert len(reasoning) == 1
        assert reasoning[0]["text"] == native_reasoning
        assert history._items.index(reasoning[0]) < history._items.index(assistant[0])
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("native_history", [False, True])
async def test_schema_and_provider_reasoning_are_stored_once(
    asynchronous, native_history
):
    payload = {"reasoning": "Visible schema explanation", "final_answer": "4"}
    native_reasoning = "Native provider reasoning"
    history_items = (
        [{"type": "reasoning", "role": "assistant", "text": native_reasoning}]
        if native_history
        else []
    )
    response = _response(payload, history_items=history_items)
    response.reasoning = native_reasoning
    agent = _agent(response, generation_schema=ChainOfThought)
    history = ChatMessages()
    try:
        if asynchronous:
            await agent.acall("Answer", messages=history)
        else:
            agent("Answer", messages=history)
        assistant = _assistant_items(history)
        assert len(assistant) == 1
        assert msgspec_loads(assistant[0]["content"]) == payload
        assert native_reasoning not in assistant[0]["content"]
        reasoning = [item for item in history if item.get("type") == "reasoning"]
        assert len(reasoning) == 1
        assert reasoning[0]["text"] == native_reasoning
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_structured_output_survives_sqlite_restart_and_next_turn(
    tmp_path, asynchronous
):
    path = tmp_path / "checkpoints.sqlite3"
    store = SQLiteCheckpointStore(path)
    payload = {"label": "positive", "score": 0}
    first = _agent(_response(payload), store=store)
    scope = ExecutionScope(namespace="structured", thread_id="thread", run_id="first")
    try:
        if asynchronous:
            await first.acall("Classify", scope=scope)
        else:
            first("Classify", scope=scope)
    finally:
        await first.aclose()
        store.close()

    reopened = SQLiteCheckpointStore(path)
    seen = []
    second = _agent(_response("done", response_type="text_generation"), store=reopened)

    def followup(**kwargs):
        seen.extend(deepcopy(_assistant_items(kwargs["messages"])))
        return _response("done", response_type="text_generation")

    async def afollowup(**kwargs):
        return followup(**kwargs)

    second.generator.forward = Mock(side_effect=followup)
    second.generator.aforward = AsyncMock(side_effect=afollowup)
    try:
        next_scope = ExecutionScope(
            namespace="structured", thread_id="thread", run_id="second"
        )
        if asynchronous:
            await second.acall("What was the label?", scope=next_scope)
        else:
            second("What was the label?", scope=next_scope)
        assert len(seen) == 1
        assert msgspec.json.decode(seen[0]["content"]) == payload
    finally:
        await second.aclose()
        reopened.close()
