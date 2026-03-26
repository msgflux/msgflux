"""Integration tests for reasoning as first-class field.

Tests reasoning with Groq gpt-oss models (returns reasoning in raw text).
Covers: sync/async, streaming/non-streaming, consume/consume_reasoning,
        has_reasoning flag, two-event system (first_chunk_event, _response_type_event).

Requires: GROQ_API_KEY in .env
"""

import time

import pytest

from msgflux.models.providers.groq import GroqChatCompletion

MODEL_ID = "openai/gpt-oss-120b"


@pytest.fixture
def model():
    return GroqChatCompletion(
        model_id=MODEL_ID,
        max_tokens=512,
        reasoning_effort="low",
        return_reasoning=True,
    )


# --- Sync non-streaming ---


def test_sync_text_generation_with_reasoning(model):
    """consume() returns str, consume_reasoning() returns reasoning text."""
    response = model("What is 2+2? Answer briefly.")

    content = response.consume()
    reasoning = response.consume_reasoning()

    assert isinstance(content, str)
    assert len(content) > 0
    assert response.response_type == "text_generation"

    assert reasoning is not None
    assert isinstance(reasoning, str)
    assert len(reasoning) > 0

    # Content should NOT contain think tags or reasoning wrapper
    assert "<think>" not in content


def test_sync_has_reasoning_true(model):
    """has_reasoning is True when reasoning is present."""
    response = model("What is 2+2? Answer briefly.")

    assert response.has_reasoning is True
    assert response.reasoning is not None


def test_sync_has_reasoning_false():
    """has_reasoning is False when return_reasoning=False."""
    model = GroqChatCompletion(
        model_id=MODEL_ID,
        max_tokens=256,
        reasoning_effort="low",
        return_reasoning=False,
    )
    response = model("What is 2+2?")

    assert response.has_reasoning is False
    assert response.consume_reasoning() is None


def test_sync_text_generation_without_reasoning():
    """When return_reasoning=False, consume_reasoning() returns None."""
    model = GroqChatCompletion(
        model_id=MODEL_ID,
        max_tokens=256,
        reasoning_effort="low",
        return_reasoning=False,
    )
    response = model("What is 2+2?")

    content = response.consume()
    reasoning = response.consume_reasoning()

    assert isinstance(content, str)
    assert reasoning is None


# --- Async non-streaming ---


@pytest.mark.asyncio
async def test_async_text_generation_with_reasoning(model):
    """Async path: consume() returns str, consume_reasoning() returns reasoning."""
    response = await model.acall("What is 2+2? Answer briefly.")

    content = response.consume()
    reasoning = response.consume_reasoning()

    assert isinstance(content, str)
    assert len(content) > 0
    assert response.response_type == "text_generation"

    assert reasoning is not None
    assert isinstance(reasoning, str)
    assert len(reasoning) > 0


@pytest.mark.asyncio
async def test_async_has_reasoning_true(model):
    """Async: has_reasoning is True when reasoning is present."""
    response = await model.acall("What is 2+2? Answer briefly.")

    assert response.has_reasoning is True
    assert response.reasoning is not None


# --- Sync streaming ---


def test_sync_streaming_with_reasoning(model):
    """Streaming: reasoning chunks accumulate in pending buffer (sync side)."""
    response = model("What is 2+2? Answer briefly.", stream=True)

    # first_chunk_event fires on first token (reasoning or content)
    fired = response.first_chunk_event.wait(timeout=10)
    assert fired, "first_chunk_event should fire on first token"

    # Wait for stream to complete (None sentinel in content queue)
    for _ in range(50):
        if response.metadata is not None:
            break
        time.sleep(0.1)

    assert response.response_type == "text_generation"

    # Accumulated reasoning should be available after stream completes
    assert response.has_reasoning is True
    assert response.reasoning is not None
    assert len(response.reasoning) > 0


def test_sync_streaming_event_order(model):
    """first_chunk_event fires before _response_type_event (reasoning comes first)."""
    response = model("What is 2+2? Answer briefly.", stream=True)

    # first_chunk_event fires early (on reasoning token)
    fired = response.first_chunk_event.wait(timeout=10)
    assert fired

    # _response_type_event fires when content type is determined
    response._response_type_event.wait(timeout=15)
    assert response._response_type_event.is_set()
    assert response.response_type in ("text_generation", "tool_call")

    # Wait for completion
    for _ in range(50):
        if response.metadata is not None:
            break
        time.sleep(0.1)


def test_sync_streaming_without_reasoning():
    """Streaming without reasoning: has_reasoning stays False."""
    model = GroqChatCompletion(
        model_id=MODEL_ID,
        max_tokens=256,
        reasoning_effort="low",
        return_reasoning=False,
    )
    response = model("What is 2+2?", stream=True)

    # Wait for stream to complete
    for _ in range(50):
        if response.metadata is not None:
            break
        time.sleep(0.1)

    assert response.has_reasoning is False
    assert response.reasoning is None


# --- Async streaming ---


@pytest.mark.asyncio
async def test_async_streaming_with_reasoning():
    """Async streaming: both queues stream independently."""
    model = GroqChatCompletion(
        model_id=MODEL_ID,
        max_tokens=512,
        reasoning_effort="low",
        return_reasoning=True,
    )
    response = await model.acall("What is 2+2? Answer briefly.", stream=True)

    # Consume content
    content_chunks = []
    async for chunk in response.consume():
        content_chunks.append(chunk)

    content = "".join(content_chunks)
    assert len(content) > 0

    # Consume reasoning
    reasoning_chunks = []
    async for chunk in response.consume_reasoning():
        reasoning_chunks.append(chunk)

    reasoning = "".join(reasoning_chunks)
    assert len(reasoning) > 0

    # Accumulated reasoning field should match
    assert response.reasoning == reasoning
    assert response.has_reasoning is True


@pytest.mark.asyncio
async def test_async_streaming_consume_reasoning_first():
    """Async streaming: consume_reasoning() can be consumed before consume()."""
    model = GroqChatCompletion(
        model_id=MODEL_ID,
        max_tokens=512,
        reasoning_effort="low",
        return_reasoning=True,
    )
    response = await model.acall("What is 2+2? Answer briefly.", stream=True)

    # Consume reasoning FIRST
    reasoning_chunks = []
    async for chunk in response.consume_reasoning():
        reasoning_chunks.append(chunk)

    reasoning = "".join(reasoning_chunks)
    assert len(reasoning) > 0

    # Then consume content
    content_chunks = []
    async for chunk in response.consume():
        content_chunks.append(chunk)

    content = "".join(content_chunks)
    assert len(content) > 0

    assert response.has_reasoning is True


@pytest.mark.asyncio
async def test_async_streaming_events():
    """Async streaming: first_chunk_event and _response_type_event fire."""
    model = GroqChatCompletion(
        model_id=MODEL_ID,
        max_tokens=512,
        reasoning_effort="low",
        return_reasoning=True,
    )
    response = await model.acall("What is 2+2? Answer briefly.", stream=True)

    # first_chunk_event should be set (stream already started)
    assert response.first_chunk_event.is_set()

    # Drain the stream
    async for _ in response.consume():
        pass
    async for _ in response.consume_reasoning():
        pass

    # After stream completes, _response_type_event must be set
    assert response._response_type_event.is_set()
    assert response.response_type == "text_generation"
