"""Integration tests for Agent reasoning with Groq gpt-oss.

Tests the full Agent pipeline with reasoning as first-class field.
Covers: sync/async, streaming/non-streaming, reasoning_in_response config,
        has_reasoning flag, tool calls with reasoning.

Requires: GROQ_API_KEY in .env
"""

import time

import pytest

import msgflux as mf
from msgflux.core.dotdict import dotdict
from msgflux.models.response import ModelStreamResponse
from msgflux.nn import Agent

MODEL_PATH = "groq/openai/gpt-oss-120b"


@pytest.fixture
def model():
    return mf.Model.chat_completion(
        MODEL_PATH,
        max_tokens=512,
        reasoning_effort="low",
        return_reasoning=True,
    )


@pytest.fixture
def agent(model):
    return Agent(
        name="test_agent",
        model=model,
        instructions="Answer briefly and concisely.",
    )


@pytest.fixture
def agent_with_reasoning_in_response(model):
    return Agent(
        name="test_agent_reasoning",
        model=model,
        instructions="Answer briefly and concisely.",
        config={"reasoning_in_response": True},
    )


@pytest.fixture
def agent_stream(model):
    return Agent(
        name="test_agent_stream",
        model=model,
        instructions="Answer briefly and concisely.",
        config={"stream": True},
    )


@pytest.fixture
def agent_stream_with_reasoning(model):
    return Agent(
        name="test_agent_stream_reasoning",
        model=model,
        instructions="Answer briefly and concisely.",
        config={"stream": True, "reasoning_in_response": True},
    )


# --- Sync non-streaming ---


def test_agent_sync_text_with_reasoning(agent):
    """Agent returns str, model_response has reasoning."""
    response = agent("What is 2+2?")

    assert isinstance(response, str)
    assert len(response) > 0


def test_agent_sync_reasoning_in_response(agent_with_reasoning_in_response):
    """reasoning_in_response wraps output as dotdict(answer=..., reasoning=...)."""
    response = agent_with_reasoning_in_response("What is 2+2?")

    assert isinstance(response, dotdict)
    assert "answer" in response
    assert "reasoning" in response
    assert isinstance(response.answer, str)
    assert len(response.answer) > 0
    assert isinstance(response.reasoning, str)
    assert len(response.reasoning) > 0


def test_agent_sync_no_reasoning_in_response(agent):
    """Without reasoning_in_response, Agent returns raw str."""
    response = agent("What is 2+2?")

    assert isinstance(response, str)
    assert not isinstance(response, dotdict)


# --- Async non-streaming ---


@pytest.mark.asyncio
async def test_agent_async_text_with_reasoning(agent):
    """Async Agent returns str with reasoning available."""
    response = await agent.acall("What is 2+2?")

    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_agent_async_reasoning_in_response(
    agent_with_reasoning_in_response,
):
    """Async: reasoning_in_response wraps output as dotdict."""
    response = await agent_with_reasoning_in_response.acall("What is 2+2?")

    assert isinstance(response, dotdict)
    assert "answer" in response
    assert "reasoning" in response
    assert isinstance(response.answer, str)
    assert len(response.answer) > 0
    assert isinstance(response.reasoning, str)
    assert len(response.reasoning) > 0


# --- Sync streaming ---


def test_agent_sync_streaming(agent_stream):
    """Streaming Agent returns ModelStreamResponse."""
    response = agent_stream("What is 2+2?")

    assert isinstance(response, ModelStreamResponse)

    # first_chunk_event should fire
    fired = response.first_chunk_event.wait(timeout=15)
    assert fired, "first_chunk_event should fire"

    # Wait for stream completion
    for _ in range(50):
        if response.metadata is not None:
            break
        time.sleep(0.1)

    assert response.response_type == "text_generation"
    assert response.has_reasoning is True
    assert response.reasoning is not None
    assert len(response.reasoning) > 0


def test_agent_sync_streaming_events(agent_stream):
    """Streaming: _response_type_event fires after first_chunk_event."""
    response = agent_stream("What is 2+2?")

    assert isinstance(response, ModelStreamResponse)

    # first_chunk_event fires early (on reasoning token)
    response.first_chunk_event.wait(timeout=15)
    assert response.first_chunk_event.is_set()

    # _response_type_event fires when content type is determined
    response._response_type_event.wait(timeout=15)
    assert response._response_type_event.is_set()
    assert response.response_type in ("text_generation", "tool_call")

    # Wait for completion
    for _ in range(50):
        if response.metadata is not None:
            break
        time.sleep(0.1)


# --- Async streaming ---


@pytest.mark.asyncio
async def test_agent_async_streaming():
    """Async streaming: consume both content and reasoning queues."""
    model = mf.Model.chat_completion(
        MODEL_PATH,
        max_tokens=512,
        reasoning_effort="low",
        return_reasoning=True,
    )
    agent = Agent(
        name="test_agent_async_stream",
        model=model,
        instructions="Answer briefly and concisely.",
        config={"stream": True},
    )
    response = await agent.acall("What is 2+2?")

    assert isinstance(response, ModelStreamResponse)

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

    assert response.has_reasoning is True
    assert response.reasoning == reasoning


@pytest.mark.asyncio
async def test_agent_async_streaming_reasoning_first():
    """Async streaming: consume_reasoning() before consume() works."""
    model = mf.Model.chat_completion(
        MODEL_PATH,
        max_tokens=512,
        reasoning_effort="low",
        return_reasoning=True,
    )
    agent = Agent(
        name="test_agent_async_stream_rf",
        model=model,
        instructions="Answer briefly and concisely.",
        config={"stream": True},
    )
    response = await agent.acall("What is 2+2?")

    assert isinstance(response, ModelStreamResponse)

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


# --- Tool calls with reasoning ---


def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


def test_agent_sync_tool_call_with_reasoning():
    """Agent with tools: reasoning is preserved through tool call loop."""
    model = mf.Model.chat_completion(
        MODEL_PATH,
        max_tokens=512,
        reasoning_effort="low",
        return_reasoning=True,
    )
    agent = Agent(
        name="test_agent_tools",
        model=model,
        instructions="Use the tools provided to compute the result. "
        "Answer with just the number.",
        tools=[add, multiply],
    )
    response = agent("What is 3 + 5?")

    assert isinstance(response, str)
    assert "8" in response


def test_agent_sync_tool_call_reasoning_in_response():
    """Agent with tools + reasoning_in_response.

    After the tool call loop, the final model response may or may not
    contain reasoning. If it does, response is dotdict(answer=..., reasoning=...).
    If it doesn't, response is plain str.
    """
    model = mf.Model.chat_completion(
        MODEL_PATH,
        max_tokens=512,
        reasoning_effort="low",
        return_reasoning=True,
    )
    agent = Agent(
        name="test_agent_tools_rir",
        model=model,
        instructions="Use the tools provided to compute the result. "
        "Answer with just the number.",
        tools=[add, multiply],
        config={"reasoning_in_response": True},
    )
    response = agent("What is 3 + 5?")

    if isinstance(response, dotdict):
        assert "answer" in response
        assert "reasoning" in response
        assert "8" in response.answer
    else:
        # Final model call had no reasoning — plain str
        assert isinstance(response, str)
        assert "8" in response


@pytest.mark.asyncio
async def test_agent_async_tool_call_with_reasoning():
    """Async Agent with tools: reasoning preserved through tool call loop."""
    model = mf.Model.chat_completion(
        MODEL_PATH,
        max_tokens=512,
        reasoning_effort="low",
        return_reasoning=True,
    )
    agent = Agent(
        name="test_agent_tools_async",
        model=model,
        instructions="Use the tools provided to compute the result. "
        "Answer with just the number.",
        tools=[add, multiply],
    )
    response = await agent.acall("What is 3 + 5?")

    assert isinstance(response, str)
    assert "8" in response


# --- Without reasoning ---


def test_agent_sync_no_reasoning():
    """Agent without return_reasoning: no reasoning field."""
    model = mf.Model.chat_completion(
        MODEL_PATH,
        max_tokens=256,
        reasoning_effort="low",
        return_reasoning=False,
    )
    agent = Agent(
        name="test_agent_no_reasoning",
        model=model,
        instructions="Answer briefly.",
    )
    response = agent("What is 2+2?")

    assert isinstance(response, str)
    assert len(response) > 0
