"""Tests for msgflux.models.response module."""

import asyncio

import pytest

from msgflux.exceptions import AbortRequestedError
from msgflux.models.response import LMStreamEvent, ModelResponse, ModelStreamResponse


@pytest.mark.asyncio
async def test_async_finalizer_runs_when_producer_finishes_without_consumer():
    stream = ModelStreamResponse(mode="async")
    finished = asyncio.Event()

    async def finalize(_state):
        finished.set()

    stream.add_finalizer(finalize)
    await asyncio.to_thread(stream.finish)
    await asyncio.wait_for(finished.wait(), timeout=1)


@pytest.mark.asyncio
async def test_async_finalizer_registered_after_finish_is_scheduled():
    stream = ModelStreamResponse(mode="async")
    stream.finish()
    finished = asyncio.Event()

    async def finalize(_state):
        finished.set()

    stream.add_finalizer(finalize)
    await asyncio.wait_for(finished.wait(), timeout=1)


class TestModelResponse:
    """Test suite for ModelResponse."""

    def test_model_response_initialization(self):
        """Test ModelResponse initialization."""
        response = ModelResponse()
        assert response.data is None
        assert response.reasoning is None
        assert response.reasoning_summary is None
        assert response.has_reasoning_summary is False
        assert response.metadata is None
        assert response.response_type is None

    def test_model_response_add_data(self):
        """Test adding data to ModelResponse."""
        response = ModelResponse()
        test_data = "Test response content"
        response.add(test_data)
        assert response.data == test_data

    def test_model_response_consume(self):
        """Test consuming data from ModelResponse."""
        response = ModelResponse()
        test_data = {"result": "success", "value": 42}
        response.add(test_data)
        consumed = response.consume()
        assert consumed == test_data

    def test_model_response_set_metadata(self):
        """Test setting metadata on ModelResponse."""
        response = ModelResponse()
        metadata = {"tokens": 100, "model": "gpt-4"}
        response.set_metadata(metadata)
        assert response.metadata == metadata

    def test_model_response_with_none_data(self):
        """Test ModelResponse with None data."""
        response = ModelResponse()
        response.add(None)
        assert response.consume() is None

    def test_model_response_keeps_reasoning_summary_separate(self):
        response = ModelResponse()
        response.reasoning = "private reasoning"
        response.reasoning_summary = "safe summary"

        assert response.consume_reasoning() == "private reasoning"
        assert response.consume_reasoning_summary() == "safe summary"
        assert response.has_reasoning_summary is True


class TestModelStreamResponse:
    """Test suite for ModelStreamResponse."""

    def test_model_stream_response_initialization(self):
        """Test ModelStreamResponse initialization."""
        response = ModelStreamResponse()
        assert response.data is None
        assert response.metadata is None
        assert response.response_type is None

    @pytest.mark.asyncio
    async def test_model_stream_response_add_and_consume(self):
        """Test adding data and consuming from ModelStreamResponse."""
        response = ModelStreamResponse()
        chunks = ["Hello", " ", "world", "!"]

        for chunk in chunks:
            response.add(chunk)
        response.finish()

        consumed_chunks = []
        async for chunk in response.consume():
            consumed_chunks.append(chunk)

        assert consumed_chunks == chunks
        assert response.data == "Hello world!"

    @pytest.mark.asyncio
    async def test_model_stream_response_next_chunk_returns_one_chunk_at_a_time(self):
        """next_chunk should expose pull-based single-chunk consumption."""
        response = ModelStreamResponse()
        chunks = ["Hello", " ", "world", "!"]

        for chunk in chunks:
            response.add(chunk)
        response.finish()

        assert await response.next_chunk() == "Hello"
        assert await response.next_chunk() == " "
        assert await response.next_chunk() == "world"
        assert await response.next_chunk() == "!"
        assert await response.next_chunk() is None
        assert response.data == "Hello world!"

    @pytest.mark.asyncio
    async def test_model_stream_response_next_chunk_supports_bytes(self):
        """next_chunk should return one binary chunk per call."""
        response = ModelStreamResponse()
        chunks = [b"\x00\x01", b"\x02", b"\x03\x04"]

        for chunk in chunks:
            response.add(chunk)
        response.finish()

        assert await response.next_chunk() == b"\x00\x01"
        assert await response.next_chunk() == b"\x02"
        assert await response.next_chunk() == b"\x03\x04"
        assert await response.next_chunk() is None
        assert response.data == b"\x00\x01\x02\x03\x04"

    @pytest.mark.asyncio
    async def test_model_stream_response_accumulates_bytes_data(self):
        """Binary streaming should keep the full bytes payload in response.data."""
        response = ModelStreamResponse()
        chunks = [b"\x00\x01", b"\x02", b"\x03\x04"]

        for chunk in chunks:
            response.add(chunk)
        response.finish()

        consumed_chunks = []
        async for chunk in response.consume():
            consumed_chunks.append(chunk)

        assert consumed_chunks == chunks
        assert response.data == b"\x00\x01\x02\x03\x04"

    @pytest.mark.asyncio
    async def test_model_stream_response_empty_stream(self):
        """Test consuming from empty ModelStreamResponse."""
        response = ModelStreamResponse()
        response.finish()

        consumed_chunks = []
        async for chunk in response.consume():
            consumed_chunks.append(chunk)

        assert consumed_chunks == []

    @pytest.mark.asyncio
    async def test_model_stream_response_raises_stored_error_on_consume(self):
        """Stored stream errors should be raised to the consumer."""
        response = ModelStreamResponse()
        response.finish(error=RuntimeError("stream failed"))

        with pytest.raises(RuntimeError, match="stream failed"):
            async for _ in response.consume():
                pass

    @pytest.mark.asyncio
    async def test_model_stream_response_next_chunk_raises_stored_error(self):
        """next_chunk should raise stored stream errors at the sentinel."""
        response = ModelStreamResponse()
        response.finish(error=RuntimeError("stream failed"))

        with pytest.raises(RuntimeError, match="stream failed"):
            await response.next_chunk()

    def test_model_stream_response_rejects_non_text_or_bytes_chunks(self):
        """Streaming content should fail fast on unsupported chunk types."""
        response = ModelStreamResponse()

        with pytest.raises(
            TypeError,
            match="only supports `str` or `bytes` chunks",
        ):
            response.add({"bad": "chunk"})

        assert isinstance(response.error, TypeError)

    @pytest.mark.asyncio
    async def test_model_stream_response_rejects_mixed_chunk_types(self):
        """Streaming content should fail fast if chunk types change mid-stream."""
        response = ModelStreamResponse()
        response.add("hello")

        with pytest.raises(
            TypeError,
            match="received mixed chunk types",
        ):
            response.add(b" world")

        with pytest.raises(
            TypeError,
            match="received mixed chunk types",
        ):
            async for _ in response.consume():
                pass

    def test_model_stream_response_finalizer_runs_once(self):
        stream = ModelStreamResponse(mode="sync")
        final_states = []

        stream.add_finalizer(final_states.append)
        stream.set_response_type("text_generation")
        stream.add("hello")
        stream.finish()
        stream.finish()

        assert len(final_states) == 1
        assert final_states[0].status == "completed"
        assert final_states[0].response_type == "text_generation"
        assert final_states[0].output == "hello"
        assert list(stream._pending_chunks) == ["hello", None]

    def test_model_stream_response_finish_closes_without_public_add(self):
        stream = ModelStreamResponse(mode="sync")
        stream.set_response_type("text_generation")
        stream.add("hello")
        stream.add = lambda data: (_ for _ in ()).throw(AssertionError(data))

        stream.finish()

        assert stream.data == "hello"
        assert list(stream._pending_chunks) == ["hello", None]
        assert list(stream._reasoning_pending_chunks) == [None]

    def test_model_stream_response_can_finish_reasoning_before_content(self):
        stream = ModelStreamResponse(mode="sync")
        stream.add_reasoning("thinking")

        stream.finish_reasoning()
        stream.add("answer")
        stream.finish()

        assert stream.reasoning is None
        assert stream.data == "answer"
        assert list(stream._reasoning_pending_chunks) == ["thinking", None]
        assert list(stream._pending_chunks) == ["answer", None]

    @pytest.mark.asyncio
    async def test_model_stream_response_has_separate_reasoning_summary_channel(self):
        stream = ModelStreamResponse(mode="async")
        assert stream.reasoning_summary_event.is_set() is False
        stream.add_reasoning("private")
        stream.add_reasoning_summary("safe summary")
        await stream.reasoning_summary_event.wait()
        assert stream.reasoning_summary_event.is_set() is True
        stream.reasoning = "private"
        stream.reasoning_summary = "safe summary"
        stream.finish()

        reasoning = [chunk async for chunk in stream.consume_reasoning()]
        summary = [chunk async for chunk in stream.consume_reasoning_summary()]

        assert reasoning == ["private"]
        assert summary == ["safe summary"]
        assert stream.has_reasoning_summary is True
        assert stream.chat_accumulator.snapshot() == [
            {
                "type": "reasoning",
                "role": "assistant",
                "text": "private",
                "summary": "safe summary",
            }
        ]

    @pytest.mark.asyncio
    async def test_model_stream_response_preserves_cross_channel_event_order(self):
        stream = ModelStreamResponse(mode="async")
        stream.add_reasoning("private")
        stream.add_reasoning_summary("safe summary")
        stream.add("visible answer")
        stream.finish()

        events = [event async for event in stream.consume_events()]

        assert events == [
            LMStreamEvent(type="reasoning.delta", data="private"),
            LMStreamEvent(
                type="reasoning_summary.delta",
                data="safe summary",
            ),
            LMStreamEvent(type="output.delta", data="visible answer"),
        ]

    @pytest.mark.asyncio
    async def test_ordered_events_do_not_drain_legacy_stream_channels(self):
        stream = ModelStreamResponse(mode="async")
        stream.add_reasoning("private")
        stream.add_reasoning_summary("safe summary")
        stream.add("visible answer")
        stream.finish()

        assert [event async for event in stream.consume_events()]
        assert [chunk async for chunk in stream.consume_reasoning()] == ["private"]
        assert [chunk async for chunk in stream.consume_reasoning_summary()] == [
            "safe summary"
        ]
        assert [chunk async for chunk in stream.consume()] == ["visible answer"]

    @pytest.mark.asyncio
    async def test_ordered_events_support_independent_replay_consumers(self):
        stream = ModelStreamResponse(mode="async")
        stream.add_reasoning_summary("plan")
        stream.add("answer")
        stream.finish()

        first = [event async for event in stream.consume_events()]
        second = [event async for event in stream.consume_events()]

        assert (
            first
            == second
            == [
                LMStreamEvent(type="reasoning_summary.delta", data="plan"),
                LMStreamEvent(type="output.delta", data="answer"),
            ]
        )

    @pytest.mark.asyncio
    async def test_ordered_events_multicast_to_concurrent_consumers(self):
        stream = ModelStreamResponse(mode="async")

        async def collect():
            return [event async for event in stream.consume_events()]

        first = asyncio.create_task(collect())
        second = asyncio.create_task(collect())
        await asyncio.sleep(0)
        stream.add_reasoning("plan")
        stream.add("answer")
        stream.finish()

        expected = [
            LMStreamEvent(type="reasoning.delta", data="plan"),
            LMStreamEvent(type="output.delta", data="answer"),
        ]
        assert await first == expected
        assert await second == expected

    def test_reasoning_summary_event_completes_when_stream_has_no_summary(self):
        stream = ModelStreamResponse(mode="sync")

        stream.finish()

        assert stream.reasoning_summary_event.is_set() is True
        assert stream.has_reasoning_summary is False

    def test_model_stream_response_rejects_chunks_after_channel_close(self):
        stream = ModelStreamResponse(mode="sync")
        stream.add_reasoning("thinking")
        stream.finish_reasoning()

        with pytest.raises(RuntimeError, match="closed stream"):
            stream.add_reasoning("late thinking")

        stream.add("answer")
        stream.finish()

        with pytest.raises(RuntimeError, match="closed stream"):
            stream.add("late answer")

    def test_model_stream_response_finalizer_added_after_finish_runs_once(self):
        stream = ModelStreamResponse(mode="sync")
        stream.set_response_type("text_generation")
        stream.add("done")
        stream.finish()
        final_states = []

        stream.add_finalizer(final_states.append)

        assert len(final_states) == 1
        assert final_states[0].status == "completed"
        assert final_states[0].output == "done"

    def test_model_stream_response_finish_with_abort_sets_interrupted_state(self):
        stream = ModelStreamResponse(mode="sync")
        final_states = []
        stream.add_finalizer(final_states.append)

        stream.finish(
            error=AbortRequestedError("user pressed esc"),
            status="interrupted",
        )

        assert len(final_states) == 1
        assert final_states[0].status == "interrupted"
        assert isinstance(final_states[0].error, AbortRequestedError)

    def test_model_stream_response_builds_ordered_history_items(self):
        stream = ModelStreamResponse(mode="sync")
        final_states = []
        stream.add_finalizer(final_states.append)

        stream.add_reasoning("first ")
        stream.add_reasoning("step")
        stream.finish_reasoning()
        stream.add("final ")
        stream.add("answer")
        stream.finish()

        assert final_states[0].items == [
            {
                "type": "reasoning",
                "role": "assistant",
                "text": "first step",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "final answer",
            },
        ]

    def test_model_stream_response_preserves_opaque_reasoning_state(self):
        stream = ModelStreamResponse(mode="sync")
        final_states = []
        stream.add_finalizer(final_states.append)
        stream.chat_accumulator.add_reasoning(
            summary="Checked inventory.",
            provider="openai",
            provider_state={"type": "reasoning", "encrypted_content": "opaque"},
        )

        stream.finish()

        assert final_states[0].items[0]["summary"] == "Checked inventory."
        assert final_states[0].items[0]["provider_state"] == {
            "provider": "openai",
            "data": {"type": "reasoning", "encrypted_content": "opaque"},
        }

    def test_delayed_reasoning_state_merges_by_responses_item_id(self):
        stream = ModelStreamResponse(mode="sync")
        final_states = []
        stream.add_finalizer(final_states.append)

        stream.add_reasoning("Checked inventory.", item_id="rs_1")
        stream.add("In stock.")
        stream.chat_accumulator.add_reasoning(
            provider="groq",
            api_mode="responses",
            codec="responses_reasoning_text",
            provider_state={"type": "reasoning", "id": "rs_1"},
            item_id="rs_1",
        )
        stream.finish()

        assert final_states[0].items == [
            {
                "type": "reasoning",
                "role": "assistant",
                "text": "Checked inventory.",
                "provider_state": {
                    "provider": "groq",
                    "api_mode": "responses",
                    "codec": "responses_reasoning_text",
                    "data": {"type": "reasoning", "id": "rs_1"},
                },
            },
            {
                "type": "message",
                "role": "assistant",
                "content": "In stock.",
            },
        ]


@pytest.mark.asyncio
async def test_sync_finalizer_returning_coroutine_runs_without_consumption():
    response = ModelStreamResponse()
    completed = asyncio.Event()

    async def settle(state):
        completed.set()

    response.add_finalizer(lambda state: settle(state))  # noqa: PLW0108
    await asyncio.to_thread(response.finish)
    await asyncio.wait_for(completed.wait(), 1)
    await response._await_pending_finalizers()
    await response._await_pending_finalizers()


@pytest.mark.asyncio
async def test_failed_sync_finalizer_is_visible_to_consumer():
    response = ModelStreamResponse()

    def fail(state):
        raise RuntimeError("settlement failed")

    response.add_finalizer(fail)
    with pytest.raises(RuntimeError, match="settlement failed"):
        response.finish()
    with pytest.raises(RuntimeError, match="settlement failed"):
        async for _ in response.consume_events():
            pass
