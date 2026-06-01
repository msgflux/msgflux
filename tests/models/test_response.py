"""Tests for msgflux.models.response module."""

import pytest

from msgflux.models.response import ModelResponse, ModelStreamResponse


class TestModelResponse:
    """Test suite for ModelResponse."""

    def test_model_response_initialization(self):
        """Test ModelResponse initialization."""
        response = ModelResponse()
        assert response.data is None
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
        response.add(None)

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
        response.add(None)

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
        response.add(None)

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
        response.add(None)

        consumed_chunks = []
        async for chunk in response.consume():
            consumed_chunks.append(chunk)

        assert consumed_chunks == chunks
        assert response.data == b"\x00\x01\x02\x03\x04"

    @pytest.mark.asyncio
    async def test_model_stream_response_empty_stream(self):
        """Test consuming from empty ModelStreamResponse."""
        response = ModelStreamResponse()
        response.add(None)

        consumed_chunks = []
        async for chunk in response.consume():
            consumed_chunks.append(chunk)

        assert consumed_chunks == []

    @pytest.mark.asyncio
    async def test_model_stream_response_raises_stored_error_on_consume(self):
        """Stored stream errors should be raised to the consumer."""
        response = ModelStreamResponse()
        response.set_error(RuntimeError("stream failed"))
        response.add(None)

        with pytest.raises(RuntimeError, match="stream failed"):
            async for _ in response.consume():
                pass

    @pytest.mark.asyncio
    async def test_model_stream_response_next_chunk_raises_stored_error(self):
        """next_chunk should raise stored stream errors at the sentinel."""
        response = ModelStreamResponse()
        response.set_error(RuntimeError("stream failed"))
        response.add(None)

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
