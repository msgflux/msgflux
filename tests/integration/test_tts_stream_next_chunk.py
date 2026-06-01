"""Live E2E tests for TTS stream next_chunk consumption.

Requires: OPENAI_API_KEY in environment or .env.
"""

import os

import pytest

import msgflux as mf
from msgflux import nn

mf.load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is required for live OpenAI integration tests.",
)


class StreamingSpeaker(nn.Speaker):
    model = mf.Model.text_to_speech("openai/gpt-4o-mini-tts")
    response_format = "pcm"
    config = {"stream": True}


@pytest.mark.asyncio
async def test_openai_tts_stream_next_chunk_then_consume_rest():
    speaker = StreamingSpeaker()
    stream = await speaker.acall("Say one short sentence for a streaming test.")

    first_chunk = await stream.next_chunk()
    assert isinstance(first_chunk, bytes)
    assert first_chunk

    remaining_chunks = []
    async for chunk in stream.consume():
        remaining_chunks.append(chunk)

    assert all(isinstance(chunk, bytes) for chunk in remaining_chunks)
    assert stream.response_type == "audio_generation"
    assert isinstance(stream.data, bytes)
    assert stream.data.startswith(first_chunk)
    assert len(stream.data) >= len(first_chunk)
