"""Tests for direct HTTP OpenAI-compatible speech generation."""

from __future__ import annotations

import json
from pathlib import Path
import httpx2
import pytest

from msgflux.models.http_transport import HTTPTransport
from msgflux.models.response import ModelStreamResponse


def test_openai_speech_writes_binary_response_and_sends_parameters(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, content=b"generated-audio")

    client = httpx2.Client(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.openai import OpenAITextToSpeech

    model = OpenAITextToSpeech(
        model_id="gpt-4o-mini-tts",
        voice="nova",
        speed=1.25,
        http_transport=HTTPTransport(client=client),
        retry=False,
    )
    response = model(
        "Hello from msgFlux.",
        prompt="Speak clearly.",
        response_format="wav",
    )

    audio_path = Path(response.consume())
    try:
        assert audio_path.suffix == ".wav"
        assert audio_path.read_bytes() == b"generated-audio"
        request = requests[0]
        assert request.url == "https://api.openai.com/v1/audio/speech"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["content-type"] == "application/json"
        assert json.loads(request.content) == {
            "model": "gpt-4o-mini-tts",
            "input": "Hello from msgFlux.",
            "response_format": "wav",
            "instructions": "Speak clearly.",
            "voice": "nova",
            "speed": 1.25,
        }
        assert response.response_type == "audio_generation"
    finally:
        audio_path.unlink(missing_ok=True)
        client.close()


@pytest.mark.asyncio
async def test_openai_speech_async_uses_async_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx2.Response(200, content=b"async-audio")

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.openai import OpenAITextToSpeech

    model = OpenAITextToSpeech(
        model_id="gpt-4o-mini-tts",
        http_transport=HTTPTransport(async_client=client),
        retry=False,
    )
    response = await model.acall("Hello.", response_format="pcm")

    audio_path = Path(response.consume())
    try:
        assert audio_path.suffix == ".pcm"
        assert audio_path.read_bytes() == b"async-audio"
        assert requests[0].url == "https://api.openai.com/v1/audio/speech"
    finally:
        audio_path.unlink(missing_ok=True)
        await client.aclose()


def test_openai_speech_streams_configured_binary_chunks(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, content=b"1234567")
        )
    )

    from msgflux.models.providers.openai import OpenAITextToSpeech

    model = OpenAITextToSpeech(
        model_id="gpt-4o-mini-tts",
        stream_chunk_size=3,
        http_transport=HTTPTransport(client=client),
        retry=False,
    )
    stream_response = ModelStreamResponse(mode="sync")
    model._stream_generate(
        input="Hello.",
        response_format="pcm",
        stream_response=stream_response,
    )

    assert stream_response.data == b"1234567"
    assert list(stream_response._pending_chunks) == [b"123", b"456", b"7", None]
    assert stream_response._final_status == "completed"
    client.close()


def test_openai_speech_removes_partial_file_on_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    created_paths = []

    from msgflux.models.providers import openai

    original_named_temporary_file = openai.tempfile.NamedTemporaryFile

    def record_temp_file(*args, **kwargs):
        temp_file = original_named_temporary_file(*args, **kwargs)
        created_paths.append(Path(temp_file.name))
        return temp_file

    def failing_stream(**kwargs):
        yield b"partial"
        raise RuntimeError("connection failed")

    monkeypatch.setattr(openai.tempfile, "NamedTemporaryFile", record_temp_file)
    model = openai.OpenAITextToSpeech(model_id="gpt-4o-mini-tts", retry=False)
    model._execute_model = failing_stream

    with pytest.raises(RuntimeError, match="connection failed"):
        model("Hello.", response_format="mp3")

    assert len(created_paths) == 1
    assert not created_paths[0].exists()


def test_openai_speech_resolves_credentials_on_request(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(200, content=b"audio")
        )
    )

    from msgflux.models.providers.openai import OpenAITextToSpeech

    model = OpenAITextToSpeech(
        model_id="gpt-4o-mini-tts",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        model("Hello.")
    client.close()


def test_openai_speech_does_not_serialize_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from msgflux.models.providers.openai import OpenAITextToSpeech

    model = OpenAITextToSpeech(
        model_id="gpt-4o-mini-tts",
        http_transport=HTTPTransport(),
        retry=False,
    )

    assert "http_transport" not in model.serialize()["state"]


def test_together_speech_uses_together_endpoint_and_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")
    monkeypatch.setenv("TOGETHER_API_KEY", "together-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, content=b"together-audio")

    client = httpx2.Client(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.together import TogetherTextToSpeech

    model = TogetherTextToSpeech(
        model_id="canopylabs/orpheus-3b-0.1-ft",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )
    response = model("Hello.", response_format="mp3")

    audio_path = Path(response.consume())
    try:
        assert audio_path.read_bytes() == b"together-audio"
        assert requests[0].url == "https://api.together.xyz/v1/audio/speech"
        assert requests[0].headers["authorization"] == "Bearer together-key"
    finally:
        audio_path.unlink(missing_ok=True)
        client.close()
