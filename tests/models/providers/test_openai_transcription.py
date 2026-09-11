"""Tests for direct multipart OpenAI transcription."""

from __future__ import annotations

import httpx2
import pytest

from msgflux.models.http_transport import HTTPTransport
from msgflux.models.response import ModelStreamResponse


def _json_transcript() -> dict:
    return {
        "task": "transcribe",
        "language": "english",
        "duration": 1.25,
        "text": "hello world",
        "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.25,
                "text": "hello world",
                "speaker": "agent",
            }
        ],
        "usage": {
            "type": "tokens",
            "input_tokens": 14,
            "input_token_details": {"text_tokens": 2, "audio_tokens": 12},
            "output_tokens": 3,
            "total_tokens": 17,
        },
    }


def test_openai_transcription_sends_text_multipart(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, text="hello world")

    client = httpx2.Client(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.openai import OpenAISpeechToText

    model = OpenAISpeechToText(
        model_id="gpt-4o-mini-transcribe",
        temperature=0.2,
        http_transport=HTTPTransport(client=client),
        retry=False,
    )
    response = model(
        b"audio-data",
        response_format="text",
        prompt="Expected vocabulary",
        language="en",
    )

    body = requests[0].content
    assert requests[0].url == "https://api.openai.com/v1/audio/transcriptions"
    assert requests[0].headers["authorization"] == "Bearer test-key"
    assert requests[0].headers["content-type"].startswith("multipart/form-data;")
    assert b'name="file"; filename="audio.wav"' in body
    assert b"audio-data" in body
    assert b'name="model"' in body and b"gpt-4o-mini-transcribe" in body
    assert b'name="temperature"' in body and b"0.2" in body
    assert b'name="response_format"' in body and b"text" in body
    assert b'name="prompt"' in body and b"Expected vocabulary" in body
    assert b'name="language"' in body and b"en" in body
    assert response.consume() == {"text": "hello world"}
    assert response.metadata.usage is None
    client.close()


def test_openai_transcription_preserves_structured_output_and_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, json=_json_transcript())

    client = httpx2.Client(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.openai import OpenAISpeechToText

    model = OpenAISpeechToText(
        model_id="gpt-4o-transcribe",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )
    response = model(
        b"audio-data",
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
        include=["logprobs"],
        chunking_strategy={"type": "server_vad", "prefix_padding_ms": 300},
    )

    transcript = response.consume()
    assert transcript["text"] == "hello world"
    assert transcript["segments"][0]["speaker"] == "agent"
    assert "usage" not in transcript
    assert response.metadata.usage.input_tokens == 14
    assert response.metadata.usage.input_tokens_details.audio_tokens == 12
    assert response.metadata.usage.input_tokens_details.text_tokens == 2
    assert response.metadata.details.duration == 1.25
    body = requests[0].content
    assert body.count(b'name="timestamp_granularities[]"') == 2
    assert b'name="include[]"' in body and b"logprobs" in body
    assert b'name="chunking_strategy"' in body
    assert b'"prefix_padding_ms":300' in body
    client.close()


@pytest.mark.asyncio
async def test_openai_transcription_async_uses_async_transport(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def handler(request):
        return httpx2.Response(200, json=_json_transcript())

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.openai import OpenAISpeechToText

    model = OpenAISpeechToText(
        model_id="gpt-4o-transcribe",
        http_transport=HTTPTransport(async_client=client),
        retry=False,
    )
    response = await model.acall(b"audio-data", response_format="json")

    assert response.consume()["text"] == "hello world"
    assert response.metadata.usage.total_tokens == 17
    await client.aclose()


def test_openai_transcription_decodes_sse_and_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    payload = (
        b'data: {"type":"transcript.text.delta","delta":"hello"}\n\n'
        b'data: {"type":"transcript.text.delta","delta":" world"}\n\n'
        b'data: {"type":"transcript.text.done","text":"hello world",'
        b'"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}}\n\n'
    )
    client = httpx2.Client(
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=payload,
            )
        )
    )

    from msgflux.models.providers.openai import OpenAISpeechToText

    model = OpenAISpeechToText(
        model_id="gpt-4o-transcribe",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )
    stream_response = ModelStreamResponse(mode="sync")
    model._stream_generate(
        file=("audio.wav", b"audio-data", "audio/x-wav"),
        model=model.model_id,
        response_format="text",
        stream=True,
        stream_response=stream_response,
    )

    assert stream_response.data == "hello world"
    assert stream_response.metadata.usage.total_tokens == 6
    assert stream_response._final_status == "completed"
    client.close()


@pytest.mark.asyncio
async def test_openai_transcription_decodes_async_sse(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    payload = (
        b'data: {"type":"transcript.text.delta","delta":"async"}\n\n'
        b'data: {"type":"transcript.text.done","text":"async"}\n\n'
    )

    async def handler(request):
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=payload,
        )

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.openai import OpenAISpeechToText

    model = OpenAISpeechToText(
        model_id="gpt-4o-transcribe",
        http_transport=HTTPTransport(async_client=client),
        retry=False,
    )
    stream_response = ModelStreamResponse(mode="async")
    await model._astream_generate(
        file=("audio.wav", b"audio-data", "audio/x-wav"),
        model=model.model_id,
        response_format="text",
        stream=True,
        stream_response=stream_response,
    )

    assert stream_response.data == "async"
    assert stream_response._final_status == "completed"
    await client.aclose()


def test_vllm_transcription_preserves_provider_endpoint_and_key(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "vllm-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, text="hello")

    client = httpx2.Client(transport=httpx2.MockTransport(handler))

    from msgflux.models.providers.vllm import VLLMSpeechToText

    model = VLLMSpeechToText(
        model_id="whisper-large-v3",
        http_transport=HTTPTransport(client=client),
        retry=False,
    )
    response = model(b"audio-data")

    assert response.consume() == {"text": "hello"}
    assert requests[0].url == "http://localhost:8000/v1/audio/transcriptions"
    assert requests[0].headers["authorization"] == "Bearer vllm-key"
    client.close()
