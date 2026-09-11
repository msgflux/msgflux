from __future__ import annotations

import asyncio

import httpx2
import msgspec
import pytest

from msgflux.models.http_transport import HTTPTransport
from msgflux.models.model_credentials import (
    ModelCredentialResolver,
    ResolvedModelCredentials,
)


class _Credentials(ModelCredentialResolver):
    def __init__(self):
        self.calls = 0

    def resolve(self, owner):
        self.calls += 1
        return ResolvedModelCredentials(
            headers={"Authorization": f"Bearer token-{self.calls}"}
        )


class _AsyncCredentials(_Credentials):
    def resolve(self, owner):
        raise AssertionError("async requests must resolve credentials asynchronously")

    async def aresolve(self, owner):
        self.calls += 1
        return ResolvedModelCredentials(
            headers={"Authorization": f"Bearer async-token-{self.calls}"}
        )


class _Owner:
    provider = "test"
    model_id = "test-model"

    def __init__(self, credentials=None):
        self.credential_resolver = credentials or _Credentials()
        self.sampling_params = {"base_url": "https://api.example.com/v1"}
        self.abort_checks = 0

    def _raise_if_aborted(self):
        self.abort_checks += 1


def test_http_transport_sends_json_with_request_time_credentials():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json={"ok": True})

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    owner = _Owner()
    transport = HTTPTransport(client=client)

    response = transport.request(
        owner,
        "/embeddings",
        headers={"Content-Type": "application/json"},
        json={"model": "test-model", "input": "hello"},
    )

    assert response.json() == {"ok": True}
    assert captured[0].url == "https://api.example.com/v1/embeddings"
    assert captured[0].headers["authorization"] == "Bearer token-1"
    assert msgspec.json.decode(captured[0].content) == {
        "model": "test-model",
        "input": "hello",
    }
    assert owner.abort_checks >= 1
    transport.close()
    assert client.is_closed is False
    client.close()


def test_http_transport_supports_multipart_without_forcing_json_headers():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json={"text": "hello"})

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    transport = HTTPTransport(client=client)

    transport.request(
        _Owner(),
        "/audio/transcriptions",
        data={"model": "test-model"},
        files={"file": ("sample.wav", b"audio", "audio/wav")},
    )

    assert captured[0].headers["content-type"].startswith("multipart/form-data;")
    assert b"test-model" in captured[0].content
    assert b"audio" in captured[0].content
    client.close()


def test_http_transport_streams_binary_chunks():
    def handler(request):
        return httpx2.Response(200, content=b"abcdef")

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    transport = HTTPTransport(client=client)

    chunks = list(
        transport.stream(
            _Owner(),
            "/audio/speech",
            json={"model": "test-model", "input": "hello"},
            iterate=lambda response: response.iter_bytes(chunk_size=3),
        )
    )

    assert chunks == [b"abc", b"def"]
    client.close()


@pytest.mark.asyncio
async def test_http_transport_supports_async_json_requests():
    captured = []

    async def handler(request):
        captured.append(request)
        return httpx2.Response(200, json={"data": []})

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    credentials = _AsyncCredentials()
    owner = _Owner(credentials)
    transport = HTTPTransport(async_client=client)

    response = await transport.arequest(
        owner,
        "/embeddings",
        json={"model": "test-model", "input": ["hello"]},
    )

    assert response.json() == {"data": []}
    assert captured[0].headers["authorization"] == "Bearer async-token-1"
    assert credentials.calls == 1
    await transport.aclose()
    assert client.is_closed is False
    await client.aclose()


def test_owned_async_client_is_recreated_for_a_new_event_loop():
    transport = HTTPTransport()
    clients = []

    async def capture_client():
        client = transport._get_async_client()
        clients.append(client)
        await client.aclose()

    asyncio.run(capture_client())
    asyncio.run(capture_client())

    assert clients[0] is not clients[1]
    asyncio.run(transport.aclose())
    assert transport._async_client is None
