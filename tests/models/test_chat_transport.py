from __future__ import annotations

from collections.abc import Mapping

import httpx2
import pytest

from msgflux.exceptions import ModelProviderHTTPError
from msgflux.models.chat_api import (
    ChatAPIAdapter,
    ChatCredentialResolver,
    PreparedChatRequest,
    ResolvedChatCredentials,
)
from msgflux.models.chat_transport import HTTPChatTransport
from msgflux.models.openai_compatible import OpenAIChatCompletionsAPI


class _Adapter(ChatAPIAdapter):
    name = "test"
    endpoint = "/responses"


class _RotatingCredentials(ChatCredentialResolver):
    def __init__(self):
        self.calls = 0

    def resolve(self, owner):
        self.calls += 1
        return ResolvedChatCredentials(
            headers={"Authorization": f"Bearer token-{self.calls}"}
        )


class _EndpointCredentials(ChatCredentialResolver):
    def resolve(self, owner):
        return ResolvedChatCredentials(
            base_url="https://subscription.example.com/backend-api/codex",
            headers={"Authorization": "Bearer subscription-token"},
        )


class _AsyncCredentials(ChatCredentialResolver):
    def __init__(self):
        self.calls = 0

    def resolve(self, owner):
        raise AssertionError("async requests must use aresolve")

    async def aresolve(self, owner):
        self.calls += 1
        return ResolvedChatCredentials(
            headers={"Authorization": f"Bearer async-token-{self.calls}"}
        )


class _Owner:
    def __init__(self, credentials=None):
        self.api_adapter = _Adapter()
        self.credential_resolver = credentials or _RotatingCredentials()
        self.sampling_params = {"base_url": "https://api.example.com/v1/"}
        self.abort_checks = 0

    def _raise_if_aborted(self):
        self.abort_checks += 1


def _request(*, stream=False):
    return PreparedChatRequest(
        api="responses",
        endpoint="/responses",
        params={
            "model": "test-model",
            "input": "hello",
            "stream": stream,
            "extra_body": {"custom": True},
            "extra_headers": {
                "X-Test": "request",
                "authorization": "must-be-replaced",
            },
        },
    )


def test_direct_transport_sends_expanded_json_and_request_time_credentials():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(
            200,
            json={"id": "resp_1", "output": [], "usage": None},
        )

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    owner = _Owner()
    transport = HTTPChatTransport(client=client)

    first = transport.create(owner, _request())
    second = transport.create(owner, _request())

    assert first.id == "resp_1"
    assert second.id == "resp_1"
    assert len(captured) == 2
    assert captured[0].url == "https://api.example.com/v1/responses"
    assert captured[0].headers["authorization"] == "Bearer token-1"
    assert captured[1].headers["authorization"] == "Bearer token-2"
    assert captured[0].headers["x-test"] == "request"
    assert captured[0].read() == (
        b'{"model":"test-model","input":"hello","stream":false,"custom":true}'
    )
    transport.close(owner)
    assert client.is_closed is False
    client.close()


def test_direct_transport_retries_retryable_status_before_returning():
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx2.Response(429, headers={"Retry-After": "0"})
        return httpx2.Response(200, json={"output": []})

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    transport = HTTPChatTransport(client=client, max_retries=1)
    credentials = _RotatingCredentials()

    response = transport.create(_Owner(credentials), _request())

    assert response.output == []
    assert calls == 2
    assert credentials.calls == 2
    client.close()


def test_direct_transport_uses_request_time_credential_endpoint():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx2.Response(200, json={"output": []})

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    transport = HTTPChatTransport(client=client)

    transport.create(_Owner(_EndpointCredentials()), _request())

    assert captured[0].url == (
        "https://subscription.example.com/backend-api/codex/responses"
    )
    assert captured[0].headers["authorization"] == "Bearer subscription-token"
    client.close()


def test_direct_transport_surfaces_structured_http_error_without_headers():
    def handler(request):
        return httpx2.Response(
            404,
            headers={"x-request-id": "req_test_123"},
            json={
                "error": {
                    "message": "No endpoints match this data policy",
                    "code": "data_policy",
                    "type": "invalid_request_error",
                    "param": "provider.zdr",
                }
            },
        )

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    transport = HTTPChatTransport(client=client, max_retries=0)

    with pytest.raises(ModelProviderHTTPError) as exc_info:
        transport.create(_Owner(), _request())

    error = exc_info.value
    assert error.status_code == 404
    assert error.description == "No endpoints match this data policy"
    assert error.code == "data_policy"
    assert error.error_type == "invalid_request_error"
    assert error.param == "provider.zdr"
    assert error.request_id == "req_test_123"
    assert str(error) == (
        "Model provider request failed with HTTP 404: "
        "No endpoints match this data policy "
        "(code=data_policy, type=invalid_request_error, "
        "param=provider.zdr, request_id=req_test_123)"
    )
    assert "authorization" not in str(error).lower()
    client.close()


def test_direct_transport_decodes_multiline_sse_events():
    payload = (
        b": keepalive\n"
        b"event: response.output_text.delta\n"
        b'data: {"type":"response.output_text.delta",\n'
        b'data: "delta":"hello"}\n\n'
        b"data: [DONE]\n\n"
    )

    def handler(request):
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=payload,
        )

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    transport = HTTPChatTransport(client=client)

    events = list(transport.create(_Owner(), _request(stream=True)))

    assert len(events) == 1
    assert events[0].type == "response.output_text.delta"
    assert events[0].delta == "hello"
    client.close()


def test_direct_transport_does_not_retry_after_stream_output():
    calls = 0

    class BrokenStream(httpx2.SyncByteStream):
        def __iter__(self):
            yield b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
            raise httpx2.ReadError("stream failed")

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=BrokenStream(),
        )

    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    transport = HTTPChatTransport(client=client, max_retries=2)
    iterator = transport.create(_Owner(), _request(stream=True))

    first = next(iterator)
    assert first.delta == "hello"
    with pytest.raises(httpx2.ReadError, match="stream failed"):
        next(iterator)
    assert calls == 1
    client.close()


@pytest.mark.asyncio
async def test_direct_transport_supports_async_json_and_credentials():
    captured: list[httpx2.Request] = []

    async def handler(request):
        captured.append(request)
        return httpx2.Response(200, json={"output": [], "usage": {"input_tokens": 2}})

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    credentials = _AsyncCredentials()
    owner = _Owner(credentials)
    transport = HTTPChatTransport(async_client=client)

    response = await transport.acreate(owner, _request())

    assert isinstance(response, Mapping)
    assert response.usage.input_tokens == 2
    assert captured[0].headers["authorization"] == "Bearer async-token-1"
    assert credentials.calls == 1
    await transport.aclose(owner)
    assert client.is_closed is False
    await client.aclose()


@pytest.mark.asyncio
async def test_direct_transport_supports_async_sse():
    payload = (
        b'data: {"type":"response.reasoning_summary_text.delta",'
        b'"delta":"checking"}\n\n'
        b"data: [DONE]\n\n"
    )

    async def handler(request):
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=payload,
        )

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    transport = HTTPChatTransport(async_client=client)

    stream = await transport.acreate(_Owner(), _request(stream=True))
    events = [event async for event in stream]

    assert len(events) == 1
    assert events[0].delta == "checking"
    await client.aclose()


def test_chat_completions_stream_surfaces_provider_error_event():
    payload = b'data: {"error":{"message":"request rejected"}}\n\n'

    def handler(request):
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=payload,
        )

    owner = _Owner()
    owner.api_adapter = OpenAIChatCompletionsAPI()
    client = httpx2.Client(transport=httpx2.MockTransport(handler))
    transport = HTTPChatTransport(client=client)
    stream = transport.create(owner, _request(stream=True))

    with pytest.raises(RuntimeError, match="request rejected"):
        next(stream)
    client.close()
