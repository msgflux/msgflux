"""Error diagnostics survive HTTP/SSE without exposing untrusted response bodies."""

import json

import httpx2
import pytest

from msgflux.runtime.service import ServiceRecoveryRequiredError
from msgflux.runtime.service.http import (
    AgentServiceClient,
    AgentServiceHTTPError,
    AgentServiceProtocolError,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code",
    [
        (401, "unauthorized"),
        (404, "not_found"),
        (422, "invalid_request"),
        (500, "internal_error"),
        (503, "future_error"),
    ],
)
async def test_http_errors_retain_public_code_and_status(status, code):
    def handler(request):
        return httpx2.Response(
            status, json={"code": code, "message": "Public reason"}, request=request
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient(
            "http://service", token="private-token", client=http
        )
        with pytest.raises(AgentServiceHTTPError, match="Public reason") as raised:
            await client.agents()
    assert raised.value.code == code
    assert raised.value.status_code == status
    assert not hasattr(raised.value, "response")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"<html>private-host-path private-token</html>",
        b'{"message":"private-token"}',
        b'{"code":"","message":"private-token"}',
        b'{"code":"internal_error","message":""}',
    ],
)
async def test_invalid_http_error_payload_has_neutral_diagnostic(body):
    def handler(request):
        return httpx2.Response(502, content=body, request=request)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient(
            "http://service", token="private-token", client=http
        )
        with pytest.raises(AgentServiceHTTPError) as raised:
            await client.agents()
    assert str(raised.value) == "AgentService request failed"
    assert raised.value.code == "http_error"
    assert raised.value.status_code == 502
    assert raised.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("before_snapshot", [True, False])
@pytest.mark.parametrize(
    "code,exception",
    [
        ("recovery_required", ServiceRecoveryRequiredError),
        ("observer_overflow", AgentServiceHTTPError),
        ("future_error", AgentServiceHTTPError),
    ],
)
async def test_sse_errors_retain_code_and_domain_type(before_snapshot, code, exception):
    prefix = "" if before_snapshot else 'event: snapshot\ndata: {"thread_id":"t"}\n\n'
    body = (
        prefix
        + f"event: error\ndata: {json.dumps({'code': code, 'message': 'Public reason'})}\n\n"
    )
    responses = []

    def handler(request):
        response = httpx2.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
            request=request,
        )
        responses.append(response)
        return response

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient(
            "http://service", token="private-token", client=http
        )
        with pytest.raises(exception, match="Public reason") as raised:
            async with client.watch("t") as watcher:
                assert not before_snapshot
                await anext(watcher)
    remote = (
        raised.value.__cause__
        if exception is ServiceRecoveryRequiredError
        else raised.value
    )
    assert isinstance(remote, AgentServiceHTTPError)
    assert remote.code == code
    assert remote.status_code is None  # The SSE connection itself succeeded.
    assert all(response.is_closed for response in responses)


@pytest.mark.asyncio
@pytest.mark.parametrize("before_snapshot", [True, False])
@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        '{"code":"","message":"private-token"}',
        '{"code":"observer_overflow","message":""}',
    ],
)
async def test_malformed_sse_error_is_protocol_failure(before_snapshot, payload):
    prefix = "" if before_snapshot else 'event: snapshot\ndata: {"thread_id":"t"}\n\n'

    def handler(request):
        return httpx2.Response(
            200,
            content=prefix + f"event: error\ndata: {payload}\n\n",
            headers={"content-type": "text/event-stream"},
            request=request,
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient(
            "http://service", token="private-token", client=http
        )
        with pytest.raises(
            AgentServiceProtocolError, match="Invalid SSE error record"
        ) as raised:
            async with client.watch("t") as watcher:
                await anext(watcher)
    assert raised.value.code == "protocol_error"
    assert raised.value.status_code is None
    assert "private-token" not in str(raised.value)


@pytest.mark.asyncio
async def test_invalid_success_response_is_protocol_failure():
    def handler(request):
        return httpx2.Response(200, json={"agents": 123}, request=request)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient(
            "http://service", token="private-token", client=http
        )
        with pytest.raises(AgentServiceProtocolError) as raised:
            await client.agents()
    assert raised.value.code == "protocol_error"
    assert raised.value.__cause__ is not None
