"""Contract tests for the native AgentService HTTP/SSE client."""

import json
from pathlib import Path

import httpx2
import pytest

from msgflux.runtime.service import (
    AdmissionReceipt,
    ServiceBusyError,
    ServiceConflictError,
    ServiceRecoveryRequiredError,
    ServiceThread,
    RunSummary,
)
from msgflux.runtime.service.http.client import (
    AgentServiceClient,
    AgentServiceProtocolError,
)


def _json_response(request, payload, status=200):
    return httpx2.Response(status, json=payload, request=request)


@pytest.mark.asyncio
async def test_open_thread_sends_host_workspace_path_and_decodes_binding():
    def handler(request):
        assert json.loads(request.content) == {
            "agent_id": "main",
            "thread_id": "project",
            "cwd": "/host/project",
        }
        return _json_response(
            request,
            {"agent_id": "main", "thread_id": "project", "cwd": "/host/project"},
            201,
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient("http://service", token="secret", client=http)
        thread = await client.open_thread(
            "main", thread_id="project", cwd=Path("/host/project")
        )
        assert thread == ServiceThread("project", "main", "/host/project")


@pytest.mark.asyncio
async def test_http_methods_encode_requests_and_decode_records():
    seen = []

    def handler(request):
        seen.append(request)
        path = request.url.path
        if path == "/v1/agents":
            return _json_response(request, {"agents": ["assistant"]})
        if path.endswith("/runs"):
            return _json_response(
                request,
                {"runs": [{"run_id": "r1", "status": "completed", "updated_at": 12.0}]},
            )
        if path == "/v1/threads" and request.method == "GET":
            return _json_response(
                request, {"threads": [{"thread_id": "t", "agent_id": "assistant"}]}
            )
        if path == "/v1/threads" and request.method == "POST":
            assert json.loads(request.content) == {
                "agent_id": "assistant",
                "thread_id": "t",
                "cwd": None,
            }
            return _json_response(request, {"thread_id": "t", "agent_id": "assistant"})
        if path.endswith("/prompt"):
            assert json.loads(request.content) == {
                "prompt": "hello",
                "request_id": "req",
            }
            return _json_response(
                request,
                {
                    "thread_id": "t",
                    "request_id": "req",
                    "run_id": "run",
                    "status": "accepted",
                    "error": None,
                    "version": 1,
                    "revision": 0,
                },
            )
        if path.endswith("/requests/req"):
            return _json_response(
                request,
                {
                    "thread_id": "t",
                    "request_id": "req",
                    "run_id": "run",
                    "status": "running",
                    "error": None,
                    "version": 1,
                    "revision": 1,
                },
            )
        if path.endswith("/interrupt"):
            return _json_response(request, {"interrupted": True})
        if path.endswith("/resume"):
            assert request.content == b"" or json.loads(request.content) == {}
            return _json_response(
                request,
                {
                    "thread_id": "t",
                    "request_id": "req",
                    "run_id": "run",
                    "status": "running",
                    "error": None,
                    "version": 1,
                    "revision": 2,
                },
            )
        raise AssertionError(f"unexpected request: {request.method} {path}")

    transport = httpx2.MockTransport(handler)
    async with httpx2.AsyncClient(transport=transport) as http:
        client = AgentServiceClient(
            "https://service.example/", token="secret", client=http
        )
        assert await client.agents() == ("assistant",)
        runs = await client.runs("t")
        assert runs == (RunSummary("r1", "completed", 12.0),)
        assert await client.threads() == (ServiceThread("t", "assistant"),)
        assert await client.open_thread("assistant", thread_id="t") == ServiceThread(
            "t", "assistant"
        )
        receipt = await client.prompt("t", "hello", request_id="req")
        assert isinstance(receipt, AdmissionReceipt)
        assert receipt.status == "accepted"
        assert (await client.receipt("t", "req")).revision == 1
        assert await client.interrupt("t", "run")
        assert (await client.resume_checkpoint("t", "run")).revision == 2

    assert all(r.headers["authorization"] == "Bearer secret" for r in seen)
    assert seen[-1].url.path == "/v1/threads/t/runs/run/resume"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "exception"),
    [
        ("service_busy", ServiceBusyError),
        ("service_conflict", ServiceConflictError),
        ("recovery_required", ServiceRecoveryRequiredError),
    ],
)
async def test_service_error_codes_map_to_service_exceptions(code, exception):
    async def handler(request):
        return _json_response(request, {"code": code, "message": "denied"}, 409)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient(
            "https://service.example", token="secret", client=http
        )
        with pytest.raises(exception, match="denied"):
            await client.agents()


@pytest.mark.asyncio
async def test_streaming_error_body_is_read_before_mapping():
    closed = []

    class ErrorBody(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"code":"service_busy",'
            yield b'"message":"try later"}'

        async def aclose(self):
            closed.append(True)

    async def handler(request):
        return httpx2.Response(409, stream=ErrorBody(), request=request)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient(
            "https://service.example", token="secret", client=http
        )
        with pytest.raises(ServiceBusyError, match="try later"):
            await client.agents()
    assert closed == [True]


@pytest.mark.asyncio
async def test_interrupt_returns_service_boolean():
    async def handler(request):
        return _json_response(request, {"interrupted": False})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient(
            "https://service.example", token="secret", client=http
        )
        assert await client.interrupt("thread", "run") is False


@pytest.mark.parametrize("token", ["", None, 1])
def test_client_rejects_missing_or_non_string_token(token):
    with pytest.raises(ValueError, match="token must be a non-empty string"):
        AgentServiceClient("https://service.example", token=token)


@pytest.mark.asyncio
async def test_watch_requires_snapshot_first_and_decodes_multiline_events():
    snapshot = {
        "thread_id": "thread/é",
        "namespace": "agent",
        "messages": [],
        "version": 1,
    }
    event = {
        "type": "message.delta",
        "timestamp": "2026-10-05T00:00:00Z",
        "data": {"text": "Olá"},
        "run_id": "run",
        "source_path": [],
        "version": 1,
    }
    body = (
        ": heartbeat\r\n\r\n"
        "event: snapshot\r\n"
        f"data: {json.dumps(snapshot, ensure_ascii=False)}\r\n\r\n"
        "event: event\r\n"
        'data: {"type":"message.delta",\r\n'
        'data: "timestamp":"2026-10-05T00:00:00Z","data":{"text":"Olá"},\r\n'
        'data: "run_id":"run","source_path":[],"version":1}\r\n\r\n'
    ).encode()

    async def handler(request):
        assert request.headers["accept"] == "text/event-stream"
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=request,
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient(
            "https://service.example", token="secret", client=http
        )
        async with client.watch("thread/é") as watcher:
            assert watcher.snapshot.thread_id == "thread/é"
            observed = [item async for item in watcher]
            assert len(observed) == 1
            assert observed[0].data == event["data"]


@pytest.mark.asyncio
async def test_watch_rejects_missing_initial_snapshot():
    async def handler(request):
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"event: event\ndata: {}\n\n",
            request=request,
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient(
            "https://service.example", token="secret", client=http
        )
        with pytest.raises(AgentServiceProtocolError, match="begin with a snapshot"):
            async with client.watch("t"):
                pass
