"""Contract tests for the native AgentService HTTP/SSE client."""

import json
from pathlib import Path

import httpx2
import pytest

from msgflux.runtime.approvals import ApprovalConflictError
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
    AgentServiceHTTPError,
    AgentServiceProtocolError,
)
from msgflux.runtime.service.http.records import HealthRecord


def _json_response(request, payload, status=200):
    return httpx2.Response(status, json=payload, request=request)


@pytest.mark.asyncio
async def test_open_thread_sends_host_workspace_path_and_decodes_binding():
    def handler(request):
        assert json.loads(request.content) == {
            "agent_id": "main",
            "thread_id": "project",
            "cwd": "/host/project",
            "interactive": None,
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
                "interactive": None,
            }
            return _json_response(request, {"thread_id": "t", "agent_id": "assistant"})
        if path.endswith("/prompt"):
            assert json.loads(request.content) == {
                "prompt": "hello",
                "request_id": "req",
                "account_name": None,
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
            assert request.url.params.get("require_waitable") in (None, "true")
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
        assert (await client.receipt("t", "req", require_waitable=True)).revision == 1
        with pytest.raises(TypeError, match="require_waitable must be a bool"):
            await client.receipt("t", "req", require_waitable=1)
        assert await client.interrupt("t", "run")
        assert (await client.resume_checkpoint("t", "run")).revision == 2

    assert all(r.headers["authorization"] == "Bearer secret" for r in seen)
    assert seen[-1].url.path == "/v1/threads/t/runs/run/resume"


@pytest.mark.asyncio
async def test_open_thread_can_omit_agent_id_for_service_default():
    def handler(request):
        assert json.loads(request.content) == {
            "agent_id": None,
            "thread_id": None,
            "cwd": None,
            "interactive": None,
        }
        return _json_response(
            request, {"thread_id": "default-thread", "agent_id": "assistant"}, 201
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient("http://service", token="secret", client=http)
        assert await client.open_thread() == ServiceThread(
            "default-thread", "assistant"
        )


@pytest.mark.asyncio
async def test_health_decodes_legacy_and_default_agent_metadata():
    payloads = iter(
        [
            {"instance_id": "legacy", "version": 1},
            {"instance_id": "configured", "version": 1, "default_agent": "main"},
        ]
    )

    def handler(request):
        return _json_response(request, next(payloads))

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient("http://service", token="secret", client=http)
        legacy = await client.health()
        configured = await client.health()

    assert legacy == HealthRecord("legacy")
    assert legacy.default_agent is None
    assert configured == HealthRecord("configured", default_agent="main")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "exception"),
    [
        ("service_busy", ServiceBusyError),
        ("service_conflict", ServiceConflictError),
        ("recovery_required", ServiceRecoveryRequiredError),
        ("approval_conflict", ApprovalConflictError),
        ("forbidden", PermissionError),
    ],
)
async def test_service_error_codes_map_to_service_exceptions(code, exception):
    async def handler(request):
        return _json_response(request, {"code": code, "message": "denied"}, 409)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient(
            "https://service.example", token="secret", client=http
        )
        with pytest.raises(exception, match="denied") as raised:
            await client.agents()
        remote = raised.value.__cause__
        assert isinstance(remote, AgentServiceHTTPError)
        assert remote.code == code
        assert remote.status_code == 409
        assert str(remote) == str(raised.value)


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


@pytest.mark.parametrize("status", [404, 401, 500])
@pytest.mark.asyncio
async def test_active_run_listing_distinguishes_missing_endpoint_from_other_errors(
    status,
):
    def handler(request):
        assert request.url.path == "/v1/runs/active"
        return _json_response(
            request, {"code": "http_error", "message": "original error"}, status
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient("http://service", token="secret", client=http)
        with pytest.raises(AgentServiceHTTPError) as raised:
            await client.active_runs()
        assert raised.value.status_code == status
        assert raised.value.code == "http_error"
        if status == 404:
            assert "Restart or upgrade" in str(raised.value)
            assert "/v1/runs/active" in str(raised.value)
            assert isinstance(raised.value.__cause__, AgentServiceHTTPError)
        else:
            assert str(raised.value) == "original error"


@pytest.fixture
def closed_event_parsers(monkeypatch):
    from contextlib import aclosing
    from msgflux.runtime.service.http import client as client_module

    closed = []
    parse = client_module._sse_events

    async def tracked_parser(lines):
        try:
            async with aclosing(parse(lines)) as events:
                async for event in events:
                    yield event
        finally:
            closed.append("parser")

    monkeypatch.setattr(client_module, "_sse_events", tracked_parser)
    return closed


@pytest.mark.asyncio
@pytest.mark.parametrize("consume_event", [False, True])
async def test_watch_exit_closes_owned_event_parser(
    closed_event_parsers, consume_event
):
    body = (
        'event: snapshot\ndata: {"thread_id":"t","namespace":"main","messages":[]}\n\n'
        'event: event\ndata: {"type":"message.delta","timestamp":"now","data":{}}\n\n'
    )

    async def handler(request):
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=request,
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient("http://service", token="secret", client=http)
        async with client.watch("t") as watcher:
            if consume_event:
                assert (await anext(watcher)).type == "message.delta"
            assert closed_event_parsers == []
        assert closed_event_parsers == ["parser"]
        await watcher.aclose()
        assert closed_event_parsers == ["parser"]
        assert not http.is_closed
        with pytest.raises(StopAsyncIteration):
            await anext(watcher)


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_snapshot", [False, True])
async def test_watch_protocol_error_closes_owned_parser(
    closed_event_parsers, invalid_snapshot
):
    body = (
        "event: snapshot\ndata: {}\n\n"
        if invalid_snapshot
        else 'event: snapshot\ndata: {"thread_id":"t"}\n\nevent: event\ndata: {}\n\n'
    )

    async def handler(request):
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=request,
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient("http://service", token="secret", client=http)
        with pytest.raises(AgentServiceProtocolError, match="Invalid"):
            async with client.watch("t") as watcher:
                await anext(watcher)
        assert closed_event_parsers == ["parser"]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_reader", [False, True])
async def test_watch_close_with_inflight_read(closed_event_parsers, cancel_reader):
    import asyncio

    reading, closed = asyncio.Event(), asyncio.Event()

    class WatchBody(httpx2.AsyncByteStream):
        async def __aiter__(self):
            yield b'event: snapshot\ndata: {"thread_id":"t"}\n\n'
            reading.set()
            await closed.wait()

        async def aclose(self):
            closed.set()

    async def handler(request):
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=WatchBody(),
            request=request,
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient("http://service", token="secret", client=http)
        async with client.watch("t") as watcher:
            reader = asyncio.create_task(anext(watcher))
            try:
                await asyncio.wait_for(reading.wait(), 2)
                if cancel_reader:
                    reader.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await reader
                else:
                    await watcher.aclose()
                    with pytest.raises(StopAsyncIteration):
                        await asyncio.wait_for(reader, 2)
            finally:
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
        assert closed.is_set()
        assert closed_event_parsers == ["parser"]


@pytest.mark.asyncio
async def test_watch_exit_closes_owned_line_iterator(monkeypatch):
    from contextlib import aclosing

    closed = []
    original_lines = httpx2.Response.aiter_lines

    async def tracked_lines(response):
        try:
            async with aclosing(original_lines(response)) as lines:
                async for line in lines:
                    yield line
        finally:
            closed.append("lines")

    monkeypatch.setattr(httpx2.Response, "aiter_lines", tracked_lines)

    async def handler(request):
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='event: snapshot\ndata: {"thread_id":"t"}\n\n',
            request=request,
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient("http://service", token="secret", client=http)
        async with client.watch("t"):
            assert closed == []
        assert closed == ["lines"]
