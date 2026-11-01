"""Native httpx2 SSE framing contract for :class:`AgentServiceClient`."""

import asyncio
import json

import httpx2
import pytest

from msgflux.runtime.service.http import (
    AgentServiceClient,
    AgentServiceProtocolError,
)

SNAPSHOT = {"thread_id": "t", "namespace": "main", "messages": []}
EVENT = {
    "type": "message.delta",
    "timestamp": "2026-10-05T00:00:00Z",
    "data": {"text": "Olá"},
    "run_id": "run",
    "source_path": [],
    "version": 1,
}
ONE_MIB = 1024 * 1024


def _sse_frame(name: str, payload, *, terminator: str = "\n") -> str:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return f"event: {name}{terminator}data: {body}{terminator}{terminator}"


class _ChunkStream(httpx2.AsyncByteStream):
    """Yield fixed byte chunks so framing sees arbitrary fragment boundaries."""

    def __init__(self, chunks, closed=None) -> None:
        self._chunks = chunks
        self._closed_calls = closed if closed is not None else []

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        self._closed_calls.append("stream")


class _BlockingStream(httpx2.AsyncByteStream):
    """Emit one snapshot frame, then block inside the pending read."""

    def __init__(self) -> None:
        self.waiting = asyncio.Event()
        self.closed = asyncio.Event()
        self.closed_calls = 0

    async def __aiter__(self):
        yield _sse_frame("snapshot", SNAPSHOT).encode()
        self.waiting.set()
        await self.closed.wait()

    async def aclose(self) -> None:
        self.closed_calls += 1
        self.closed.set()


class _TruncatedLineStream(httpx2.AsyncByteStream):
    """Emit an oversized *unterminated* line while the stream stays open."""

    def __init__(self) -> None:
        self.chunks_sent = 0

    async def __aiter__(self):
        yield _sse_frame("snapshot", SNAPSHOT).encode()
        self.chunks_sent = 1
        yield b"event: event\ndata: " + b"x" * 400
        self.chunks_sent = 2
        # The pending, newline-free line exceeds the cap while the stream is
        # still open; only a bounded decoder can fail here instead of hanging.
        yield b"x" * 800
        self.chunks_sent = 3
        await asyncio.Event().wait()  # pragma: no cover - never resumed

    async def aclose(self) -> None:
        return None


def _handler(body, *, content_type="text/event-stream", seen=None, stream=None):
    async def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": content_type},
            content=None if stream is not None else body,
            stream=stream,
            request=request,
        )

    return handler


def _client(http, **kwargs) -> AgentServiceClient:
    return AgentServiceClient(
        "https://service.example", token="secret", client=http, **kwargs
    )


@pytest.mark.parametrize(
    "value,error",
    [
        (True, TypeError),
        (False, TypeError),
        ("1024", TypeError),
        (1024.0, TypeError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_max_event_size_rejects_non_positive_and_non_integer_values(value, error):
    with pytest.raises(error, match="max_event_size"):
        AgentServiceClient("https://service", token="secret", max_event_size=value)


@pytest.mark.asyncio
async def test_snapshot_larger_than_one_mib_streams_by_default():
    padding = "x" * (ONE_MIB + 4096)
    snapshot = {**SNAPSHOT, "messages": [{"role": "user", "content": padding}]}
    body = (_sse_frame("snapshot", snapshot) + _sse_frame("event", EVENT)).encode()
    assert len(body) > ONE_MIB
    seen = []
    handler = _handler(body, seen=seen)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = _client(http)
        async with client.watch("t") as watcher:
            assert watcher.snapshot.messages[0]["content"] == padding
            assert (await anext(watcher)).data == EVENT["data"]

    request = seen[0]
    assert request.method == "GET"
    assert request.url.path == "/v1/threads/t/watch"
    assert request.headers["authorization"] == "Bearer secret"
    assert request.headers["accept"] == "text/event-stream"
    # Watch streams wait indefinitely for the next durable event.
    assert request.extensions["timeout"]["read"] is None


@pytest.mark.asyncio
async def test_explicit_limit_rejects_oversized_snapshot_as_protocol_error():
    padding = "x" * 4096
    snapshot = {**SNAPSHOT, "messages": [{"role": "user", "content": padding}]}
    body = (_sse_frame("snapshot", snapshot) + _sse_frame("event", EVENT)).encode()
    responses = []

    async def handler(request):
        response = httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=body,
            request=request,
        )
        responses.append(response)
        return response

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = _client(http, max_event_size=1024)
        with pytest.raises(AgentServiceProtocolError) as raised:
            async with client.watch("t"):
                pytest.fail("oversized snapshot must not stream")

    assert raised.value.code == "protocol_error"
    assert isinstance(raised.value.__cause__, httpx2.SSEError)
    assert all(response.is_closed for response in responses)


@pytest.mark.asyncio
async def test_caller_selected_limit_allows_legitimately_large_snapshot():
    padding = "x" * (ONE_MIB + 4096)
    snapshot = {**SNAPSHOT, "messages": [{"role": "user", "content": padding}]}
    body = (_sse_frame("snapshot", snapshot) + _sse_frame("event", EVENT)).encode()
    handler = _handler(body)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = _client(http, max_event_size=len(body) + 1)
        async with client.watch("t") as watcher:
            assert watcher.snapshot.messages[0]["content"] == padding
            assert (await anext(watcher)).type == "message.delta"


@pytest.mark.asyncio
async def test_fragmented_crlf_multiline_frames_are_framed_natively():
    snapshot = {**SNAPSHOT, "namespace": "Olá"}
    payload = json.dumps(snapshot, ensure_ascii=False).encode()
    # Split the JSON payload at a token boundary: SSE joins ``data`` lines with
    # a newline, which is legal JSON whitespace between tokens only.
    split = payload.index(b'"namespace"')
    event_payload = json.dumps(EVENT).encode()
    event_split = event_payload.index(b'"timestamp"')
    body = (
        b": heartbeat\r\n\r\n"
        b"event: snapshot\r\n"
        b"data: "
        + payload[:split]
        + b"\r"
        + b"\n"
        + b"data: "
        + payload[split:]
        + b"\r\n\r\n"
        + b"event: event\r\n"
        + b"data: "
        + event_payload[:event_split]
        + b"\r"
        + b"\n"
        + b"data: "
        + event_payload[event_split:]
        + b"\r\n\r\n"
    )
    closed = []
    stream = _ChunkStream(
        [body[index : index + 7] for index in range(0, len(body), 7)], closed=closed
    )
    handler = _handler(None, stream=stream)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = _client(http)
        async with client.watch("t") as watcher:
            assert watcher.snapshot.namespace == "Olá"
            observed = [item async for item in watcher]
            assert [item.data for item in observed] == [EVENT["data"]]
        assert closed == ["stream"]


@pytest.mark.asyncio
async def test_unterminated_frame_never_yields_a_complete_event():
    body = (
        _sse_frame("snapshot", SNAPSHOT) + 'event: event\ndata: {"type":"message.delta"'
    ).encode()
    handler = _handler(body)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = _client(http)
        async with client.watch("t") as watcher:
            observed = [item async for item in watcher]
        assert observed == []


@pytest.mark.asyncio
async def test_oversized_unterminated_line_fails_before_stream_eof():
    stream = _TruncatedLineStream()
    handler = _handler(None, stream=stream)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = _client(http, max_event_size=1024)
        async with client.watch("t") as watcher:
            with pytest.raises(AgentServiceProtocolError) as raised:
                # The stream never reaches EOF, so a decoder that only checks
                # the size after a terminated line would hang until timeout.
                await asyncio.wait_for(anext(watcher), 2)
            assert stream.chunks_sent == 2

    assert isinstance(raised.value.__cause__, httpx2.SSEError)
    assert "1024" in str(raised.value)


@pytest.mark.asyncio
async def test_non_sse_content_type_is_native_protocol_error():
    handler = _handler(json.dumps(SNAPSHOT), content_type="application/json")

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = _client(http)
        with pytest.raises(AgentServiceProtocolError) as raised:
            async with client.watch("t"):
                pytest.fail("non-SSE response must not stream")

    assert raised.value.code == "protocol_error"
    assert isinstance(raised.value.__cause__, httpx2.SSEError)


@pytest.mark.asyncio
async def test_caller_error_during_inflight_read_keeps_sentinel_and_closes():
    stream = _BlockingStream()
    handler = _handler(None, stream=stream)
    sentinel = RuntimeError("caller body failed")
    responses = []

    async def capturing(request):
        response = await handler(request)
        responses.append(response)
        return response

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(capturing)) as http:
        client = _client(http)
        reader = None
        with pytest.raises(RuntimeError) as raised:
            async with client.watch("t") as watcher:
                reader = asyncio.create_task(anext(watcher, None))
                await asyncio.wait_for(stream.waiting.wait(), 2)
                assert reader.done() is False
                raise sentinel
        # An in-flight read must not turn the caller error into a
        # "generator already running" teardown failure.
        assert raised.value is sentinel
        assert reader is not None
        assert await asyncio.wait_for(reader, 2) is None
        assert stream.closed.is_set()
        assert all(response.is_closed for response in responses)
        assert not http.is_closed


@pytest.mark.asyncio
async def test_corrupt_content_encoding_is_protocol_error_and_closes_response():
    responses = []
    closed = []

    async def handler(request):
        response = httpx2.Response(
            200,
            headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
            stream=_ChunkStream([b"not a gzip stream"], closed),
            request=request,
        )
        responses.append(response)
        return response

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = _client(http)
        with pytest.raises(AgentServiceProtocolError) as raised:
            async with client.watch("t"):
                pytest.fail("corrupt encoded response must not produce a snapshot")
        assert raised.value.code == "protocol_error"
        assert isinstance(raised.value.__cause__, httpx2.DecodingError)
        assert responses[0].is_closed
        assert closed == ["stream"]
        assert not http.is_closed
