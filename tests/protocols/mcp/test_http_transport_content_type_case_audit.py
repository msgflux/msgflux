"""HTTP content type matching is case-insensitive for SSE responses."""

import asyncio
import json

import pytest

from msgflux.protocols.mcp.transports import HTTPTransport


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "with_notification"),
    [
        ("Text/Event-Stream", False),
        ("TEXT/EVENT-STREAM", True),
        ("text/event-stream; charset=utf-8", False),
    ],
)
async def test_http_transport_parses_sse_with_case_insensitive_media_type(
    content_type, with_notification
):
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {"progress": 1},
    }
    response = {"jsonrpc": "2.0", "id": "1", "result": {"ok": True}}
    events = []
    if with_notification:
        events.append(f"data: {json.dumps(notification)}\n\n")
    events.append(f"data: {json.dumps(response)}\n\n")
    response_body = "".join(events).encode()
    request_received = asyncio.Event()

    async def handle_request(reader, writer):
        try:
            headers = await reader.readuntil(b"\r\n\r\n")
            content_length = next(
                int(line.split(b":", 1)[1])
                for line in headers.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            await reader.readexactly(content_length)
            request_received.set()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                + f"Content-Type: {content_type}\r\n".encode()
                + f"Content-Length: {len(response_body)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + response_body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    transport = HTTPTransport(f"http://127.0.0.1:{port}/mcp", timeout=2)
    await transport.connect()
    try:
        callback = events.append if with_notification else None
        result = await transport.send_request(
            "tools/call",
            on_notification=callback,
        )
        await asyncio.wait_for(request_received.wait(), timeout=1)

        assert result == response
        if with_notification:
            assert events[-1] == notification
    finally:
        await transport.disconnect()
        server.close()
        await server.wait_closed()
