"""Pending MCP stdio requests settle and release state on failure paths."""

import asyncio
import gc
import sys

import pytest

from msgflux.protocols.mcp.exceptions import MCPConnectionError
from msgflux.protocols.mcp.transports import StdioTransport


async def _wait_for_pending(transport):
    for _ in range(200):
        if transport._pending_requests:
            return next(iter(transport._pending_requests))
        await asyncio.sleep(0.01)
    raise AssertionError("MCP request was not registered")


async def _release_child(transport):
    process = transport._process
    assert process is not None
    process.stdin.write(b"\n")
    await asyncio.wait_for(process.stdin.drain(), timeout=2)


def _transport(tmp_path, source, *, timeout=5.0):
    server = tmp_path / "stdio_server.py"
    server.write_text(source)
    return StdioTransport(sys.executable, ["-u", str(server)], timeout=timeout)


def _close_stdout_source(prefix=""):
    """Close stdout and loop-created aliases so the parent observes EOF."""
    return (
        prefix
        + "import os, time\n"
        + "stdout = os.fstat(1)\n"
        + "for name in os.listdir('/proc/self/fd'):\n"
        + "    fd = int(name)\n"
        + "    if fd <= 2:\n"
        + "        continue\n"
        + "    try:\n"
        + "        info = os.fstat(fd)\n"
        + "    except OSError:\n"
        + "        continue\n"
        + "    if (info.st_dev, info.st_ino) == (stdout.st_dev, stdout.st_ino):\n"
        + "        os.close(fd)\n"
        + "os.close(1)\n"
        + "time.sleep(10)\n"
    )


@pytest.mark.asyncio
async def test_canceled_stdio_request_removes_pending_future_and_progress_callback(
    tmp_path,
):
    transport = _transport(
        tmp_path, "import sys, time\nsys.stdin.readline()\ntime.sleep(30)\n"
    )
    await transport.connect()
    request = asyncio.create_task(
        transport.send_request(
            "tools/call",
            {"_meta": {"progressToken": "progress-1"}},
            on_notification=lambda _event: None,
        )
    )
    try:
        await _wait_for_pending(transport)
        assert "progress-1" in transport._notification_callbacks
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert transport._pending_requests == {}
        assert transport._notification_callbacks == {}
    finally:
        if not request.done():
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        await transport.disconnect()


@pytest.mark.asyncio
async def test_stdout_eof_fails_pending_request_before_request_timeout(tmp_path):
    transport = _transport(
        tmp_path, "import sys\nsys.stdin.readline()\nsys.stdin.readline()\n", timeout=5
    )
    await transport.connect()
    request = asyncio.create_task(transport.send_request("tools/call"))
    try:
        await _wait_for_pending(transport)
        await _release_child(transport)
        with pytest.raises(MCPConnectionError, match="closed stdout"):
            await asyncio.wait_for(request, timeout=2)
        assert transport._pending_requests == {}
    finally:
        if not request.done():
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        await transport.disconnect()


@pytest.mark.asyncio
async def test_disconnect_after_natural_exit_cleans_pending_request(tmp_path):
    transport = _transport(
        tmp_path, "import sys\nsys.stdin.readline()\nsys.stdin.readline()\n", timeout=5
    )
    await transport.connect()
    request = asyncio.create_task(transport.send_request("tools/call"))
    try:
        await _wait_for_pending(transport)
        process = transport._process
        assert process is not None
        await _release_child(transport)
        await asyncio.wait_for(process.wait(), timeout=2)

        await transport.disconnect()
        assert transport._process is None
        assert transport._pending_requests == {}
        assert transport._notification_callbacks == {}
        with pytest.raises((asyncio.CancelledError, MCPConnectionError)):
            await asyncio.wait_for(request, timeout=1)
        assert request.done()
    finally:
        if not request.done():
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        await transport.disconnect()


@pytest.mark.skipif(
    sys.platform != "linux", reason="Uses /proc to close duplicated stdout descriptors"
)
@pytest.mark.asyncio
async def test_request_after_stdout_eof_is_rejected_while_process_is_alive(tmp_path):
    source = _close_stdout_source()
    transport = StdioTransport(sys.executable, ["-u", "-c", source], timeout=5)
    await transport.connect()
    try:
        reader = transport._read_task
        assert reader is not None
        await asyncio.wait_for(asyncio.shield(reader), timeout=2)
        assert transport._process is not None
        assert transport._process.returncode is None

        with pytest.raises(MCPConnectionError, match="reader is no longer running"):
            await asyncio.wait_for(transport.send_request("tools/call"), timeout=0.5)

        assert transport._pending_requests == {}
    finally:
        await transport.disconnect()


@pytest.mark.skipif(
    sys.platform != "linux", reason="Uses /proc to close duplicated stdout descriptors"
)
@pytest.mark.asyncio
async def test_cancel_while_drain_blocked_consumes_early_reader_error(tmp_path):
    source = _close_stdout_source("import sys\nsys.stdin.readline()\n")
    transport = StdioTransport(sys.executable, ["-u", "-c", source], timeout=5)
    await transport.connect()
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop_errors = []
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    entered_drain, release_drain = asyncio.Event(), asyncio.Event()
    request = None
    try:
        original_drain = transport._process.stdin.drain

        async def blocked_drain():
            await original_drain()
            entered_drain.set()
            await release_drain.wait()

        transport._process.stdin.drain = blocked_drain
        request = asyncio.create_task(transport.send_request("tools/call"))
        await asyncio.wait_for(entered_drain.wait(), timeout=2)
        reader = transport._read_task
        assert reader is not None
        await asyncio.wait_for(asyncio.shield(reader), timeout=2)
        assert transport._pending_requests == {}

        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        release_drain.set()
        request = None
        gc.collect()
        await asyncio.sleep(0)

        assert not [
            context
            for context in loop_errors
            if "Future exception was never retrieved" in context.get("message", "")
        ]
    finally:
        release_drain.set()
        if request is not None and not request.done():
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
        loop.set_exception_handler(previous_handler)
        await transport.disconnect()
