"""MCP client and stdio transport connection lifecycle regressions."""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from msgflux.protocols.mcp.client import MCPClient
from msgflux.protocols.mcp import transports as mcp_transports
from msgflux.protocols.mcp.transports import (
    BaseTransport,
    HTTPTransport,
    StdioTransport,
)


_SERVER = r"""
import json
import os
import sys
import threading
import time

pid_path, mode, marker_path = sys.argv[1:]
with open(pid_path, "a", encoding="utf-8") as handle:
    handle.write(f"{os.getpid()}\n")
    handle.flush()
with open(pid_path, encoding="utf-8") as handle:
    process_number = len(handle.readlines())
threading.Thread(target=lambda: (time.sleep(15), os._exit(0)), daemon=True).start()

for line in sys.stdin:
    request = json.loads(line)
    method = request.get("method")
    if method == "server/discover":
        with open(marker_path, "a", encoding="utf-8") as marker:
            marker.write(f"{os.getpid()}\n")
            marker.flush()
        if mode == "hold-discovery":
            sys.stdin.readline()
        elif mode == "cancel-first" and process_number == 1:
            sys.stdin.readline()
            continue
        result = {"supportedVersions": ["2026-07-28"]}
    elif method == "tools/list":
        result = {"tools": []}
    else:
        result = {}
    if "id" in request:
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


def _stdio_client(tmp_path, *, mode="normal", retries=1):
    pid_path = tmp_path / "pids"
    marker_path = tmp_path / "requests"
    server_path = tmp_path / "server.py"
    server_path.write_text(_SERVER, encoding="utf-8")
    client = MCPClient.from_stdio(
        sys.executable,
        ["-u", str(server_path), str(pid_path), mode, str(marker_path)],
        timeout=5,
        max_retries=retries,
        retry_delay=0,
    )
    return client, pid_path, marker_path


async def _wait_for_lines(path, count):
    async def wait():
        while (
            not path.exists()
            or len(path.read_text(encoding="utf-8").splitlines()) < count
        ):
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout=5)


def _read_pids(path):
    if not path.exists():
        return []
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _kill_pids(pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.asyncio
async def test_concurrent_cold_list_tools_shares_one_connection(tmp_path):
    client, pid_path, _ = _stdio_client(tmp_path)
    process = None
    try:
        results = await asyncio.gather(
            *(client.list_tools(use_cache=False) for _ in range(6))
        )
        process = client.transport._process

        assert results == [[], [], [], [], [], []]
        assert len(_read_pids(pid_path)) == 1
        assert process is not None and process.returncode is None
    finally:
        await client.disconnect()
        _kill_pids(_read_pids(pid_path))
    assert process is not None and process.returncode is not None


@pytest.mark.asyncio
async def test_concurrent_explicit_connect_calls_share_one_connection(tmp_path):
    client, pid_path, _ = _stdio_client(tmp_path)
    try:
        await asyncio.gather(*(client.connect() for _ in range(6)))
        process = client.transport._process

        assert client._initialized
        assert len(_read_pids(pid_path)) == 1
        assert process is not None and process.returncode is None
    finally:
        await client.disconnect()
        _kill_pids(_read_pids(pid_path))


@pytest.mark.asyncio
async def test_disconnect_waits_for_connect_negotiation(tmp_path):
    client, pid_path, marker_path = _stdio_client(tmp_path, mode="hold-discovery")
    connect_task = asyncio.create_task(client.connect())
    disconnect_started = asyncio.Event()

    async def disconnect():
        disconnect_started.set()
        await client.disconnect()

    disconnect_task = None
    try:
        await _wait_for_lines(marker_path, 1)
        process = client.transport._process
        assert process is not None
        disconnect_task = asyncio.create_task(disconnect())
        await disconnect_started.wait()
        await asyncio.sleep(0)
        assert not disconnect_task.done()

        process.stdin.write(b"\n")
        await process.stdin.drain()
        await connect_task
        await disconnect_task

        assert not client._initialized
        assert process.returncode is not None
        assert len(_read_pids(pid_path)) == 1
    finally:
        for task in (connect_task, disconnect_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (connect_task, disconnect_task) if task is not None),
            return_exceptions=True,
        )
        await client.disconnect()
        _kill_pids(_read_pids(pid_path))


@pytest.mark.asyncio
async def test_cancelled_negotiation_cleans_up_and_allows_retry(tmp_path):
    client, pid_path, marker_path = _stdio_client(
        tmp_path, mode="cancel-first", retries=1
    )
    connect_task = asyncio.create_task(client.connect())
    try:
        await _wait_for_lines(marker_path, 1)
        process = client.transport._process
        reader = client.transport._read_task
        assert process is not None and reader is not None

        connect_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await connect_task

        assert process.returncode is not None
        assert reader.done()
        assert client.transport._process is None
        assert not client._initialized

        await client.connect()
        assert client._initialized
        assert len(_read_pids(pid_path)) == 2
    finally:
        if not connect_task.done():
            connect_task.cancel()
            await asyncio.gather(connect_task, return_exceptions=True)
        await client.disconnect()
        _kill_pids(_read_pids(pid_path))


@pytest.mark.asyncio
async def test_direct_concurrent_stdio_connect_calls_share_one_process(tmp_path):
    client, pid_path, _ = _stdio_client(tmp_path)
    transport = client.transport
    try:
        await asyncio.gather(*(transport.connect() for _ in range(6)))
        await _wait_for_lines(pid_path, 1)
        process = transport._process

        assert len(_read_pids(pid_path)) == 1
        assert process is not None and process.returncode is None
    finally:
        await transport.disconnect()
        _kill_pids(_read_pids(pid_path))


@pytest.mark.asyncio
async def test_cancelled_connect_preserves_cancellation_if_cleanup_fails():
    entered_discovery = asyncio.Event()
    cleanup_error = OSError("cleanup failed")

    class BlockingTransport(BaseTransport):
        async def connect(self):
            pass

        async def disconnect(self):
            raise cleanup_error

        async def send_request(self, method, params=None, *, on_notification=None):
            entered_discovery.set()
            await asyncio.Event().wait()

        async def send_notification(self, method, params=None):
            pass

    client = MCPClient(BlockingTransport(), max_retries=1)
    task = asyncio.create_task(client.connect())
    await entered_discovery.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    assert not client._initialized
    assert any("cleanup failed" in note for note in exc_info.value.__notes__)


@pytest.mark.asyncio
async def test_repeated_stdio_connect_cancellation_during_spawn_reaps_child(
    tmp_path, monkeypatch
):
    server_path = tmp_path / "server.py"
    server_path.write_text("import sys\nsys.stdin.readline()\n", encoding="utf-8")
    transport = StdioTransport(sys.executable, ["-u", str(server_path)])
    original_spawn = asyncio.create_subprocess_exec
    spawned = asyncio.Event()
    joining_start = asyncio.Event()
    release = asyncio.Event()
    child = None
    original_settle = mcp_transports._settle_task

    async def delayed_spawn(*args, **kwargs):
        nonlocal child
        child = await original_spawn(*args, **kwargs)
        spawned.set()
        await release.wait()
        return child

    async def observe_start_join(task):
        if task.get_coro().__name__ == "delayed_spawn":
            joining_start.set()
        return await original_settle(task)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
    monkeypatch.setattr(mcp_transports, "_settle_task", observe_start_join)
    connect_task = asyncio.create_task(transport.connect())
    try:
        await asyncio.wait_for(spawned.wait(), timeout=5)
        connect_task.cancel()
        await asyncio.wait_for(joining_start.wait(), timeout=5)
        connect_task.cancel()
        release.set()

        with pytest.raises(asyncio.CancelledError):
            await connect_task

        assert child is not None and child.returncode is not None
        assert transport._process is None
        assert transport._read_task is None
    finally:
        release.set()
        if not connect_task.done():
            connect_task.cancel()
            await asyncio.gather(connect_task, return_exceptions=True)
        if child is not None and child.returncode is None:
            child.kill()
            await child.wait()
        await transport.disconnect()


@pytest.mark.asyncio
async def test_cancelled_stdio_disconnect_reaps_sigterm_resistant_child(tmp_path):
    ready = tmp_path / "ready"
    term_received = tmp_path / "term-received"
    server_path = tmp_path / "ignore_term.py"
    server_path.write_text(
        "import signal, sys, time\n"
        "ready, marker = sys.argv[1:]\n"
        "open(ready, 'w').close()\n"
        "signal.signal(signal.SIGTERM, lambda *_: open(marker, 'w').close())\n"
        "while True: time.sleep(0.05)\n",
        encoding="utf-8",
    )
    transport = StdioTransport(
        sys.executable, [str(server_path), str(ready), str(term_received)]
    )
    process = None
    disconnect_task = None

    async def wait_for_file(path):
        async def wait():
            while not path.exists():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait(), timeout=5)

    try:
        await transport.connect()
        process = transport._process
        assert process is not None
        await wait_for_file(ready)

        disconnect_task = asyncio.create_task(transport.disconnect())
        await wait_for_file(term_received)
        disconnect_task.cancel()
        await asyncio.sleep(0)
        assert not disconnect_task.done()
        assert transport._process is process
        assert process.returncode is None
        disconnect_task.cancel()
        await asyncio.sleep(0)
        assert not disconnect_task.done()
        assert transport._process is process

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(disconnect_task, timeout=8)

        assert process.returncode is not None
        assert transport._process is None
        assert transport._read_task is None
    finally:
        if disconnect_task is not None and not disconnect_task.done():
            disconnect_task.cancel()
            await asyncio.gather(disconnect_task, return_exceptions=True)
        if process is not None and process.returncode is None:
            process.kill()
            await asyncio.wait_for(process.wait(), timeout=3)
        await transport.disconnect()


@pytest.mark.asyncio
async def test_concurrent_cold_http_calls_share_one_owned_client(monkeypatch):
    clients = []

    async def post(_url, *, json, **_kwargs):
        response = MagicMock()
        response.status_code = 200
        response.headers = {"content-type": "application/json"}
        response.raise_for_status.return_value = None
        if json["method"] == "server/discover":
            result = {"supportedVersions": [MCPClient.MODERN_VERSION]}
        else:
            result = {"tools": []}
        response.json.return_value = {"result": result}
        return response

    def make_client(**_kwargs):
        http_client = AsyncMock()
        http_client.post.side_effect = post
        clients.append(http_client)
        return http_client

    monkeypatch.setattr(mcp_transports.httpx2, "AsyncClient", make_client)
    monkeypatch.setattr(mcp_transports.httpx2, "Timeout", MagicMock())
    monkeypatch.setattr(mcp_transports.httpx2, "Limits", MagicMock())
    client = MCPClient.from_http("http://127.0.0.1:1/mcp", max_retries=1)

    try:
        results = await asyncio.gather(
            *(client.list_tools(use_cache=False) for _ in range(6))
        )

        assert results == [[], [], [], [], [], []]
        assert len(clients) == 1
        assert client.transport._http_client is clients[0]
    finally:
        await client.disconnect()

    clients[0].aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_http_negotiation_closes_owned_client(monkeypatch):
    request_started = asyncio.Event()
    http_client = AsyncMock()

    async def pending_post(*_args, **_kwargs):
        request_started.set()
        await asyncio.Event().wait()

    http_client.post.side_effect = pending_post
    monkeypatch.setattr(
        mcp_transports.httpx2, "AsyncClient", MagicMock(return_value=http_client)
    )
    monkeypatch.setattr(mcp_transports.httpx2, "Timeout", MagicMock())
    monkeypatch.setattr(mcp_transports.httpx2, "Limits", MagicMock())
    client = MCPClient.from_http("http://127.0.0.1:1/mcp", max_retries=1)
    task = asyncio.create_task(client.connect())
    await request_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.transport._http_client is None
    http_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_direct_http_connect_waits_for_disconnect_to_finish(monkeypatch):
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    clients = []

    class FakeClient:
        async def aclose(self):
            close_started.set()
            await release_close.wait()

    def make_client(**_kwargs):
        client = FakeClient()
        clients.append(client)
        return client

    monkeypatch.setattr(mcp_transports.httpx2, "AsyncClient", make_client)
    monkeypatch.setattr(mcp_transports.httpx2, "Timeout", MagicMock())
    monkeypatch.setattr(mcp_transports.httpx2, "Limits", MagicMock())
    transport = HTTPTransport("http://127.0.0.1:1/mcp")
    disconnect_task = None
    connect_task = None

    try:
        await transport.connect()
        disconnect_task = asyncio.create_task(transport.disconnect())
        await asyncio.wait_for(close_started.wait(), timeout=3)
        connect_task = asyncio.create_task(transport.connect())
        await asyncio.sleep(0)
        assert not connect_task.done()

        release_close.set()
        await asyncio.wait_for(disconnect_task, timeout=3)
        await asyncio.wait_for(connect_task, timeout=3)

        assert len(clients) == 2
        assert transport._http_client is clients[1]
    finally:
        release_close.set()
        await asyncio.gather(
            *(task for task in (disconnect_task, connect_task) if task is not None),
            return_exceptions=True,
        )
