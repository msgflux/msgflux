"""Shared fixtures for MCP protocol live tests."""

import asyncio
import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import pytest_asyncio

from msgflux.protocols.mcp import MCPClient

_SERVER_SCRIPT = str(Path(__file__).parent / "mcp_simple_server.py")
_LEGACY_SERVER_SCRIPT = str(Path(__file__).parent / "mcp_legacy_server.py")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def live_mcp_client():
    """Shared MCP client that spawns the test server once per module.

    Using module scope reduces subprocess startup overhead from ~1.5s per test
    to a single startup for the entire test_mcp_live.py module.
    """
    async with MCPClient.from_stdio(
        command="uv",
        args=["run", _SERVER_SCRIPT],
        timeout=15.0,
    ) as client:
        yield client


@asynccontextmanager
async def _http_client(script: str, *, page_size: int | None = None):
    """Start a FastMCP script over HTTP and clean up its process."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {
        **os.environ,
        "MSGFLUX_TEST_MCP_HTTP_PORT": str(port),
    }
    if page_size is not None:
        env["MSGFLUX_TEST_MCP_PAGE_SIZE"] = str(page_size)
    process = await asyncio.create_subprocess_exec(
        "uv",
        "run",
        script,
        env=env,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        for _ in range(150):
            if process.returncode is not None:
                raise RuntimeError("FastMCP HTTP server exited during startup")
            try:
                _, writer = await asyncio.open_connection("127.0.0.1", port)
            except OSError:
                await asyncio.sleep(0.1)
            else:
                writer.close()
                await writer.wait_closed()
                break
        else:
            raise TimeoutError("FastMCP HTTP server did not start")

        async with MCPClient.from_http(
            base_url=f"http://127.0.0.1:{port}/mcp", max_retries=1
        ) as client:
            yield client
    finally:
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()


@pytest_asyncio.fixture
async def live_http_mcp_client():
    """Run a paginated modern FastMCP HTTP server."""
    async with _http_client(_SERVER_SCRIPT, page_size=1) as client:
        yield client


@pytest_asyncio.fixture
async def live_legacy_http_mcp_client():
    """Run a FastMCP 3 server for HTTP fallback tests."""
    async with _http_client(_LEGACY_SERVER_SCRIPT) as client:
        yield client
