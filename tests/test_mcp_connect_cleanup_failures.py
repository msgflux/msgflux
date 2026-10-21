from __future__ import annotations

import asyncio

import pytest

from msgflux.protocols.mcp.client import MCPClient
from msgflux.protocols.mcp.exceptions import MCPConnectionError
from msgflux.protocols.mcp.transports import BaseTransport


class _Transport(BaseTransport):
    def __init__(self):
        super().__init__()
        self.connect_calls = 0
        self.disconnect_calls = 0
        self.request_calls = []
        self.notification_error = None
        self.connect_errors = []
        self.disconnect_errors = []

    async def connect(self):
        self.connect_calls += 1
        if self.connect_errors:
            raise self.connect_errors.pop(0)

    async def disconnect(self):
        self.disconnect_calls += 1
        if self.disconnect_errors:
            raise self.disconnect_errors.pop(0)

    async def send_request(self, method, params=None, *, on_notification=None):
        self.request_calls.append(method)
        if method == "server/discover":
            return {"error": {"code": -32601, "message": "legacy server"}}
        if method == "initialize":
            return {
                "result": {
                    "protocolVersion": MCPClient.LEGACY_VERSION,
                    "sessionId": "session",
                }
            }
        raise AssertionError(f"Unexpected MCP request: {method}")

    async def send_notification(self, method, params=None):
        if self.notification_error is not None:
            raise self.notification_error


@pytest.mark.asyncio
async def test_connect_cleanup_failure_preserves_primary_and_clears_state():
    transport = _Transport()
    primary = RuntimeError("connect failed")
    cleanup = OSError("disconnect failed")
    transport.connect_errors.append(primary)
    transport.disconnect_errors.append(cleanup)
    client = MCPClient(transport, max_retries=3, retry_delay=0)

    with pytest.raises(MCPConnectionError) as exc_info:
        await client.connect()

    assert exc_info.value.__cause__ is primary
    assert any("disconnect failed" in note for note in primary.__notes__)
    assert transport.connect_calls == 1
    assert transport.disconnect_calls == 1
    assert client._last_error is primary
    assert client._initialized is False
    assert client.protocol_version is None


@pytest.mark.asyncio
async def test_failed_initialized_notification_and_cleanup_clear_session_state():
    transport = _Transport()
    primary = RuntimeError("initialized notification failed")
    cleanup = OSError("disconnect failed")
    transport.notification_error = primary
    transport.disconnect_errors.append(cleanup)
    client = MCPClient(transport, max_retries=3, retry_delay=0)

    with pytest.raises(MCPConnectionError) as exc_info:
        await client.connect()

    assert exc_info.value.__cause__ is primary
    assert any("disconnect failed" in note for note in primary.__notes__)
    assert transport.connect_calls == 1
    assert transport.disconnect_calls == 1
    assert client._initialized is False
    assert client.protocol_version is None
    assert client._modern is False


@pytest.mark.asyncio
async def test_connect_retries_after_successful_cleanup():
    transport = _Transport()
    transport.connect_errors.append(RuntimeError("first attempt failed"))
    client = MCPClient(transport, max_retries=3, retry_delay=0)

    await client.connect()

    assert transport.connect_calls == 2
    assert transport.disconnect_calls == 1
    assert client._initialized is True
    assert client.protocol_version == MCPClient.LEGACY_VERSION


@pytest.mark.asyncio
async def test_public_disconnect_resets_state_when_transport_cleanup_fails():
    transport = _Transport()
    cleanup = OSError("disconnect failed")
    transport.disconnect_errors.append(cleanup)
    client = MCPClient(transport)
    client._initialized = True
    client.protocol_version = MCPClient.LEGACY_VERSION
    client._modern = True
    client._tools_cache = []
    client._resources_cache = []
    client._prompts_cache = []
    client._cache_expiry["tools"] = 123.0

    with pytest.raises(OSError) as exc_info:
        await client.disconnect()

    assert exc_info.value is cleanup
    assert client._initialized is False
    assert client.protocol_version is None
    assert client._modern is False
    assert client._tools_cache is None
    assert client._resources_cache is None
    assert client._prompts_cache is None
    assert client._cache_expiry == {}


@pytest.mark.asyncio
async def test_public_disconnect_preserves_cancellation_and_resets_state():
    transport = _Transport()
    cancellation = asyncio.CancelledError("disconnect cancelled")
    transport.disconnect_errors.append(cancellation)
    client = MCPClient(transport)
    client._initialized = True
    client.protocol_version = MCPClient.LEGACY_VERSION
    client._modern = True
    client._tools_cache = []

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await client.disconnect()

    assert exc_info.value is cancellation
    assert client._initialized is False
    assert client.protocol_version is None
    assert client._modern is False
    assert client._tools_cache is None
