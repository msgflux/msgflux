from __future__ import annotations

import pytest

from msgflux.protocols.mcp.client import MCPClient
from msgflux.protocols.mcp.exceptions import MCPConnectionError, MCPTimeoutError
from msgflux.protocols.mcp.transports import BaseTransport


class _DiscoveryTransport(BaseTransport):
    def __init__(self, discovery_results):
        super().__init__()
        self.discovery_results = list(discovery_results)
        self.methods = []
        self.disconnect_calls = 0

    async def connect(self):
        pass

    async def disconnect(self):
        self.disconnect_calls += 1

    async def send_request(self, method, params=None, *, on_notification=None):
        self.methods.append(method)
        if method == "server/discover":
            result = self.discovery_results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        if method == "initialize":
            return {"result": {"protocolVersion": MCPClient.LEGACY_VERSION}}
        raise AssertionError(f"Unexpected request: {method}")

    async def send_notification(self, method, params=None):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        MCPTimeoutError("discovery timed out"),
        MCPConnectionError("discovery connection failed"),
    ],
)
async def test_discovery_transport_failure_does_not_fallback_to_initialize(failure):
    transport = _DiscoveryTransport([failure])
    client = MCPClient(transport, max_retries=1, retry_delay=0)

    with pytest.raises(MCPConnectionError) as exc_info:
        await client.connect()

    assert exc_info.value.__cause__ is failure
    assert transport.methods == ["server/discover"]
    assert transport.disconnect_calls == 1


@pytest.mark.asyncio
async def test_discovery_timeout_retries_discovery_before_legacy_initialize():
    transport = _DiscoveryTransport(
        [
            MCPTimeoutError("temporary discovery timeout"),
            {"result": {"supportedVersions": [MCPClient.MODERN_VERSION]}},
        ]
    )
    client = MCPClient(transport, max_retries=2, retry_delay=0)

    await client.connect()

    assert transport.methods == ["server/discover", "server/discover"]
    assert transport.disconnect_calls == 1
    assert client.protocol_version == MCPClient.MODERN_VERSION


@pytest.mark.asyncio
async def test_discovery_method_not_found_still_falls_back_to_legacy_initialize():
    transport = _DiscoveryTransport(
        [{"error": {"code": -32601, "message": "Method not found"}}]
    )
    client = MCPClient(transport, max_retries=1, retry_delay=0)

    await client.connect()

    assert transport.methods == ["server/discover", "initialize"]
    assert client.protocol_version == MCPClient.LEGACY_VERSION
