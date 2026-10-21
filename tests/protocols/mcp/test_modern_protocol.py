"""Protocol-era and wire-format regression tests for MCP 2026-07-28."""

import pytest

from msgflux.protocols.mcp import (
    MCPClient,
    MCPConnectionError,
    MCPInputRequiredError,
    MCPTool,
    MCPToolResult,
    extract_tool_result_text,
)
from msgflux.protocols.mcp.transports import BaseTransport, HTTPTransport


class ScriptedTransport(BaseTransport):
    def __init__(self, responses):
        super().__init__()
        self.responses = responses
        self.requests = []
        self.notifications = []

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send_request(self, method, params=None, *, on_notification=None):
        self.requests.append((method, params))
        response = self.responses[method]
        return response.pop(0) if isinstance(response, list) else response

    async def send_notification(self, method, params=None):
        self.notifications.append((method, params))


@pytest.mark.asyncio
async def test_modern_probe_and_per_request_metadata():
    transport = ScriptedTransport(
        {
            "server/discover": {"result": {"supportedVersions": ["2026-07-28"]}},
            "tools/list": {"result": {"tools": [], "ttlMs": 0}},
        }
    )
    client = MCPClient(transport, max_retries=1)

    await client.connect()
    await client.list_tools()

    assert client.protocol_version == "2026-07-28"
    assert [method for method, _ in transport.requests] == [
        "server/discover",
        "tools/list",
    ]
    assert transport.notifications == []
    for _, params in transport.requests:
        assert (
            params["_meta"]["io.modelcontextprotocol/protocolVersion"] == "2026-07-28"
        )
        assert "io.modelcontextprotocol/clientCapabilities" in params["_meta"]


@pytest.mark.asyncio
async def test_legacy_fallback_keeps_handshake_and_notification():
    transport = ScriptedTransport(
        {
            "server/discover": {"error": {"code": -32601, "message": "Not found"}},
            "initialize": {"result": {"protocolVersion": "2024-11-05"}},
            "tools/list": {"result": {"tools": []}},
        }
    )
    client = MCPClient(transport, max_retries=1)

    await client.connect()
    await client.list_tools()

    assert client.protocol_version == "2024-11-05"
    assert [method for method, _ in transport.requests] == [
        "server/discover",
        "initialize",
        "tools/list",
    ]
    assert transport.requests[-1][1] is None
    assert transport.notifications == [("notifications/initialized", None)]


@pytest.mark.asyncio
async def test_version_mismatch_does_not_downgrade_to_legacy():
    transport = ScriptedTransport(
        {
            "server/discover": {
                "error": {
                    "code": -32022,
                    "data": {"supported": ["2099-01-01"]},
                }
            }
        }
    )
    client = MCPClient(transport, max_retries=1)

    with pytest.raises(MCPConnectionError, match="No supported MCP protocol"):
        await client.connect()

    assert [method for method, _ in transport.requests] == ["server/discover"]


@pytest.mark.asyncio
async def test_modern_zero_ttl_refetches_tool_list():
    transport = ScriptedTransport(
        {
            "server/discover": {"result": {"supportedVersions": ["2026-07-28"]}},
            "tools/list": {"result": {"tools": [], "ttlMs": 0}},
        }
    )
    client = MCPClient(transport, max_retries=1)

    await client.connect()
    await client.list_tools()
    await client.list_tools()

    assert [method for method, _ in transport.requests].count("tools/list") == 2


def test_http_headers_and_header_annotation_encoding():
    transport = HTTPTransport("http://localhost:8000/mcp")
    tool = MCPTool(
        name="lookup",
        description="Lookup",
        inputSchema={
            "type": "object",
            "properties": {
                "region": {"type": "string", "x-mcp-header": "Region"},
            },
        },
    )
    assert transport.register_tool_schemas([tool]) == [tool]

    headers = transport._request_headers(
        "tools/call",
        {
            "name": "lookup",
            "arguments": {"region": "São Paulo"},
            "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"},
        },
        {},
    )

    assert headers["MCP-Protocol-Version"] == "2026-07-28"
    assert headers["Mcp-Method"] == "tools/call"
    assert headers["Mcp-Name"] == "lookup"
    assert headers["Mcp-Param-region"] == "=?base64?U8OjbyBQYXVsbw==?="


def test_http_rejects_invalid_header_annotation():
    transport = HTTPTransport("http://localhost:8000/mcp")
    tool = MCPTool(
        name="bad",
        description="Bad",
        inputSchema={
            "type": "object",
            "properties": {
                "region": {"type": "number", "x-mcp-header": "Region"},
            },
        },
    )

    assert transport.register_tool_schemas([tool]) == []


def test_sse_skips_notifications_and_uses_final_matching_response():
    messages = (
        'data: {"jsonrpc":"2.0","method":"notifications/progress",'
        '"params":{"progress":1}}\n\n'
        'data: {"jsonrpc":"2.0","id":"99","result":{"wrong":true}}\n\n'
        'data: {"jsonrpc":"2.0","id":"1","result":{"ok":true}}\n\n'
    )

    assert HTTPTransport._parse_sse(messages, "1") == {
        "jsonrpc": "2.0",
        "id": "1",
        "result": {"ok": True},
    }


def test_agent_proxy_does_not_treat_input_required_as_success():
    result = MCPToolResult(
        content=[],
        resultType="input_required",
        inputRequests={"confirm": {"method": "elicitation/create"}},
        requestState="opaque",
    )

    with pytest.raises(MCPInputRequiredError) as exc_info:
        extract_tool_result_text(result)

    assert exc_info.value.input_requests == result.inputRequests
    assert exc_info.value.request_state == "opaque"
