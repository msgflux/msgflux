"""Integration test for a FastMCP server using the legacy handshake."""

from pathlib import Path

import pytest

from msgflux.protocols.mcp import MCPClient


@pytest.mark.asyncio
async def test_legacy_server_fallback_and_tool_call():
    script = str(Path(__file__).parent / "mcp_legacy_server.py")

    async with MCPClient.from_stdio(
        command="uv", args=["run", script], timeout=15, max_retries=1
    ) as client:
        assert client.protocol_version != "2026-07-28"
        tools = await client.list_tools()
        result = await client.call_tool("add", {"a": 2, "b": 5})

    assert [tool.name for tool in tools] == ["add"]
    assert result.isError is False
    assert result.content[0].text == "7"


@pytest.mark.asyncio
async def test_legacy_http_server_fallback_and_tool_call(live_legacy_http_mcp_client):
    client = live_legacy_http_mcp_client

    assert client.protocol_version != "2026-07-28"
    tools = await client.list_tools()
    result = await client.call_tool("add", {"a": 2, "b": 5})

    assert [tool.name for tool in tools] == ["add"]
    assert result.isError is False
    assert result.content[0].text == "7"
