"""End-to-end auth tests for MCP HTTP transport."""

import threading
from http.server import ThreadingHTTPServer

import pytest

from msgflux.protocols.mcp import BearerTokenAuth, MCPClient
from msgflux.protocols.mcp.exceptions import MCPConnectionError

from .mcp_auth_http_server import AuthMCPHandler


@pytest.fixture
def auth_mcp_server(monkeypatch):
    monkeypatch.setenv("MCP_AUTH_TOKEN", "test-token")

    server = ThreadingHTTPServer(("127.0.0.1", 0), AuthMCPHandler)
    server.auth_token = "test-token"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        yield f"http://127.0.0.1:{server.server_port}/mcp"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_http_bearer_auth_reaches_mcp_server(auth_mcp_server):
    client = MCPClient.from_http(
        base_url=auth_mcp_server,
        auth=BearerTokenAuth("test-token"),
    )

    async with client:
        tools = await client.list_tools()
        result = await client.call_tool("whoami", {"name": "msgflux"})

    assert [tool.name for tool in tools] == ["whoami"]
    assert result.isError is False
    assert result.content[0].text == "authenticated:msgflux"


@pytest.mark.asyncio
async def test_http_bearer_auth_rejects_invalid_token(auth_mcp_server):
    client = MCPClient.from_http(
        base_url=auth_mcp_server,
        auth=BearerTokenAuth("wrong-token"),
        max_retries=1,
    )

    with pytest.raises(MCPConnectionError, match="HTTP error 401"):
        await client.connect()


@pytest.mark.asyncio
async def test_http_bearer_auth_refreshes_before_request(auth_mcp_server):
    async def refresh_token() -> str:
        return "test-token"

    auth = BearerTokenAuth(
        "expired-token",
        expires_in=-1,
        refresh_callback=refresh_token,
    )
    client = MCPClient.from_http(base_url=auth_mcp_server, auth=auth)

    async with client:
        tools = await client.list_tools()

    assert [tool.name for tool in tools] == ["whoami"]
