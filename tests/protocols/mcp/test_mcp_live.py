"""Live integration tests for MCPClient with a real FastMCP server.

The server script (mcp_simple_server.py) is launched as a subprocess by
StdioTransport using `uv run`, which installs fastmcp automatically via
the inline `# /// script` dependency metadata in that file.

All tests share a single server process via the `live_mcp_client` fixture
defined in conftest.py, reducing subprocess startup overhead significantly.
All tests use module-scoped event loop to share the same loop as the fixture.
"""

import pytest

# Share the same module-scoped event loop as the live_mcp_client fixture.
pytestmark = [pytest.mark.asyncio(loop_scope="module")]


async def test_connect_and_list_tools(live_mcp_client):
    """Connect to the server and verify the expected tools are listed."""
    assert live_mcp_client.protocol_version == "2026-07-28"
    tools = await live_mcp_client.list_tools()

    names = {t.name for t in tools}
    assert "add" in names
    assert "echo" in names
    assert "divide" in names


async def test_call_add(live_mcp_client):
    """Call the add tool and verify the result."""
    result = await live_mcp_client.call_tool("add", {"a": 3, "b": 4})

    assert not result.isError
    assert "7" in result.content[0].text


async def test_call_echo(live_mcp_client):
    """Call the echo tool and verify the message is returned unchanged."""
    result = await live_mcp_client.call_tool("echo", {"message": "hello msgflux"})

    assert not result.isError
    assert "hello msgflux" in result.content[0].text


async def test_tool_error_returned_as_result(live_mcp_client):
    """A tool that raises should come back as isError=True, not an exception."""
    result = await live_mcp_client.call_tool("divide", {"a": 1.0, "b": 0.0})

    assert result.isError
    assert result.content  # error message present


async def test_ping(live_mcp_client):
    """Verify ping returns True when server is running."""
    assert await live_mcp_client.ping() is True


async def test_list_tools_cache(live_mcp_client):
    """Second call with use_cache=True should return same objects without a round-trip."""
    tools_first = await live_mcp_client.list_tools(use_cache=False)
    tools_cached = await live_mcp_client.list_tools(use_cache=True)

    assert [t.name for t in tools_first] == [t.name for t in tools_cached]


async def test_stdio_modern_progress(live_mcp_client):
    progress = []

    result = await live_mcp_client.call_tool(
        "report_progress",
        _progress_callback=lambda value, message: progress.append((value, message)),
    )

    assert result.structuredContent == {"result": "done"}
    assert progress == [(1, "half"), (2, "done")]
