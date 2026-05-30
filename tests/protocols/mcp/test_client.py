"""Tests for MCP Client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from msgflux.protocols.mcp.client import MCPClient
from msgflux.protocols.mcp.exceptions import MCPConnectionError, MCPError
from msgflux.protocols.mcp.transports import BaseTransport
from msgflux.protocols.mcp.types import MCPContent, MCPTool, MCPToolResult


class MockTransport(BaseTransport):
    """Mock transport for testing."""

    def __init__(self):
        self.connected = False
        self.responses = {}

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def send_request(self, method, params=None):
        if not self.connected:
            raise MCPConnectionError("Not connected")
        return self.responses.get(method, {"jsonrpc": "2.0", "id": "1", "result": {}})

    async def send_notification(self, method, params=None):
        pass


class TestMCPClient:
    """Tests for MCPClient."""

    def test_init(self):
        """Test MCPClient initialization."""
        transport = MockTransport()
        client = MCPClient(
            transport=transport, max_retries=5, retry_delay=2.0, auto_reconnect=True
        )

        assert client.transport is transport
        assert client.max_retries == 5
        assert client.retry_delay == 2.0
        assert client.auto_reconnect is True

    def test_init_defaults(self):
        """Test default parameters."""
        transport = MockTransport()
        client = MCPClient(transport=transport)

        assert client.max_retries == 3
        assert client.retry_delay == 1.0
        assert client.auto_reconnect is True

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test basic connection."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"capabilities": {}},
        }

        client = MCPClient(transport=transport)
        await client.connect()

        assert client._initialized is True
        assert transport.connected is True

    @pytest.mark.asyncio
    async def test_connect_with_retry_success_first_attempt(self):
        """Test successful connection on first attempt."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"capabilities": {}},
        }

        client = MCPClient(transport=transport, max_retries=3)
        await client.connect()

        assert client._connection_attempts == 1
        assert client._last_error is None

    @pytest.mark.asyncio
    async def test_connect_with_retry_failure_then_success(self):
        """Test connection succeeds after initial failures."""
        transport = MockTransport()
        attempt = {"count": 0}

        original_connect = transport.connect

        async def failing_connect():
            attempt["count"] += 1
            if attempt["count"] < 3:
                raise Exception("Connection failed")
            await original_connect()

        transport.connect = failing_connect
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"capabilities": {}},
        }

        client = MCPClient(transport=transport, max_retries=5, retry_delay=0.01)
        await client.connect()

        assert client._connection_attempts == 3
        assert client._initialized is True

    @pytest.mark.asyncio
    async def test_connect_max_retries_exceeded(self):
        """Test connection fails after max retries."""
        transport = MockTransport()

        async def always_fail():
            raise Exception("Connection failed")

        transport.connect = always_fail

        client = MCPClient(transport=transport, max_retries=3, retry_delay=0.01)

        with pytest.raises(MCPConnectionError) as exc_info:
            await client.connect()

        assert "Failed to connect after 3 attempts" in str(exc_info.value)
        assert client._connection_attempts == 3

    @pytest.mark.asyncio
    async def test_connect_disconnects_transport_after_initialize_failure(self):
        """A failed initialize should not leave a connected transport behind."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "error": {"message": "initialize failed"},
        }
        transport.disconnect = AsyncMock(side_effect=transport.disconnect)

        client = MCPClient(transport=transport, max_retries=1)

        with pytest.raises(MCPConnectionError):
            await client.connect()

        transport.disconnect.assert_awaited_once()
        assert transport.connected is False

    @pytest.mark.asyncio
    async def test_connect_preserves_initialize_error_when_cleanup_fails(self):
        """Cleanup failures should not replace the original connection error."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "error": {"message": "initialize failed"},
        }
        transport.disconnect = AsyncMock(side_effect=RuntimeError("cleanup failed"))

        client = MCPClient(transport=transport, max_retries=1)

        with pytest.raises(MCPConnectionError, match="Failed to initialize"):
            await client.connect()

        transport.disconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnect."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"capabilities": {}},
        }

        client = MCPClient(transport=transport)
        await client.connect()
        await client.disconnect()

        assert client._initialized is False
        assert transport.connected is False

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test listing tools."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"capabilities": {}},
        }
        transport.responses["tools/list"] = {
            "jsonrpc": "2.0",
            "id": "2",
            "result": {
                "tools": [
                    {
                        "name": "read_file",
                        "description": "Read a file",
                        "inputSchema": {"type": "object"},
                    },
                    {
                        "name": "write_file",
                        "description": "Write a file",
                        "inputSchema": {"type": "object"},
                    },
                ]
            },
        }

        client = MCPClient(transport=transport)
        await client.connect()

        tools = await client.list_tools()

        assert len(tools) == 2
        assert tools[0].name == "read_file"
        assert tools[1].name == "write_file"

    @pytest.mark.asyncio
    async def test_list_tools_caching(self):
        """Test that list_tools uses cache."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"capabilities": {}},
        }
        transport.responses["tools/list"] = {
            "jsonrpc": "2.0",
            "id": "2",
            "result": {
                "tools": [{"name": "test", "description": "Test", "inputSchema": {}}]
            },
        }

        client = MCPClient(transport=transport)
        await client.connect()

        # First call without cache
        tools1 = await client.list_tools(use_cache=False)
        # Second call with cache should return the same cached result
        tools2 = await client.list_tools(use_cache=True)

        assert len(tools1) == 1
        assert len(tools2) == 1
        assert tools1[0].name == tools2[0].name  # Same tool data

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test successful tool call."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"capabilities": {}},
        }
        transport.responses["tools/call"] = {
            "jsonrpc": "2.0",
            "id": "3",
            "result": {
                "content": [{"type": "text", "text": "File content here"}],
                "isError": False,
            },
        }

        client = MCPClient(transport=transport)
        await client.connect()

        result = await client.call_tool("read_file", {"path": "/test.txt"})

        assert isinstance(result, MCPToolResult)
        assert result.isError is False
        assert len(result.content) == 1
        assert result.content[0].text == "File content here"

    @pytest.mark.asyncio
    async def test_call_tool_error(self):
        """Test tool call that returns an error."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"capabilities": {}},
        }
        transport.responses["tools/call"] = {
            "jsonrpc": "2.0",
            "id": "3",
            "error": {"message": "File not found"},
        }

        client = MCPClient(transport=transport)
        await client.connect()

        result = await client.call_tool("read_file", {"path": "/nonexistent.txt"})

        assert isinstance(result, MCPToolResult)
        assert result.isError is True
        assert "File not found" in result.content[0].text

    @pytest.mark.asyncio
    async def test_auto_reconnect_on_call(self):
        """Test auto-reconnect when calling tool while disconnected."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"capabilities": {}},
        }
        transport.responses["tools/call"] = {
            "jsonrpc": "2.0",
            "id": "2",
            "result": {
                "content": [{"type": "text", "text": "Success"}],
                "isError": False,
            },
        }

        client = MCPClient(transport=transport, auto_reconnect=True)

        # Don't connect initially
        # Call tool should trigger auto-reconnect
        result = await client.call_tool("test_tool", {})

        assert client._initialized is True
        assert result.isError is False

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using client as async context manager."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"capabilities": {}},
        }

        async with MCPClient(transport=transport) as client:
            assert client._initialized is True

        # After exiting context
        assert client._initialized is False

    def test_from_stdio_factory(self):
        """Test creating client from stdio factory method."""
        client = MCPClient.from_stdio(
            command="mcp-server",
            args=["--flag"],
            max_retries=5,
            retry_delay=2.0,
            auto_reconnect=False,
        )

        assert client.max_retries == 5
        assert client.retry_delay == 2.0
        assert client.auto_reconnect is False

    def test_from_http_factory(self):
        """Test creating client from http factory method."""
        client = MCPClient.from_http(
            base_url="http://localhost:8080",
            max_retries=3,
            pool_limits={"max_connections": 50, "max_keepalive_connections": 10},
        )

        assert client.max_retries == 3
        assert client.transport.pool_limits["max_connections"] == 50

    @pytest.mark.asyncio
    async def test_ping(self):
        """Test ping method."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"capabilities": {}},
        }
        transport.responses["ping"] = {"jsonrpc": "2.0", "id": "2", "result": {}}

        client = MCPClient(transport=transport)
        await client.connect()

        is_alive = await client.ping()
        assert is_alive is True

    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test listing resources."""
        transport = MockTransport()
        transport.responses["initialize"] = {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"capabilities": {}},
        }
        transport.responses["resources/list"] = {
            "jsonrpc": "2.0",
            "id": "2",
            "result": {
                "resources": [
                    {
                        "uri": "file:///test.txt",
                        "name": "test_file",
                        "description": "A test file",
                    }
                ]
            },
        }

        client = MCPClient(transport=transport)
        await client.connect()

        resources = await client.list_resources()

        assert len(resources) == 1
        assert resources[0].uri == "file:///test.txt"
        assert resources[0].name == "test_file"
