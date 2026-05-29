"""Tests for ToolLibrary with MCP integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from msgflux.protocols.mcp.types import MCPTool


class TestToolLibraryMCPIntegration:
    """Tests for ToolLibrary MCP integration."""

    @patch("msgflux.nn.modules.tool.MCPClient")
    @patch("msgflux.nn.modules.tool.filter_tools")
    @patch("msgflux.nn.modules.tool.F")
    def test_initialize_stdio_mcp_clients(
        self, mock_F, mock_filter_tools, mock_mcp_client
    ):
        """Test initializing ToolLibrary with stdio MCP server."""
        from msgflux.nn.modules.tool import ToolLibrary

        # Mock MCP client with async methods
        mock_client_instance = MagicMock()
        mock_client_instance.connect = AsyncMock()
        mock_client_instance.list_tools = AsyncMock()
        mock_mcp_client.from_stdio.return_value = mock_client_instance

        # Mock tools
        mock_tools = [
            MCPTool(name="read_file", description="Read file", inputSchema={}),
            MCPTool(name="write_file", description="Write file", inputSchema={}),
        ]
        mock_filter_tools.return_value = mock_tools

        # Mock F.wait_for to return appropriate values
        # First call is client.connect(), second is client.list_tools()
        mock_F.wait_for.side_effect = [None, mock_tools]

        # Create ToolLibrary with MCP server
        mcp_servers = [
            {
                "name": "fs",
                "transport": "stdio",
                "command": "mcp-server-fs",
                "args": ["--flag"],
            }
        ]

        library = ToolLibrary(name="test", tools=[], mcp_servers=mcp_servers)

        # Verify client was created
        mock_mcp_client.from_stdio.assert_called_once()
        call_kwargs = mock_mcp_client.from_stdio.call_args[1]
        assert call_kwargs["command"] == "mcp-server-fs"
        assert call_kwargs["args"] == ["--flag"]

        # Verify MCP client was stored
        assert "fs" in library.mcp_clients
        assert library.mcp_clients["fs"]["client"] is mock_client_instance

    @patch("msgflux.nn.modules.tool.MCPClient")
    @patch("msgflux.nn.modules.tool.filter_tools")
    @patch("msgflux.nn.modules.tool.F")
    def test_initialize_http_mcp_clients(
        self, mock_F, mock_filter_tools, mock_mcp_client
    ):
        """Test initializing ToolLibrary with HTTP MCP server."""
        from msgflux.nn.modules.tool import ToolLibrary

        mock_client_instance = MagicMock()
        mock_client_instance.connect = AsyncMock()
        mock_client_instance.list_tools = AsyncMock()
        mock_mcp_client.from_http.return_value = mock_client_instance

        mock_tools = [MCPTool(name="api_call", description="API call", inputSchema={})]
        mock_filter_tools.return_value = mock_tools

        mock_F.wait_for.side_effect = [None, mock_tools]

        mcp_servers = [
            {
                "name": "api",
                "transport": "http",
                "base_url": "http://localhost:8080",
                "headers": {"Auth": "token"},
            }
        ]

        library = ToolLibrary(name="test", tools=[], mcp_servers=mcp_servers)

        mock_mcp_client.from_http.assert_called_once()
        call_kwargs = mock_mcp_client.from_http.call_args[1]
        assert call_kwargs["base_url"] == "http://localhost:8080"
        assert call_kwargs["headers"] == {"Auth": "token"}

    @patch("msgflux.nn.modules.tool.MCPClient")
    @patch("msgflux.nn.modules.tool.filter_tools")
    @patch("msgflux.nn.modules.tool.F")
    def test_filter_mcp_tools(self, mock_F, mock_filter_tools, mock_mcp_client):
        """Test filtering MCP tools with include_tools."""
        from msgflux.nn.modules.tool import ToolLibrary

        mock_client_instance = MagicMock()
        mock_client_instance.connect = AsyncMock()
        mock_client_instance.list_tools = AsyncMock()
        mock_mcp_client.from_stdio.return_value = mock_client_instance

        all_tools = [
            MCPTool(name="read_file", description="Read", inputSchema={}),
            MCPTool(name="write_file", description="Write", inputSchema={}),
            MCPTool(name="delete_file", description="Delete", inputSchema={}),
        ]

        # filter_tools should be called and return filtered list
        filtered_tools = [all_tools[0], all_tools[1]]  # Only read and write
        mock_filter_tools.return_value = filtered_tools

        # Mock F.wait_for to handle different calls
        def wait_for_mock(func_or_coro, *args, **kwargs):
            # Check if it's the connect call or list_tools call
            if hasattr(func_or_coro, "__name__") and func_or_coro.__name__ == "connect":
                return None
            elif (
                hasattr(func_or_coro, "__name__")
                and func_or_coro.__name__ == "list_tools"
            ):
                return all_tools
            # Default: try to call it if it's a mock
            if callable(func_or_coro):
                return all_tools if "list_tools" in str(func_or_coro) else None
            return None

        mock_F.wait_for.side_effect = wait_for_mock

        mcp_servers = [
            {
                "name": "fs",
                "transport": "stdio",
                "command": "mcp-server-fs",
                "include_tools": ["read_file", "write_file"],
            }
        ]

        library = ToolLibrary(name="test", tools=[], mcp_servers=mcp_servers)

        # Verify filter_tools was called
        mock_filter_tools.assert_called_once()

        # Verify only filtered tools were stored (the important behavior)
        assert len(library.mcp_clients["fs"]["tools"]) == 2
        assert library.mcp_clients["fs"]["tools"] == filtered_tools

    @patch("msgflux.nn.modules.tool.MCPClient")
    @patch("msgflux.nn.modules.tool.filter_tools")
    @patch("msgflux.nn.modules.tool.F")
    def test_tool_config_storage(self, mock_F, mock_filter_tools, mock_mcp_client):
        """Test that tool_config is stored correctly."""
        from msgflux.nn.modules.tool import ToolLibrary

        mock_client_instance = MagicMock()
        mock_client_instance.connect = AsyncMock()
        mock_client_instance.list_tools = AsyncMock()
        mock_mcp_client.from_stdio.return_value = mock_client_instance

        mock_tools = [MCPTool(name="read_file", description="Read", inputSchema={})]
        mock_filter_tools.return_value = mock_tools

        mock_F.wait_for.side_effect = [None, mock_tools]

        tool_config = {"read_file": {"inject_vars": ["context"], "return_direct": True}}

        mcp_servers = [
            {
                "name": "fs",
                "transport": "stdio",
                "command": "mcp-server-fs",
                "tool_config": tool_config,
            }
        ]

        library = ToolLibrary(name="test", tools=[], mcp_servers=mcp_servers)

        # Verify tool_config was stored
        assert library.mcp_clients["fs"]["tool_config"] == tool_config

    @patch("msgflux.nn.modules.tool.MCPClient")
    @patch("msgflux.nn.modules.tool.filter_tools")
    @patch("msgflux.nn.modules.tool.F")
    def test_get_mcp_tool_names(self, mock_F, mock_filter_tools, mock_mcp_client):
        """Test getting MCP tool names with namespace."""
        from msgflux.nn.modules.tool import ToolLibrary

        mock_client_instance = MagicMock()
        mock_client_instance.connect = AsyncMock()
        mock_client_instance.list_tools = AsyncMock()
        mock_mcp_client.from_stdio.return_value = mock_client_instance

        mock_tools = [
            MCPTool(name="read_file", description="Read", inputSchema={}),
            MCPTool(name="write_file", description="Write", inputSchema={}),
        ]
        mock_filter_tools.return_value = mock_tools

        mock_F.wait_for.side_effect = [None, mock_tools]

        mcp_servers = [{"name": "fs", "transport": "stdio", "command": "mcp-server-fs"}]

        library = ToolLibrary(name="test", tools=[], mcp_servers=mcp_servers)

        mcp_tool_names = library.get_mcp_tool_names()

        assert len(mcp_tool_names) == 2
        assert "fs__read_file" in mcp_tool_names
        assert "fs__write_file" in mcp_tool_names

    @patch("msgflux.nn.modules.tool.MCPClient")
    @patch("msgflux.nn.modules.tool.filter_tools")
    @patch("msgflux.nn.modules.tool.F")
    @patch("msgflux.nn.modules.tool.convert_mcp_schema_to_tool_schema")
    def test_get_tool_json_schemas_includes_mcp(
        self, mock_convert_schema, mock_F, mock_filter_tools, mock_mcp_client
    ):
        """Test that get_tool_json_schemas includes MCP tools."""
        from msgflux.nn.modules.tool import ToolLibrary

        mock_client_instance = MagicMock()
        mock_client_instance.connect = AsyncMock()
        mock_client_instance.list_tools = AsyncMock()
        mock_mcp_client.from_stdio.return_value = mock_client_instance

        mock_tools = [MCPTool(name="read_file", description="Read", inputSchema={})]
        mock_filter_tools.return_value = mock_tools

        mock_F.wait_for.side_effect = [None, mock_tools]

        # Mock schema conversion
        mock_convert_schema.return_value = {
            "type": "function",
            "function": {"name": "fs__read_file", "description": "Read"},
        }

        mcp_servers = [{"name": "fs", "transport": "stdio", "command": "mcp-server-fs"}]

        library = ToolLibrary(name="test", tools=[], mcp_servers=mcp_servers)

        schemas = library.get_tool_json_schemas()

        # Should include MCP tool schema
        assert len(schemas) >= 1
        mock_convert_schema.assert_called()

    @patch("msgflux.nn.modules.tool.MCPClient")
    @patch("msgflux.nn.modules.tool.filter_tools")
    @patch("msgflux.nn.modules.tool.F")
    @patch("msgflux.nn.modules.tool.convert_mcp_schema_to_tool_schema")
    def test_on_demand_mcp_tools_are_hidden_until_loaded(
        self, mock_convert_schema, mock_F, mock_filter_tools, mock_mcp_client
    ):
        """Test that on-demand MCP tools are exposed only after tool_search."""
        from msgflux.nn.modules.tool import ToolLibrary

        mock_client_instance = MagicMock()
        mock_client_instance.connect = AsyncMock()
        mock_client_instance.list_tools = AsyncMock()
        mock_mcp_client.from_stdio.return_value = mock_client_instance

        mock_tools = [MCPTool(name="read_file", description="Read", inputSchema={})]
        mock_filter_tools.return_value = mock_tools
        mock_F.wait_for.side_effect = [None, mock_tools]
        mock_F.scatter_gather.side_effect = lambda prepared: [call() for call in prepared]
        mock_convert_schema.return_value = {
            "type": "function",
            "function": {"name": "fs__read_file", "description": "Read"},
        }

        library = ToolLibrary(
            name="test",
            tools=[],
            mcp_servers=[
                {
                    "name": "fs",
                    "transport": "stdio",
                    "command": "mcp-server-fs",
                    "tool_config": {"read_file": {"on_demand": True}},
                }
            ],
        )

        initial_schema_names = [
            schema["function"]["name"] for schema in library.get_tool_json_schemas()
        ]
        assert initial_schema_names == ["tool_search"]

        result = library(
            [("call_1", "tool_search", {"query": "select:fs__read_file"})]
        ).tool_calls[0].result

        loaded_schema_names = [
            schema["function"]["name"] for schema in library.get_tool_json_schemas()
        ]
        assert result["matches"] == ["fs__read_file"]
        assert result["loaded"] == ["fs__read_file"]
        assert "tool_search" in loaded_schema_names
        assert "fs__read_file" in loaded_schema_names

    def test_mcp_servers_none(self):
        """Test ToolLibrary with no MCP servers."""
        from msgflux.nn.modules.tool import ToolLibrary

        library = ToolLibrary(name="test", tools=[], mcp_servers=None)

        assert len(library.mcp_clients) == 0
        assert library.get_mcp_tool_names() == []

    def test_mcp_server_missing_name(self):
        """Test MCP server config without name raises error."""
        from msgflux.nn.modules.tool import ToolLibrary

        mcp_servers = [
            {
                "transport": "stdio",
                "command": "mcp-server-fs",
                # Missing "name" field
            }
        ]

        with pytest.raises(ValueError, match="must include 'name' field"):
            ToolLibrary(name="test", tools=[], mcp_servers=mcp_servers)

    def test_mcp_server_invalid_transport(self):
        """Test MCP server with invalid transport raises error."""
        from msgflux.nn.modules.tool import ToolLibrary

        mcp_servers = [
            {"name": "test", "transport": "invalid_transport", "command": "test"}
        ]

        with pytest.raises(ValueError, match="Unknown transport type"):
            ToolLibrary(name="test", tools=[], mcp_servers=mcp_servers)
