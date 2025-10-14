"""MCP (Model Context Protocol) integration for msgflux."""

from .client import MCPClient
from .exceptions import MCPConnectionError, MCPError, MCPTimeoutError, MCPToolError
from .integration import (
    convert_mcp_schema_to_tool_schema,
    extract_tool_result_text,
    filter_tools,
)
from .loglevels import LogLevel
from .transports import BaseTransport, HTTPTransport, StdioTransport
from .types import MCPContent, MCPPrompt, MCPResource, MCPTool, MCPToolResult

__all__ = [
    # Client
    "MCPClient",
    # Transports
    "BaseTransport",
    "HTTPTransport",
    "StdioTransport",
    # Types
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPContent",
    "MCPToolResult",
    # Exceptions
    "MCPError",
    "MCPConnectionError",
    "MCPTimeoutError",
    "MCPToolError",
    # Utilities
    "LogLevel",
    "convert_mcp_schema_to_tool_schema",
    "filter_tools",
    "extract_tool_result_text",
]
