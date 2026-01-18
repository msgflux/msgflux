"""MCP (Model Context Protocol) integration for msgflux."""

from msgflux.protocols.mcp.auth import (
    APIKeyAuth,
    BaseAuth,
    BasicAuth,
    BearerTokenAuth,
    CustomHeaderAuth,
    OAuth2Auth,
)
from msgflux.protocols.mcp.client import MCPClient
from msgflux.protocols.mcp.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
    MCPToolError,
)
from msgflux.protocols.mcp.integration import (
    convert_mcp_schema_to_tool_schema,
    extract_tool_result_text,
    filter_tools,
)
from msgflux.protocols.mcp.loglevels import LogLevel
from msgflux.protocols.mcp.transports import (
    BaseTransport,
    HTTPTransport,
    StdioTransport,
)
from msgflux.protocols.mcp.types import (
    MCPContent,
    MCPPrompt,
    MCPResource,
    MCPTool,
    MCPToolResult,
)

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
    # Authentication
    "BaseAuth",
    "BearerTokenAuth",
    "APIKeyAuth",
    "BasicAuth",
    "OAuth2Auth",
    "CustomHeaderAuth",
    # Utilities
    "LogLevel",
    "convert_mcp_schema_to_tool_schema",
    "filter_tools",
    "extract_tool_result_text",
]
