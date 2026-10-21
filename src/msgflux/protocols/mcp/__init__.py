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
    MCPInputRequiredError,
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
    "APIKeyAuth",
    "BaseAuth",
    "BaseTransport",
    "BasicAuth",
    "BearerTokenAuth",
    "CustomHeaderAuth",
    "HTTPTransport",
    "LogLevel",
    "MCPClient",
    "MCPConnectionError",
    "MCPContent",
    "MCPError",
    "MCPInputRequiredError",
    "MCPPrompt",
    "MCPResource",
    "MCPTimeoutError",
    "MCPTool",
    "MCPToolError",
    "MCPToolResult",
    "OAuth2Auth",
    "StdioTransport",
    "convert_mcp_schema_to_tool_schema",
    "extract_tool_result_text",
    "filter_tools",
]
