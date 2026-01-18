"""MCP (Model Context Protocol) Client integration for msgflux library.

A lightweight implementation that supports multiple transports (stdio, HTTP/SSE).
"""

import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from msgflux.protocols.mcp.exceptions import MCPConnectionError, MCPError
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
from msgflux.telemetry import Spans

if TYPE_CHECKING:
    from msgflux.protocols.mcp.auth.base import BaseAuth


class MCPClient:
    """Lightweight MCP client with pluggable transports.

    Features:
    - Multiple transports: stdio (subprocess), HTTP/SSE
    - Async API compatible with msgflux
    - Tool execution with structured outputs
    - Resource and prompt management
    - Progress tracking and logging
    """

    def __init__(
        self,
        transport: BaseTransport,
        client_info: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        auto_reconnect: bool = True,
    ):
        """Initialize MCP client with a transport.

        Args:
            transport: Transport implementation (HTTPTransport or StdioTransport)
            client_info: Client identification info
            max_retries: Maximum number of connection retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
            auto_reconnect: Automatically reconnect on connection failures
        """
        self.transport = transport
        self.client_info = client_info or {
            "name": "msgflux-mcp-client",
            "version": "1.0.0",
        }
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.auto_reconnect = auto_reconnect

        self._initialized = False
        self._tools_cache: Optional[List[MCPTool]] = None
        self._resources_cache: Optional[List[MCPResource]] = None
        self._prompts_cache: Optional[List[MCPPrompt]] = None
        self._connection_attempts = 0
        self._last_error: Optional[Exception] = None

    @classmethod
    def from_stdio(
        cls,
        command: str,
        args: Optional[list] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        client_info: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        auto_reconnect: bool = True,
    ):
        """Create MCP client with stdio transport.

        Args:
            command: Command to launch MCP server
            args: Command arguments
            cwd: Working directory for subprocess
            env: Environment variables
            timeout: Request timeout in seconds
            client_info: Client identification
            max_retries: Maximum connection retry attempts
            retry_delay: Initial delay between retries
            auto_reconnect: Enable automatic reconnection

        Returns:
            MCPClient instance configured for stdio
        """
        transport = StdioTransport(
            command=command, args=args, cwd=cwd, env=env, timeout=timeout
        )
        return cls(
            transport=transport,
            client_info=client_info,
            max_retries=max_retries,
            retry_delay=retry_delay,
            auto_reconnect=auto_reconnect,
        )

    @classmethod
    def from_http(
        cls,
        base_url: str,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        auth: Optional["BaseAuth"] = None,
        client_info: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        auto_reconnect: bool = True,
        pool_limits: Optional[Dict[str, int]] = None,
    ):
        """Create MCP client with HTTP transport.

        Args:
            base_url: Base URL of MCP server
            timeout: Request timeout in seconds
            headers: Additional HTTP headers
            auth: Authentication provider (BearerTokenAuth, APIKeyAuth, etc.)
            client_info: Client identification
            max_retries: Maximum connection retry attempts
            retry_delay: Initial delay between retries
            auto_reconnect: Enable automatic reconnection
            pool_limits: Connection pool limits (max_connections, max_keepalive_connections)

        Returns:
            MCPClient instance configured for HTTP

        Example:
            ```python
            from msgflux.protocols.mcp import MCPClient, BearerTokenAuth

            auth = BearerTokenAuth(token="your-jwt-token")
            client = MCPClient.from_http(
                base_url="https://api.example.com/mcp",
                auth=auth
            )
            ```
        """
        transport = HTTPTransport(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
            pool_limits=pool_limits,
            auth=auth,
        )
        return cls(
            transport=transport,
            client_info=client_info,
            max_retries=max_retries,
            retry_delay=retry_delay,
            auto_reconnect=auto_reconnect,
        )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    @Spans.ainstrument(attributes={"mcp.operation": "connect"})
    async def connect(self):
        """Establish connection to MCP server with retry logic."""
        await self._connect_with_retry()

    async def disconnect(self):
        """Close connection to MCP server."""
        await self.transport.disconnect()
        self._initialized = False
        self._clear_caches()

    def _clear_caches(self):
        """Clear all cached data."""
        self._tools_cache = None
        self._resources_cache = None
        self._prompts_cache = None

    async def _connect_with_retry(self):
        """Connect with exponential backoff retry logic."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                self._connection_attempts = attempt + 1
                await self.transport.connect()
                await self._initialize_session()
                self._last_error = None
                return
            except Exception as e:
                last_exception = e
                self._last_error = e

                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)
                else:
                    # Max retries reached
                    raise MCPConnectionError(
                        f"Failed to connect after {self.max_retries} attempts: {e}"
                    ) from e

    async def _ensure_connected(self):
        """Ensure client is connected, reconnecting if necessary."""
        if not self._initialized and self.auto_reconnect:
            await self._connect_with_retry()
        elif not self._initialized:
            raise MCPConnectionError("Client not connected. Call connect() first.")

    async def _initialize_session(self):
        """Initialize MCP session with server."""
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "logging": {},
                "sampling": {},
                "roots": {"listChanged": True},
            },
            "clientInfo": self.client_info,
        }

        response = await self.transport.send_request("initialize", params)

        if "error" in response:
            raise MCPError(f"Failed to initialize: {response['error']}")

        # Extract session ID from response if available (for HTTP transport)
        if isinstance(self.transport, HTTPTransport):
            session_id = None

            # Try to get session ID from response body
            result = response.get("result", {})
            session_id = result.get("sessionId") or result.get("session_id")

            # Try meta field as well
            if not session_id:
                meta = response.get("meta", {})
                session_id = meta.get("sessionId") or meta.get("session_id")

            if session_id:
                self.transport.set_session_id(session_id)

        self._initialized = True

        # Send initialized notification
        await self.transport.send_notification("notifications/initialized")

    # Resource Methods
    @Spans.ainstrument(attributes={"mcp.operation": "list_resources"})
    async def list_resources(self, use_cache: bool = True) -> List[MCPResource]:
        """List available resources."""
        await self._ensure_connected()

        if use_cache and self._resources_cache is not None:
            return self._resources_cache

        response = await self.transport.send_request("resources/list")

        if "error" in response:
            raise MCPError(f"Failed to list resources: {response['error']}")

        resources = []
        for resource_data in response.get("result", {}).get("resources", []):
            resources.append(
                MCPResource(
                    uri=resource_data["uri"],
                    name=resource_data["name"],
                    description=resource_data.get("description"),
                    mimeType=resource_data.get("mimeType"),
                    annotations=resource_data.get("annotations"),
                )
            )

        self._resources_cache = resources
        return resources

    @Spans.ainstrument(attributes={"mcp.operation": "read_resource"})
    async def read_resource(self, uri: str) -> List[MCPContent]:
        """Read content from a resource."""
        await self._ensure_connected()
        response = await self.transport.send_request("resources/read", {"uri": uri})

        if "error" in response:
            raise MCPError(f"Failed to read resource {uri}: {response['error']}")

        contents = []
        for content_data in response.get("result", {}).get("contents", []):
            contents.append(
                MCPContent(
                    type=content_data["type"],
                    text=content_data.get("text"),
                    data=content_data.get("data"),
                    mimeType=content_data.get("mimeType"),
                )
            )

        return contents

    # Tool Methods
    @Spans.ainstrument(attributes={"mcp.operation": "list_tools"})
    async def list_tools(self, use_cache: bool = True) -> List[MCPTool]:
        """List available tools."""
        await self._ensure_connected()

        if use_cache and self._tools_cache is not None:
            return self._tools_cache

        response = await self.transport.send_request("tools/list")

        if "error" in response:
            raise MCPError(f"Failed to list tools: {response['error']}")

        tools = []
        for tool_data in response.get("result", {}).get("tools", []):
            tools.append(
                MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    inputSchema=tool_data.get("inputSchema", {}),
                )
            )

        self._tools_cache = tools
        return tools

    @Spans.ainstrument(attributes={"mcp.operation": "call_tool"})
    async def call_tool(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
    ) -> MCPToolResult:
        """Execute a tool.

        Args:
            name: Tool name
            arguments: Tool arguments
            progress_callback: Optional callback for progress updates

        Returns:
            MCPToolResult with content and error status
        """
        await self._ensure_connected()

        start_time = time.time()
        params = {"name": name, "arguments": arguments or {}}

        response = await self.transport.send_request("tools/call", params)
        duration = time.time() - start_time

        if "error" in response:
            error_msg = response["error"].get("message", str(response["error"]))
            # Return error as MCPToolResult instead of raising
            return MCPToolResult(
                content=[MCPContent(type="text", text=error_msg)], isError=True
            )

        result = response.get("result", {})
        contents = []

        for content_data in result.get("content", []):
            contents.append(
                MCPContent(
                    type=content_data["type"],
                    text=content_data.get("text"),
                    data=content_data.get("data"),
                    mimeType=content_data.get("mimeType"),
                )
            )

        return MCPToolResult(content=contents, isError=result.get("isError", False))

    # Prompt Methods
    async def list_prompts(self, use_cache: bool = True) -> List[MCPPrompt]:
        """List available prompts."""
        if use_cache and self._prompts_cache is not None:
            return self._prompts_cache

        response = await self.transport.send_request("prompts/list")

        if "error" in response:
            raise MCPError(f"Failed to list prompts: {response['error']}")

        prompts = []
        for prompt_data in response.get("result", {}).get("prompts", []):
            prompts.append(
                MCPPrompt(
                    name=prompt_data["name"],
                    description=prompt_data["description"],
                    arguments=prompt_data.get("arguments"),
                )
            )

        self._prompts_cache = prompts
        return prompts

    async def get_prompt(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> List[MCPContent]:
        """Get a prompt with optional arguments."""
        params = {"name": name, "arguments": arguments or {}}

        response = await self.transport.send_request("prompts/get", params)

        if "error" in response:
            raise MCPError(f"Failed to get prompt {name}: {response['error']}")

        contents = []
        for message in response.get("result", {}).get("messages", []):
            if "content" in message:
                if isinstance(message["content"], str):
                    contents.append(MCPContent(type="text", text=message["content"]))
                elif isinstance(message["content"], list):
                    for content_data in message["content"]:
                        contents.append(
                            MCPContent(
                                type=content_data["type"],
                                text=content_data.get("text"),
                                data=content_data.get("data"),
                                mimeType=content_data.get("mimeType"),
                            )
                        )

        return contents

    # Utility Methods
    async def ping(self) -> bool:
        """Send ping to check server connectivity."""
        try:
            response = await self.transport.send_request("ping")
            return "result" in response
        except Exception:
            return False

    async def set_logging_level(self, level: LogLevel):
        """Set server logging level."""
        await self.transport.send_notification(
            "logging/setLevel", {"level": level.value}
        )
