"""MCP (Model Context Protocol) Client integration for msgflux library.

A lightweight implementation that supports multiple transports (stdio, HTTP/SSE).
"""

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from msgflux.protocols.mcp.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPInputRequiredError,
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

    MODERN_VERSION = "2026-07-28"
    LEGACY_VERSION = "2024-11-05"

    def __init__(
        self,
        transport: BaseTransport,
        client_info: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        *,
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
        self.protocol_version: Optional[str] = None
        self._modern = False
        self._log_level: Optional[LogLevel] = None
        self._tools_cache: Optional[List[MCPTool]] = None
        self._resources_cache: Optional[List[MCPResource]] = None
        self._prompts_cache: Optional[List[MCPPrompt]] = None
        self._cache_expiry: Dict[str, float] = {}
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
        *,
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
        *,
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
            pool_limits: Connection pool limits (max_connections,
                max_keepalive_connections)

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
        self.protocol_version = None
        self._modern = False
        self._clear_caches()

    def _clear_caches(self):
        """Clear all cached data."""
        self._tools_cache = None
        self._resources_cache = None
        self._prompts_cache = None
        self._cache_expiry.clear()

    def _cache_valid(self, key: str) -> bool:
        """Check the server's freshness hint for a local cache entry."""
        return not self._modern or time.monotonic() < self._cache_expiry.get(key, 0)

    def _set_cache_expiry(self, key: str, result: Dict[str, Any]) -> None:
        """Apply the minimum supported modern TTL to a list cache."""
        if self._modern:
            ttl_ms = result.get("ttlMs", 0)
            self._cache_expiry[key] = time.monotonic() + max(0, ttl_ms) / 1000

    async def _list_pages(self, method: str, key: str) -> tuple[list, Dict[str, Any]]:
        """Collect every page of a list endpoint."""
        items: list = []
        cursor: Optional[str] = None
        seen: set[str] = set()
        last_result: Dict[str, Any] = {}
        min_ttl: Optional[int] = None
        while True:
            response = await self._request(
                method, {"cursor": cursor} if cursor is not None else None
            )
            if "error" in response:
                raise MCPError(f"Failed to list {key}: {response['error']}")
            last_result = response.get("result", {})
            if self._modern:
                ttl = max(0, last_result.get("ttlMs", 0))
                min_ttl = ttl if min_ttl is None else min(min_ttl, ttl)
            items.extend(last_result.get(key, []))
            cursor = last_result.get("nextCursor")
            if cursor is None:
                if min_ttl is not None:
                    last_result = {**last_result, "ttlMs": min_ttl}
                return items, last_result
            if cursor in seen:
                raise MCPError(f"Repeated cursor while listing {key}: {cursor}")
            seen.add(cursor)

    async def _connect_with_retry(self):
        """Connect with exponential backoff retry logic."""
        for attempt in range(self.max_retries):
            try:
                self._connection_attempts = attempt + 1
                await self.transport.connect()
                await self._negotiate_protocol()
                self._last_error = None
                return
            except Exception as e:
                self._last_error = e
                await self.transport.disconnect()

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

    def _modern_params(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Attach the required per-request metadata without changing caller data."""
        result = dict(params or {})
        meta = dict(result.get("_meta", {}))
        meta.update(
            {
                "io.modelcontextprotocol/protocolVersion": self.MODERN_VERSION,
                "io.modelcontextprotocol/clientInfo": self.client_info,
                "io.modelcontextprotocol/clientCapabilities": {},
            }
        )
        if self._log_level is not None:
            meta["io.modelcontextprotocol/logLevel"] = self._log_level.value
        result["_meta"] = meta
        return result

    async def _request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        on_notification: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Send a request according to the negotiated protocol revision."""
        if self._modern:
            params = self._modern_params(params)
        if on_notification is not None:
            return await self.transport.send_request(
                method, params, on_notification=on_notification
            )
        return await self.transport.send_request(method, params)

    async def _negotiate_protocol(self):
        """Probe the stateless revision, falling back to the legacy handshake."""
        try:
            response = await self.transport.send_request(
                "server/discover", self._modern_params()
            )
        except MCPError:
            response = {}

        error = response.get("error", {})
        if error.get("code") == -32022:
            supported = error.get("data", {}).get("supported", [])
            if self.MODERN_VERSION not in supported:
                raise MCPError(f"No supported MCP protocol version: {supported}")
            response = await self.transport.send_request(
                "server/discover", self._modern_params()
            )

        result = response.get("result", {})
        if "supportedVersions" in result:
            if self.MODERN_VERSION not in result["supportedVersions"]:
                raise MCPError(
                    f"No supported MCP protocol version: {result['supportedVersions']}"
                )
            self._modern = True
            self.protocol_version = self.MODERN_VERSION
            self._initialized = True
            return

        await self._initialize_session()

    async def _initialize_session(self):
        """Initialize MCP session with server."""
        params = {
            "protocolVersion": self.LEGACY_VERSION,
            "capabilities": {},
            "clientInfo": self.client_info,
        }

        response = await self.transport.send_request("initialize", params)

        if "error" in response:
            raise MCPError(f"Failed to initialize: {response['error']}")

        # Extract session ID from response body and forward to transport.
        # BaseTransport.set_session_id is a no-op; only HTTP transport stores it.
        result = response.get("result", {})
        negotiated = result.get("protocolVersion", self.LEGACY_VERSION)
        if negotiated not in {
            self.LEGACY_VERSION,
            "2025-03-26",
            "2025-06-18",
            "2025-11-25",
        }:
            raise MCPError(f"Unsupported legacy MCP protocol version: {negotiated}")
        self.protocol_version = negotiated
        session_id = (
            result.get("sessionId")
            or result.get("session_id")
            or response.get("meta", {}).get("sessionId")
            or response.get("meta", {}).get("session_id")
        )
        if session_id:
            self.transport.set_session_id(session_id)

        self._initialized = True

        # Send initialized notification
        await self.transport.send_notification("notifications/initialized")

    # Resource Methods
    @Spans.ainstrument(attributes={"mcp.operation": "list_resources"})
    async def list_resources(self, *, use_cache: bool = True) -> List[MCPResource]:
        """List available resources."""
        await self._ensure_connected()

        if (
            use_cache
            and self._resources_cache is not None
            and self._cache_valid("resources")
        ):
            return self._resources_cache

        resource_data_list, cache_result = await self._list_pages(
            "resources/list", "resources"
        )

        resources = []
        for resource_data in resource_data_list:
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
        self._set_cache_expiry("resources", cache_result)
        return resources

    @Spans.ainstrument(attributes={"mcp.operation": "read_resource"})
    async def read_resource(
        self,
        uri: str,
        *,
        input_responses: Optional[Dict[str, Any]] = None,
        request_state: Any = None,
    ) -> List[MCPContent]:
        """Read content from a resource."""
        await self._ensure_connected()
        params = {"uri": uri}
        if input_responses is not None:
            params["inputResponses"] = input_responses
        if request_state is not None:
            params["requestState"] = request_state
        response = await self._request("resources/read", params)

        if "error" in response:
            raise MCPError(f"Failed to read resource {uri}: {response['error']}")
        result = response.get("result", {})
        if result.get("resultType") == "input_required":
            raise MCPInputRequiredError(
                result.get("inputRequests", {}), result.get("requestState")
            )

        contents = []
        for content_data in result.get("contents", []):
            contents.append(
                MCPContent(
                    type=content_data.get(
                        "type", "text" if "text" in content_data else "blob"
                    ),
                    text=content_data.get("text"),
                    data=content_data.get("data", content_data.get("blob")),
                    mimeType=content_data.get("mimeType"),
                    uri=content_data.get("uri"),
                    resource=content_data.get("resource"),
                    annotations=content_data.get("annotations"),
                    raw=content_data,
                )
            )

        return contents

    # Tool Methods
    @Spans.ainstrument(attributes={"mcp.operation": "list_tools"})
    async def list_tools(self, *, use_cache: bool = True) -> List[MCPTool]:
        """List available tools."""
        await self._ensure_connected()

        if use_cache and self._tools_cache is not None and self._cache_valid("tools"):
            return self._tools_cache

        tool_data_list, cache_result = await self._list_pages("tools/list", "tools")

        tools = []
        for tool_data in tool_data_list:
            tools.append(
                MCPTool(
                    name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    inputSchema=tool_data.get("inputSchema", {}),
                    outputSchema=tool_data.get("outputSchema"),
                    annotations=tool_data.get("annotations"),
                    title=tool_data.get("title"),
                    icons=tool_data.get("icons"),
                )
            )

        if self._modern and isinstance(self.transport, HTTPTransport):
            tools = self.transport.register_tool_schemas(tools)

        self._tools_cache = tools
        self._set_cache_expiry("tools", cache_result)
        return tools

    @Spans.ainstrument(attributes={"mcp.operation": "call_tool"})
    async def call_tool(  # noqa: C901
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        _progress_callback: Optional[Callable[[float, Optional[str]], None]] = None,
        *,
        input_responses: Optional[Dict[str, Any]] = None,
        request_state: Any = None,
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

        if self._modern and isinstance(self.transport, HTTPTransport):
            if name not in self.transport._tool_schemas:
                await self.list_tools(use_cache=False)
            if name not in self.transport._tool_schemas:
                raise MCPError(f"MCP tool is unavailable over HTTP: {name}")

        params = {"name": name, "arguments": arguments or {}}
        on_notification = None
        if _progress_callback is not None:
            params["_meta"] = {"progressToken": uuid.uuid4().hex}

            def on_notification(message: Dict[str, Any]) -> None:
                if message.get("method") == "notifications/progress":
                    progress = message.get("params", {})
                    _progress_callback(
                        progress.get("progress", 0), progress.get("message")
                    )

        if input_responses is not None:
            params["inputResponses"] = input_responses
        if request_state is not None:
            params["requestState"] = request_state

        response = await self._request(
            "tools/call", params, on_notification=on_notification
        )

        if "error" in response:
            error_msg = response["error"].get("message", str(response["error"]))
            # Return error as MCPToolResult instead of raising
            return MCPToolResult(
                content=[MCPContent(type="text", text=error_msg)], isError=True
            )

        result = response.get("result", {})
        if result.get("resultType") == "input_required":
            return MCPToolResult(
                content=[],
                resultType="input_required",
                inputRequests=result.get("inputRequests", {}),
                requestState=result.get("requestState"),
            )
        contents = []

        for content_data in result.get("content", []):
            contents.append(
                MCPContent(
                    type=content_data["type"],
                    text=content_data.get("text"),
                    data=content_data.get("data"),
                    mimeType=content_data.get("mimeType"),
                    uri=content_data.get("uri"),
                    resource=content_data.get("resource"),
                    annotations=content_data.get("annotations"),
                    raw=content_data,
                )
            )

        return MCPToolResult(
            content=contents,
            isError=result.get("isError", False),
            structuredContent=result.get("structuredContent"),
        )

    # Prompt Methods
    async def list_prompts(self, *, use_cache: bool = True) -> List[MCPPrompt]:
        """List available prompts."""
        await self._ensure_connected()

        if (
            use_cache
            and self._prompts_cache is not None
            and self._cache_valid("prompts")
        ):
            return self._prompts_cache

        prompt_data_list, cache_result = await self._list_pages(
            "prompts/list", "prompts"
        )

        prompts = []
        for prompt_data in prompt_data_list:
            prompts.append(
                MCPPrompt(
                    name=prompt_data["name"],
                    description=prompt_data.get("description", ""),
                    arguments=prompt_data.get("arguments"),
                )
            )

        self._prompts_cache = prompts
        self._set_cache_expiry("prompts", cache_result)
        return prompts

    async def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        input_responses: Optional[Dict[str, Any]] = None,
        request_state: Any = None,
    ) -> List[MCPContent]:
        """Get a prompt with optional arguments."""
        await self._ensure_connected()

        params = {"name": name, "arguments": arguments or {}}
        if input_responses is not None:
            params["inputResponses"] = input_responses
        if request_state is not None:
            params["requestState"] = request_state

        response = await self._request("prompts/get", params)

        if "error" in response:
            raise MCPError(f"Failed to get prompt {name}: {response['error']}")
        result = response.get("result", {})
        if result.get("resultType") == "input_required":
            raise MCPInputRequiredError(
                result.get("inputRequests", {}), result.get("requestState")
            )

        contents = []
        for message in result.get("messages", []):
            if "content" in message:
                if isinstance(message["content"], str):
                    contents.append(MCPContent(type="text", text=message["content"]))
                else:
                    blocks = message["content"]
                    if isinstance(blocks, dict):
                        blocks = [blocks]
                    for content_data in blocks:
                        contents.append(
                            MCPContent(
                                type=content_data["type"],
                                text=content_data.get("text"),
                                data=content_data.get("data"),
                                mimeType=content_data.get("mimeType"),
                                resource=content_data.get("resource"),
                                annotations=content_data.get("annotations"),
                                raw=content_data,
                            )
                        )

        return contents

    # Utility Methods
    async def ping(self) -> bool:
        """Send ping to check server connectivity."""
        try:
            response = await self._request(
                "server/discover" if self._modern else "ping"
            )
            return "result" in response
        except Exception:
            return False

    async def set_logging_level(self, level: LogLevel):
        """Set server logging level."""
        if self._modern:
            self._log_level = level
            return
        await self.transport.send_notification(
            "logging/setLevel", {"level": level.value}
        )
