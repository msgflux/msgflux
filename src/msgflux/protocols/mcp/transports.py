"""MCP transport implementations."""

import asyncio
import base64
import json
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import httpx2

from msgflux.logger import logger
from msgflux.protocols.mcp.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
)

if TYPE_CHECKING:
    from msgflux.protocols.mcp.auth.base import BaseAuth


class BaseTransport(ABC):
    """Abstract base class for MCP transports."""

    def __init__(self):
        self._request_id_counter = 0

    def _get_next_request_id(self) -> str:
        """Generate next request ID."""
        self._request_id_counter += 1
        return str(self._request_id_counter)

    def set_session_id(self, session_id: str):  # noqa: B027
        """Store a session ID received from the server.

        Transports that support session persistence should override this method.
        The default implementation is a no-op.
        """

    @abstractmethod
    async def connect(self):
        """Establish connection to MCP server."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Close connection to MCP server."""
        pass

    @abstractmethod
    async def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        on_notification: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Send a JSON-RPC request and wait for response."""
        pass

    @abstractmethod
    async def send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ):
        """Send a JSON-RPC notification (no response expected)."""
        pass


class HTTPTransport(BaseTransport):
    """HTTP/SSE transport for MCP.

    Uses Server-Sent Events for server-initiated messages.
    Supports connection pooling and authentication.
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        pool_limits: Optional[Dict[str, int]] = None,
        auth: Optional["BaseAuth"] = None,
    ):
        """Initialize HTTP transport.

        Args:
            base_url: Base URL of the MCP server.
            timeout: Request timeout in seconds.
            headers: Additional headers to include in requests.
            pool_limits: Connection pool configuration.
            auth: Authentication provider (optional).
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}
        self.pool_limits = pool_limits or {
            "max_connections": 100,
            "max_keepalive_connections": 20,
        }
        self.auth = auth
        self._http_client: Optional[httpx2.AsyncClient] = None
        self._session_id: Optional[str] = None
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
        super().__init__()

    async def connect(self):
        """Establish HTTP connection with pooling."""
        if self._http_client is not None:
            return

        # Don't generate session ID here - let the server create it during initialize
        # The session ID will be captured from the initialize response

        # Create limits with connection pooling
        limits = httpx2.Limits(
            max_connections=self.pool_limits["max_connections"],
            max_keepalive_connections=self.pool_limits["max_keepalive_connections"],
        )

        self._http_client = httpx2.AsyncClient(
            timeout=httpx2.Timeout(self.timeout), headers=self.headers, limits=limits
        )

    async def disconnect(self):
        """Close HTTP connection."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._session_id = None
        self._tool_schemas.clear()

    def set_session_id(self, session_id: str):
        """Set session ID for subsequent requests."""
        self._session_id = session_id

    @staticmethod
    def _header_value(value: str) -> str:
        """Encode a name or parameter value according to the MCP HTTP binding."""
        safe = all(32 <= ord(char) <= 126 for char in value)
        if (
            not safe
            or value != value.strip()
            or (value.startswith("=?base64?") and value.endswith("?="))
        ):
            encoded = base64.b64encode(value.encode("utf-8")).decode("ascii")
            return f"=?base64?{encoded}?="
        return value

    @staticmethod
    def _tool_header_paths(  # noqa: C901
        schema: Dict[str, Any],
    ) -> Dict[str, tuple[str, ...]]:
        """Find valid header annotations along plain object property paths."""
        headers: Dict[str, tuple[str, ...]] = {}

        def contains_header(value: Any) -> bool:
            if isinstance(value, dict):
                return "x-mcp-header" in value or any(
                    contains_header(item) for item in value.values()
                )
            if isinstance(value, list):
                return any(contains_header(item) for item in value)
            return False

        def visit(node: Any, path: tuple[str, ...]) -> None:
            if not isinstance(node, dict):
                return
            if "x-mcp-header" in node:
                header = node["x-mcp-header"]
                if (
                    not path
                    or not isinstance(header, str)
                    or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", header)
                    or not isinstance(node.get("type"), str)
                    or node["type"] not in {"string", "integer", "boolean"}
                    or header.lower() in headers
                ):
                    raise ValueError("Invalid x-mcp-header annotation")
                headers[header.lower()] = path
            for name, property_schema in node.get("properties", {}).items():
                visit(property_schema, (*path, name))
            for key, value in node.items():
                if key not in {"properties", "x-mcp-header"} and isinstance(
                    value, (dict, list)
                ):
                    if contains_header(value):
                        raise ValueError("x-mcp-header must be on a property path")

        visit(schema, ())
        return headers

    def register_tool_schemas(self, tools: list) -> list:
        """Keep HTTP-valid tools for parameter header mirroring."""
        valid = []
        self._tool_schemas.clear()
        for tool in tools:
            try:
                self._tool_header_paths(tool.inputSchema)
            except ValueError:
                logger.warning(
                    f"Ignoring MCP tool with invalid x-mcp-header: {tool.name}"
                )
                continue
            self._tool_schemas[tool.name] = tool.inputSchema
            valid.append(tool)
        return valid

    def _request_headers(  # noqa: C901
        self, method: str, params: Optional[Dict[str, Any]], headers: Dict[str, str]
    ) -> Dict[str, str]:
        """Mirror modern request metadata into required HTTP headers."""
        meta = (params or {}).get("_meta", {})
        version = meta.get("io.modelcontextprotocol/protocolVersion")
        if not version:
            return headers
        headers["MCP-Protocol-Version"] = version
        headers["Mcp-Method"] = method
        name = (params or {}).get("name") or (params or {}).get("uri")
        if (
            method in {"tools/call", "resources/read", "prompts/get"}
            and name is not None
        ):
            headers["Mcp-Name"] = self._header_value(str(name))
        if method == "tools/call" and name in self._tool_schemas:
            arguments = (params or {}).get("arguments", {})
            for header, path in self._tool_header_paths(
                self._tool_schemas[name]
            ).items():
                value = arguments
                for part in path:
                    if not isinstance(value, dict) or part not in value:
                        value = None
                        break
                    value = value[part]
                if value is None:
                    continue
                if isinstance(value, int) and not isinstance(value, bool):
                    if abs(value) > 2**53 - 1:
                        raise MCPError(
                            "MCP parameter header integer exceeds safe range"
                        )
                if isinstance(value, bool):
                    value = str(value).lower()
                headers[f"Mcp-Param-{header}"] = self._header_value(str(value))
        return headers

    @staticmethod
    def _parse_sse(
        body: str,
        request_id: str,
        on_notification: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Return the final JSON-RPC response after request-scoped events."""
        data_lines: list[str] = []
        for line in [*body.splitlines(), ""]:
            if not line:
                if data_lines:
                    message = json.loads("\n".join(data_lines))
                    if message.get("id") == request_id and (
                        "result" in message or "error" in message
                    ):
                        return message
                    if on_notification and message.get("method"):
                        on_notification(message)
                    data_lines = []
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        raise MCPError("No matching JSON-RPC response in SSE stream")

    async def _get_headers(self, *, include_session_id: bool = True) -> Dict[str, str]:
        """Get headers with authentication applied.

        Args:
            include_session_id: Whether to include session ID in headers (default True)

        Returns:
            Headers with auth credentials if auth provider is configured.
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        headers.update(self.headers)

        # Add session ID if available and requested
        if self._session_id and include_session_id:
            # Use the same header name as FastMCP server expects
            headers["mcp-session-id"] = self._session_id

        # Apply authentication if configured
        if self.auth:
            # Refresh token if needed
            await self.auth.refresh_if_needed()
            # Apply auth headers
            headers = self.auth.apply_auth(headers)

        return headers

    async def send_request(  # noqa: C901
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        on_notification: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Send HTTP POST request with JSON-RPC.

        Automatically applies authentication and refreshes tokens if needed.
        """
        if not self._http_client:
            raise MCPConnectionError("Transport not connected")

        request_data = {
            "jsonrpc": "2.0",
            "id": self._get_next_request_id(),
            "method": method,
        }

        if params is not None:
            request_data["params"] = params

        try:
            # Don't send session ID on initialize - server will create it
            modern_request = bool(
                (params or {})
                .get("_meta", {})
                .get("io.modelcontextprotocol/protocolVersion")
            )
            headers = await self._get_headers(
                include_session_id=(method != "initialize" and not modern_request)
            )
            headers = self._request_headers(method, params, headers)

            if on_notification is not None:
                return await self._stream_request(
                    request_data, headers, on_notification
                )

            response = await self._http_client.post(
                self.base_url, json=request_data, headers=headers
            )
            if isinstance(response.status_code, int) and response.status_code >= 400:
                try:
                    error_body = response.json()
                except (ValueError, json.JSONDecodeError):
                    response.raise_for_status()
                else:
                    if isinstance(error_body, dict) and "error" in error_body:
                        return error_body
                    response.raise_for_status()
            response.raise_for_status()

            # Capture session ID from response headers if present
            # Try multiple possible header names
            for header_name in [
                "mcp-session-id",
                "MCP-Session-ID",
                "X-Session-ID",
                "Session-ID",
                "X-Session-Id",
                "Session-Id",
            ]:
                session_id = response.headers.get(header_name)
                if session_id:
                    # Always update session ID if server provides one
                    # (server's ID takes precedence)
                    self._session_id = session_id
                    break

            # Check Content-Type to handle both JSON and SSE responses
            content_type = response.headers.get("content-type", "")

            if "text/event-stream" in content_type:
                # Parse SSE format: data: {...}\n\n
                return self._parse_sse(response.text, request_data["id"])
            else:
                # Regular JSON response
                return response.json()
        except httpx2.TimeoutException as e:
            raise MCPTimeoutError(f"Request to {method} timed out") from e
        except httpx2.HTTPStatusError as e:
            status_code = e.response.status_code
            error_text = e.response.text
            raise MCPError(f"HTTP error {status_code}: {error_text}") from e
        except json.JSONDecodeError as e:
            raise MCPError(f"Failed to decode JSON response: {e}") from e

    async def _stream_request(  # noqa: C901
        self,
        request_data: Dict[str, Any],
        headers: Dict[str, str],
        on_notification: Callable[[Dict[str, Any]], None],
    ) -> Dict[str, Any]:
        """Read request-scoped notifications before the final response."""
        async with self._http_client.stream(
            "POST", self.base_url, json=request_data, headers=headers
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                try:
                    error_body = response.json()
                except ValueError:
                    response.raise_for_status()
                if isinstance(error_body, dict) and "error" in error_body:
                    return error_body
            response.raise_for_status()
            if "text/event-stream" not in response.headers.get("content-type", ""):
                await response.aread()
                return response.json()
            event_lines: list[str] = []
            async for line in response.aiter_lines():
                if line:
                    if line.startswith("data:"):
                        event_lines.append(line[5:].lstrip(" "))
                    continue
                if event_lines:
                    message = json.loads("\n".join(event_lines))
                    event_lines = []
                    if message.get("id") == request_data["id"] and (
                        "result" in message or "error" in message
                    ):
                        return message
                    if message.get("method"):
                        on_notification(message)
            raise MCPError("No matching JSON-RPC response in SSE stream")

    async def send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ):
        """Send HTTP notification (fire-and-forget).

        Automatically applies authentication.
        """
        if not self._http_client:
            raise MCPConnectionError("Transport not connected")

        notification_data = {
            "jsonrpc": "2.0",
            "method": method,
        }

        if params is not None:
            notification_data["params"] = params

        try:
            headers = await self._get_headers()
            await self._http_client.post(
                self.base_url, json=notification_data, headers=headers
            )
        except Exception:  # noqa: S110
            # Notifications are fire-and-forget, log but don't raise
            pass


class StdioTransport(BaseTransport):
    """Stdio transport for MCP.

    Launches server as subprocess and communicates via stdin/stdout.
    Messages are JSON-RPC encoded, UTF-8, newline-delimited.
    """

    def __init__(
        self,
        command: str,
        args: Optional[list] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        self.command = command
        self.args = args or []
        self.cwd = cwd
        self.env = env
        self.timeout = timeout
        self._process: Optional[asyncio.subprocess.Process] = None
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._notification_callbacks: Dict[Any, Callable[[Dict[str, Any]], None]] = {}
        self._read_task: Optional[asyncio.Task] = None
        super().__init__()

    async def connect(self):
        """Launch subprocess and start reading from stdout."""
        if self._process is not None:
            return

        try:
            full_command = [self.command, *self.args]

            self._process = await asyncio.create_subprocess_exec(
                *full_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=self.env,
            )

            # Start background task to read responses
            self._read_task = asyncio.create_task(self._read_responses())

        except Exception as e:
            raise MCPConnectionError(f"Failed to start MCP server: {e}") from e

    async def disconnect(self):
        """Terminate subprocess and cleanup."""
        if self._read_task:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
            self._read_task = None

        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
            finally:
                self._process = None

        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()
        self._notification_callbacks.clear()

    async def _read_responses(self):  # noqa: C901
        """Background task to read JSON-RPC responses from stdout."""
        if not self._process or not self._process.stdout:
            return

        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break

                try:
                    response = json.loads(line.decode("utf-8").strip())

                    # Handle response to a request
                    if "id" in response and response["id"] in self._pending_requests:
                        future = self._pending_requests.pop(response["id"])
                        if not future.done():
                            future.set_result(response)

                    if response.get("method") == "notifications/progress":
                        params = response.get("params", {})
                        callback = self._notification_callbacks.get(
                            params.get("progressToken")
                        )
                        if callback:
                            callback(response)

                except json.JSONDecodeError as e:
                    # Invalid JSON, skip
                    logger.debug(f"Invalid JSON response: {e}")
                    continue
                except Exception as e:
                    # Error processing response
                    logger.debug(f"Error processing response: {e}")
                    continue

        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: S110
            pass

    async def send_request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        on_notification: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Send JSON-RPC request via stdin and wait for response."""
        if not self._process or not self._process.stdin:
            raise MCPConnectionError("Transport not connected")

        request_id = self._get_next_request_id()
        request_data = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }

        if params is not None:
            request_data["params"] = params

        # Create future for response
        future: asyncio.Future = asyncio.Future()
        self._pending_requests[request_id] = future
        progress_token = (params or {}).get("_meta", {}).get("progressToken")
        if on_notification is not None and progress_token is not None:
            self._notification_callbacks[progress_token] = on_notification

        try:
            # Send request (newline-delimited JSON)
            message = json.dumps(request_data) + "\n"
            self._process.stdin.write(message.encode("utf-8"))
            await self._process.stdin.drain()

            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=self.timeout)
            return response

        except asyncio.TimeoutError:
            self._pending_requests.pop(request_id, None)
            raise MCPTimeoutError(f"Request to {method} timed out") from None
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            raise MCPError(f"Failed to send request: {e}") from e
        finally:
            if progress_token is not None:
                self._notification_callbacks.pop(progress_token, None)

    async def send_notification(
        self, method: str, params: Optional[Dict[str, Any]] = None
    ):
        """Send JSON-RPC notification via stdin (no response expected)."""
        if not self._process or not self._process.stdin:
            raise MCPConnectionError("Transport not connected")

        notification_data = {
            "jsonrpc": "2.0",
            "method": method,
        }

        if params is not None:
            notification_data["params"] = params

        try:
            message = json.dumps(notification_data) + "\n"
            self._process.stdin.write(message.encode("utf-8"))
            await self._process.stdin.drain()
        except Exception:  # noqa: S110
            # Notifications are fire-and-forget
            pass
