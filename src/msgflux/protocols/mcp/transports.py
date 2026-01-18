"""MCP transport implementations."""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

try:
    import httpx
except ImportError:
    httpx = None

from msgflux.protocols.mcp.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
)

if TYPE_CHECKING:
    from msgflux.protocols.mcp.auth.base import BaseAuth


class BaseTransport(ABC):
    """Abstract base class for MCP transports."""

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
        self, method: str, params: Optional[Dict[str, Any]] = None
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
    Requires httpx to be installed.
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
        if httpx is None:
            raise ImportError(
                "httpx is required for HTTP transport. "
                "Install it with: `pip install msgflux[httpx]`"
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.headers = headers or {}
        self.pool_limits = pool_limits or {
            "max_connections": 100,
            "max_keepalive_connections": 20,
        }
        self.auth = auth
        self._http_client: Optional[httpx.AsyncClient] = None
        self._request_id_counter = 0
        self._session_id: Optional[str] = None

    def _get_next_request_id(self) -> str:
        """Generate next request ID."""
        self._request_id_counter += 1
        return str(self._request_id_counter)

    async def connect(self):
        """Establish HTTP connection with pooling."""
        if self._http_client is not None:
            return

        # Don't generate session ID here - let the server create it during initialize
        # The session ID will be captured from the initialize response

        # Create limits with connection pooling
        limits = httpx.Limits(
            max_connections=self.pool_limits["max_connections"],
            max_keepalive_connections=self.pool_limits["max_keepalive_connections"],
        )

        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout), headers=self.headers, limits=limits
        )

    async def disconnect(self):
        """Close HTTP connection."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def set_session_id(self, session_id: str):
        """Set session ID for subsequent requests."""
        self._session_id = session_id

    async def _get_headers(self, include_session_id: bool = True) -> Dict[str, str]:
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

    async def send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
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
            headers = await self._get_headers(
                include_session_id=(method != "initialize")
            )

            response = await self._http_client.post(
                self.base_url, json=request_data, headers=headers
            )
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
                    # Always update session ID if server provides one (server's ID takes precedence)
                    self._session_id = session_id
                    break

            # Check Content-Type to handle both JSON and SSE responses
            content_type = response.headers.get("content-type", "")

            if "text/event-stream" in content_type:
                # Parse SSE format: data: {...}\n\n
                text = response.text
                for line in text.split("\n"):
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        return json.loads(data_str)
                raise MCPError("No data found in SSE response")
            else:
                # Regular JSON response
                return response.json()
        except httpx.TimeoutException:
            raise MCPTimeoutError(f"Request to {method} timed out")
        except httpx.HTTPStatusError as e:
            raise MCPError(f"HTTP error {e.response.status_code}: {e.response.text}")
        except json.JSONDecodeError as e:
            raise MCPError(f"Failed to decode JSON response: {e}")

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
        except Exception:
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
        self._request_id_counter = 0
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._read_task: Optional[asyncio.Task] = None

    def _get_next_request_id(self) -> str:
        """Generate next request ID."""
        self._request_id_counter += 1
        return str(self._request_id_counter)

    async def connect(self):
        """Launch subprocess and start reading from stdout."""
        if self._process is not None:
            return

        try:
            full_command = [self.command] + self.args

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
            raise MCPConnectionError(f"Failed to start MCP server: {e}")

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

    async def _read_responses(self):
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

                    # Handle notifications from server (ignore for now)
                    # Could be extended to handle server notifications

                except json.JSONDecodeError:
                    # Invalid JSON, skip
                    continue
                except Exception:
                    # Error processing response
                    continue

        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    async def send_request(
        self, method: str, params: Optional[Dict[str, Any]] = None
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
            raise MCPTimeoutError(f"Request to {method} timed out")
        except Exception as e:
            self._pending_requests.pop(request_id, None)
            raise MCPError(f"Failed to send request: {e}")

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
        except Exception:
            # Notifications are fire-and-forget
            pass
