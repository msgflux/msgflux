"""Minimal authenticated HTTP MCP server for local and Docker validation."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

HOST = os.getenv("MCP_AUTH_HOST", "127.0.0.1")
PORT = int(os.getenv("MCP_AUTH_PORT", "8765"))
TOKEN = os.getenv("MCP_AUTH_TOKEN", "test-token")


def _response_for(method: str, params: dict[str, Any] | None) -> dict[str, Any]:
    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "msgflux-auth-test-server", "version": "1.0.0"},
            "sessionId": "auth-session",
        }

    if method == "tools/list":
        return {
            "tools": [
                {
                    "name": "whoami",
                    "description": "Return authenticated caller details.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                }
            ]
        }

    if method == "tools/call":
        arguments = (params or {}).get("arguments") or {}
        name = arguments.get("name", "unknown")
        return {
            "content": [{"type": "text", "text": f"authenticated:{name}"}],
            "isError": False,
        }

    if method == "ping":
        return {}

    return {"content": [{"type": "text", "text": f"unknown method: {method}"}]}


class AuthMCPHandler(BaseHTTPRequestHandler):
    server_version = "MsgFluxAuthMCP/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:
        if self.path != "/mcp":
            self.send_error(404)
            return

        expected_token = getattr(self.server, "auth_token", TOKEN)
        if self.headers.get("Authorization") != f"Bearer {expected_token}":
            self._send_json({"error": "unauthorized"}, status=401)
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        request_id = payload.get("id")

        if request_id is None:
            self._send_json({}, status=202)
            return

        method = payload.get("method")
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": _response_for(method, payload.get("params")),
        }
        self._send_json(response, headers={"mcp-session-id": "auth-session"})

    def _send_json(
        self,
        payload: dict[str, Any],
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AuthMCPHandler)
    print(f"Authenticated MCP test server listening on http://{HOST}:{PORT}/mcp")
    server.serve_forever()


if __name__ == "__main__":
    main()
