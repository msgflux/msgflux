"""Manual authenticated MCP HTTP client check.

Run against the Docker/server helper:
    uv run python tests/protocols/mcp/check_auth_http_client.py
"""

import asyncio
import os

from msgflux.protocols.mcp import BearerTokenAuth, MCPClient


async def main() -> None:
    base_url = os.getenv("MCP_AUTH_URL", "http://127.0.0.1:8765/mcp")
    token = os.getenv("MCP_AUTH_TOKEN", "test-token")

    async with MCPClient.from_http(
        base_url=base_url,
        auth=BearerTokenAuth(token),
    ) as client:
        tools = await client.list_tools()
        result = await client.call_tool("whoami", {"name": "docker"})

    print(f"tools={[tool.name for tool in tools]}")
    print(result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
