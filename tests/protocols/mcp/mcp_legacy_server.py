# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=3.4.7,<4"]
# ///
"""FastMCP 3 fixture for legacy protocol compatibility."""

import os

from fastmcp import FastMCP

mcp = FastMCP("msgflux-legacy-test-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


if __name__ == "__main__":
    if os.environ.get("MSGFLUX_TEST_MCP_HTTP_PORT"):
        mcp.run(
            transport="http",
            host="127.0.0.1",
            port=int(os.environ["MSGFLUX_TEST_MCP_HTTP_PORT"]),
            show_banner=False,
        )
    else:
        mcp.run(show_banner=False)
