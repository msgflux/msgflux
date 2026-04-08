# /// script
# requires-python = ">=3.10"
# dependencies = ["fastmcp"]
# ///
"""Minimal FastMCP server used as a live test fixture.

Run directly with:
    uv run tests/protocols/mcp/mcp_simple_server.py
"""

from fastmcp import FastMCP

mcp = FastMCP("msgflux-test-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def echo(message: str) -> str:
    """Echo a message back unchanged."""
    return message


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


if __name__ == "__main__":
    mcp.run()
