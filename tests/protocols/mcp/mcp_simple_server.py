# /// script
# requires-python = ">=3.11"
# dependencies = ["fastmcp>=4.0.10,<5"]
# ///
"""Minimal FastMCP server used as a live test fixture.

Run directly with:
    uv run tests/protocols/mcp/mcp_simple_server.py
"""

import asyncio
import os
from typing import Annotated

from fastmcp import Context, FastMCP
from mcp.types import ElicitRequest, ElicitRequestFormParams, InputRequiredResult
from pydantic import Field

mcp = FastMCP(
    "msgflux-test-server",
    list_page_size=int(os.environ.get("MSGFLUX_TEST_MCP_PAGE_SIZE", "0")) or None,
)


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


@mcp.tool()
def echo_region(
    region: Annotated[str, Field(json_schema_extra={"x-mcp-header": "Region"})],
) -> str:
    """Echo a region mirrored into an MCP request header."""
    return region


@mcp.tool()
async def report_progress(ctx: Context) -> str:
    """Emit progress before completing a request."""
    await ctx.report_progress(1, total=2, message="half")
    await asyncio.sleep(0.05)
    await ctx.report_progress(2, total=2, message="done")
    return "done"


@mcp.tool()
def confirm_action(ctx: Context):
    """Require an explicit answer before returning a result."""
    if ctx.input_responses is None:
        return InputRequiredResult(
            input_requests={
                "confirm": ElicitRequest(
                    params=ElicitRequestFormParams(
                        message="Continue?",
                        requested_schema={
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                        },
                    )
                )
            },
            request_state="confirm-action",
        )
    return ctx.input_responses["confirm"].content["answer"]


@mcp.resource("msgflux://hello")
def hello_resource() -> str:
    """Return a sample resource."""
    return "Hello from resource"


@mcp.prompt()
def greet(name: str) -> str:
    """Build a short greeting prompt."""
    return f"Greet {name}"


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
