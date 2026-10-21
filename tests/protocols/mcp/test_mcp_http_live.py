"""Integration tests against a current FastMCP Streamable HTTP server."""

import pytest


@pytest.mark.asyncio
async def test_http_modern_negotiation_and_paginated_tools(live_http_mcp_client):
    assert live_http_mcp_client.protocol_version == "2026-07-28"

    tools = await live_http_mcp_client.list_tools(use_cache=False)

    assert {tool.name for tool in tools} == {
        "add",
        "echo",
        "divide",
        "echo_region",
        "report_progress",
        "confirm_action",
    }
    assert next(tool for tool in tools if tool.name == "add").outputSchema is not None


@pytest.mark.asyncio
async def test_http_modern_tool_structured_result(live_http_mcp_client):
    result = await live_http_mcp_client.call_tool("add", {"a": 3, "b": 4})

    assert result.resultType == "complete"
    assert result.isError is False
    assert result.structuredContent == {"result": 7}
    assert result.content[0].text == "7"


@pytest.mark.asyncio
async def test_http_modern_tool_parameter_header(live_http_mcp_client):
    result = await live_http_mcp_client.call_tool(
        "echo_region", {"region": "São Paulo"}
    )

    assert result.isError is False
    assert result.structuredContent == {"result": "São Paulo"}


@pytest.mark.asyncio
async def test_http_modern_progress_stream(live_http_mcp_client):
    progress = []

    result = await live_http_mcp_client.call_tool(
        "report_progress",
        _progress_callback=lambda value, message: progress.append((value, message)),
    )

    assert result.structuredContent == {"result": "done"}
    assert progress == [(1, "half"), (2, "done")]


@pytest.mark.asyncio
async def test_http_modern_input_required_round_trip(live_http_mcp_client):
    initial = await live_http_mcp_client.call_tool("confirm_action")

    assert initial.resultType == "input_required"
    assert initial.inputRequests["confirm"]["method"] == "elicitation/create"
    assert initial.requestState

    completed = await live_http_mcp_client.call_tool(
        "confirm_action",
        input_responses={"confirm": {"action": "accept", "content": {"answer": "yes"}}},
        request_state=initial.requestState,
    )

    assert completed.resultType == "complete"
    assert completed.content[0].text == "yes"


@pytest.mark.asyncio
async def test_http_modern_resource_and_prompt(live_http_mcp_client):
    resources = await live_http_mcp_client.list_resources()
    prompts = await live_http_mcp_client.list_prompts()
    content = await live_http_mcp_client.read_resource("msgflux://hello")
    greeting = await live_http_mcp_client.get_prompt("greet", {"name": "Ada"})

    assert [resource.uri for resource in resources] == ["msgflux://hello"]
    assert [prompt.name for prompt in prompts] == ["greet"]
    assert content[0].text == "Hello from resource"
    assert greeting[0].text == "Greet Ada"
