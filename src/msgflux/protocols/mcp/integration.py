"""Integration utilities for MCP with msgflux tool system."""

import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from msgflux.protocols.mcp.exceptions import MCPInputRequiredError
from msgflux.protocols.mcp.types import MCPTool, MCPToolResult


def convert_mcp_schema_to_tool_schema(
    mcp_tool: MCPTool, namespace: Optional[str] = None
) -> Dict[str, Any]:
    """Convert MCP tool schema to msgflux/OpenAI function calling format.

    Args:
        mcp_tool: MCP tool definition
        namespace: Optional namespace prefix for tool name

    Returns:
        Tool schema in OpenAI function calling format
    """
    tool_name = f"{namespace}__{mcp_tool.name}" if namespace else mcp_tool.name

    # MCP uses inputSchema which is already JSON Schema format.
    # Copy to avoid mutating the original MCPTool.inputSchema.
    parameters = {**mcp_tool.inputSchema}
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": mcp_tool.description,
            "parameters": parameters,
        },
    }


def filter_tools(
    tools: List[MCPTool],
    include_tools: Optional[List[str]] = None,
    exclude_tools: Optional[List[str]] = None,
) -> List[MCPTool]:
    """Filter MCP tools based on include/exclude lists.

    Args:
        tools: List of all available MCP tools
        include_tools: If provided, only include these tools
        exclude_tools: If provided, exclude these tools

    Returns:
        Filtered list of MCP tools

    Note:
        - If include_tools is set, only those tools are included
        - If exclude_tools is set, all tools except those are included
        - If both are set, include_tools takes priority
        - If neither is set, all tools are included
    """
    # Handle None or empty tools list
    if not tools:
        return []

    if include_tools:
        # Only include specified tools
        return [tool for tool in tools if tool.name in include_tools]

    if exclude_tools:
        # Include all except specified tools
        return [tool for tool in tools if tool.name not in exclude_tools]

    # No filter, return all
    return tools


def extract_tool_result_text(result: Any) -> str:
    """Extract text content from MCP tool result.

    Args:
        result: MCPToolResult or any content

    Returns:
        Extracted text content
    """
    if isinstance(result, MCPToolResult):
        if result.resultType == "input_required":
            raise MCPInputRequiredError(result.inputRequests, result.requestState)
        output = []
        for content in result.content:
            if content.type == "text" and content.text:
                output.append(content.text)
            elif content.type == "resource" and content.data:
                output.append(content.data)

        if output:
            return "\n".join(output)
        if result.structuredContent is not None:
            return json.dumps(result.structuredContent, ensure_ascii=False)
        if result.content:
            return json.dumps(
                [content.raw or asdict(content) for content in result.content],
                ensure_ascii=False,
            )
        return "Tool executed successfully"

    # Fallback for other types
    return str(result)
