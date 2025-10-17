"""Integration utilities for MCP with msgflux tool system."""

from typing import Any, Dict, List, Optional

from msgflux.protocols.mcp.types import MCPTool


def convert_mcp_schema_to_tool_schema(
    mcp_tool: MCPTool,
    namespace: Optional[str] = None
) -> Dict[str, Any]:
    """Convert MCP tool schema to msgflux/OpenAI function calling format.

    Args:
        mcp_tool: MCP tool definition
        namespace: Optional namespace prefix for tool name

    Returns:
        Tool schema in OpenAI function calling format
    """
    tool_name = f"{namespace}__{mcp_tool.name}" if namespace else mcp_tool.name

    # MCP uses inputSchema which is already JSON Schema format
    # We need to convert to OpenAI function calling format
    schema = {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema
        }
    }

    # Ensure parameters has required structure
    if "type" not in schema["function"]["parameters"]:
        schema["function"]["parameters"]["type"] = "object"

    if "properties" not in schema["function"]["parameters"]:
        schema["function"]["parameters"]["properties"] = {}

    return schema


def filter_tools(
    tools: List[MCPTool],
    include_tools: Optional[List[str]] = None,
    exclude_tools: Optional[List[str]] = None
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
    from .types import MCPToolResult, MCPContent

    if isinstance(result, MCPToolResult):
        output = []
        for content in result.content:
            if content.type == "text" and content.text:
                output.append(content.text)
            elif content.type == "resource" and content.data:
                output.append(content.data)

        return "\n".join(output) if output else "Tool executed successfully"

    # Fallback for other types
    return str(result)
