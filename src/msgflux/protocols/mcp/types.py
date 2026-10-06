"""MCP type definitions."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MCPResource:
    """Represents an MCP resource."""

    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None
    annotations: Optional[Dict[str, Any]] = None


@dataclass
class MCPTool:
    """Represents an MCP tool."""

    name: str
    description: str
    inputSchema: Dict[str, Any]
    outputSchema: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None
    title: Optional[str] = None
    icons: Optional[List[Dict[str, Any]]] = None


@dataclass
class MCPPrompt:
    """Represents an MCP prompt."""

    name: str
    description: str
    arguments: Optional[List[Dict[str, Any]]] = None


@dataclass
class MCPContent:
    """Represents MCP content block."""

    type: str
    text: Optional[str] = None
    data: Optional[str] = None
    mimeType: Optional[str] = None
    uri: Optional[str] = None
    resource: Optional[Dict[str, Any]] = None
    annotations: Optional[Dict[str, Any]] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class MCPToolResult:
    """Result from tool execution."""

    content: List[MCPContent]
    isError: bool = False
    structuredContent: Any = None
    resultType: str = "complete"
    inputRequests: Optional[Dict[str, Any]] = None
    requestState: Any = None
