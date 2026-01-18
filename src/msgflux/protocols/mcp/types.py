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


@dataclass
class MCPToolResult:
    """Result from tool execution."""

    content: List[MCPContent]
    isError: bool = False
