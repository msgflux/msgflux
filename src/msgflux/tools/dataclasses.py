from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping


@dataclass
class ToolMetadata:
    """Normalized metadata extracted from a Python callable tool."""

    name: str
    description: str
    annotations: Dict[str, Any]
    tool_config: Dict[str, Any]
    impl: Callable
    display_name: str | None = None
    usage_guidance: str | None = None
    source_tool: Any | None = None


@dataclass
class PreparedToolExecution:
    """One resolved tool call after library-managed argument injection."""

    id: str
    name: str
    tool: Any
    config: Mapping[str, Any]
    call_params: Dict[str, Any]
    response_params: Dict[str, Any] | None
    mode: str
