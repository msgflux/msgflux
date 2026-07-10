from dataclasses import dataclass, field
from typing import Any, Callable, Dict


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
class InternalToolState:
    background_tool_names: set[str] = field(default_factory=set)
    background_agent_tool_names: set[str] = field(default_factory=set)
    background_task_tool_names: set[str] = field(default_factory=set)
    agent_task_tool_names: set[str] = field(default_factory=set)
    disabled_background_task_tool_names: set[str] = field(default_factory=set)

    def clear(self) -> None:
        self.background_tool_names.clear()
        self.background_agent_tool_names.clear()
        self.background_task_tool_names.clear()
        self.agent_task_tool_names.clear()
        self.disabled_background_task_tool_names.clear()
