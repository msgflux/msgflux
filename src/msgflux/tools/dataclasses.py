from dataclasses import dataclass, field


@dataclass
class InternalToolState:
    background_tool_names: set[str] = field(default_factory=set)
    background_agent_tool_names: set[str] = field(default_factory=set)
    background_task_tool_names: set[str] = field(default_factory=set)
    agent_task_tool_names: set[str] = field(default_factory=set)
    disabled_background_task_tool_names: set[str] = field(default_factory=set)
    tool_search_disabled: bool = False

    def clear(self) -> None:
        self.background_tool_names.clear()
        self.background_agent_tool_names.clear()
        self.background_task_tool_names.clear()
        self.agent_task_tool_names.clear()
        self.disabled_background_task_tool_names.clear()
        self.tool_search_disabled = False
