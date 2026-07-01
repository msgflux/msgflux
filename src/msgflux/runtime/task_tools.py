from __future__ import annotations

from typing import Any

from msgflux.runtime.tools.task import (
    AGENT_TASK_TOOLS,
    BASE_TASK_TOOLS,
    TaskRuntimeContext,
    TaskRuntimeTool,
)


class TaskRuntimeTools(TaskRuntimeContext):
    """Registers task-control tool objects for background execution."""

    def __init__(self, library_handle: Any):
        super().__init__(library_handle)
        self._base_enabled = False
        self._agent_enabled = False

    def reset(self) -> None:
        self._base_enabled = False
        self._agent_enabled = False

    def _add_runtime_tool(self, tool: type[TaskRuntimeTool]) -> None:
        instance = tool(self)
        self.library_handle.add_runtime_tool(instance.to_metadata())

    def ensure_base_tools(self) -> None:
        if self._base_enabled:
            return
        self._base_enabled = True
        for tool in BASE_TASK_TOOLS:
            self._add_runtime_tool(tool)

    def ensure_agent_tools(self) -> None:
        if self._agent_enabled:
            return
        self._agent_enabled = True
        for tool in AGENT_TASK_TOOLS:
            self._add_runtime_tool(tool)
