"""Built-in agent tools ready for use out of the box."""

from msgflux.tools.builtin.agent_skills import SkillSearchTool, SkillTool
from msgflux.tools.builtin.agent_tool import AgentTool
from msgflux.tools.builtin.context_scope import close_context_scope, open_context_scope
from msgflux.tools.builtin.task_tool import (
    BACKGROUND_CAPABILITY_TOOLS,
    BASE_TASK_TOOLS,
    TaskActivityTool,
    TaskInterruptTool,
    TaskListTool,
    TaskMessageTool,
    TaskOutputTool,
    TaskStatusTool,
    TaskWaitTool,
    build_background_dispatch_result,
    build_task_result,
    build_task_timeout_result,
    build_task_timing_fields,
    format_task_activity_entry,
    truncate_activity_text,
)
from msgflux.tools.builtin.tool_search import ToolSearchTool
from msgflux.tools.builtin.weather import WeatherTool
from msgflux.tools.builtin.web_fetch import WebFetchTool
from msgflux.tools.builtin.web_search import WebSearchTool

__all__ = [
    "AgentTool",
    "close_context_scope",
    "open_context_scope",
    "BACKGROUND_CAPABILITY_TOOLS",
    "BASE_TASK_TOOLS",
    "SkillSearchTool",
    "SkillTool",
    "TaskActivityTool",
    "TaskInterruptTool",
    "TaskListTool",
    "TaskMessageTool",
    "TaskOutputTool",
    "TaskStatusTool",
    "TaskWaitTool",
    "ToolSearchTool",
    "WeatherTool",
    "WebFetchTool",
    "WebSearchTool",
    "build_background_dispatch_result",
    "build_task_result",
    "build_task_timeout_result",
    "build_task_timing_fields",
    "format_task_activity_entry",
    "truncate_activity_text",
]
