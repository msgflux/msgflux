"""Built-in agent tools ready for use out of the box."""

from msgflux.tools.builtin.agent import AgentTool
from msgflux.tools.builtin.agent_skills import (
    ActivateSkillTool,
    SkillSearchTool,
)
from msgflux.tools.builtin.tool_search import ToolSearchTool
from msgflux.tools.builtin.weather import WeatherTool
from msgflux.tools.builtin.web_fetch import WebFetchTool
from msgflux.tools.builtin.web_search import WebSearchTool

__all__ = [
    "ActivateSkillTool",
    "AgentTool",
    "SkillSearchTool",
    "ToolSearchTool",
    "WeatherTool",
    "WebFetchTool",
    "WebSearchTool",
]
