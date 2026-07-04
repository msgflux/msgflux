"""Built-in agent tools ready for use out of the box."""

from msgflux.tools.builtin.agent import AgentTool
from msgflux.tools.builtin.agent_skills import ActivateSkill, SkillSearch
from msgflux.tools.builtin.tool_search import ToolSearchTool
from msgflux.tools.builtin.weather import Weather
from msgflux.tools.builtin.web_fetch import WebFetch
from msgflux.tools.builtin.web_search import WebSearch

__all__ = [
    "ActivateSkill",
    "AgentTool",
    "SkillSearch",
    "ToolSearchTool",
    "Weather",
    "WebFetch",
    "WebSearch",
]
