"""Opt-in usage guidance helpers for built-in tools."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from msgflux.core.dotdict import dotdict

BUILTIN_TOOL_USAGE_GUIDANCE: dict[str, str] = {
    "web_fetch": (
        "Use when the user provides a URL or asks about a specific web page. "
        "Fetch the page before answering questions about its contents."
    ),
    "web_search": (
        "Use when the user asks for current, recent, external, or factual "
        "information that may not be available in the model context. Prefer "
        "specific search queries."
    ),
}


def _tool_name(tool: Any) -> str | None:
    return getattr(tool, "name", None) or getattr(tool, "__name__", None)


def _tool_config(tool: Any) -> dotdict:
    config = getattr(tool, "tool_config", None)
    if config is None:
        config = dotdict()
    elif not isinstance(config, dotdict):
        config = dotdict(config)
    return config


def apply_tool_guidance(
    tools: Iterable[Any],
    guidance: dict[str, str] | None = None,
) -> list[Any]:
    """Apply builtin usage guidance to tools that do not define it.

    The function mutates each tool/class/function by setting ``tool_config``.
    Existing explicit ``usage_guidance`` values are preserved.
    """
    guidance = BUILTIN_TOOL_USAGE_GUIDANCE if guidance is None else guidance
    configured_tools = []

    for tool in tools:
        name = _tool_name(tool)
        config = _tool_config(tool)

        if name in guidance and not config.get("usage_guidance"):
            config["usage_guidance"] = guidance[name]

        tool.tool_config = config
        configured_tools.append(tool)

    return configured_tools
