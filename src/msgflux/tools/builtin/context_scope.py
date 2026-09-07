"""Built-in tools for opening and closing nested conversation scopes."""

from __future__ import annotations

from typing import Any

from msgflux.tools.config import tool_config


@tool_config(tool_kind="context_scope")
def open_context_scope(
    name: str,
    summary: str = "",
) -> dict[str, Any]:
    """Open a nested context branch and continue execution inside it."""
    return {
        "type": "context_scope_transition",
        "action": "open",
        "name": name,
        "summary": summary or None,
    }


@tool_config(tool_kind="context_scope")
def close_context_scope(
    name: str = "",
    summary: str = "",
) -> dict[str, Any]:
    """Close the active context branch and return to its parent."""
    return {
        "type": "context_scope_transition",
        "action": "close",
        "name": name,
        "summary": summary or None,
    }


__all__ = ["close_context_scope", "open_context_scope"]
