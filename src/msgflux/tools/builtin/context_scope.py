"""Built-in tools for opening and closing nested conversation scopes."""

from __future__ import annotations

from typing import Any

from msgflux.tools.config import tool_config


@tool_config(return_direct=True)
def open_context_scope(
    name: str,
    summary: str | None = None,
) -> dict[str, Any]:
    """Open a nested context branch and continue execution inside it."""
    return {
        "type": "context_scope_transition",
        "action": "open",
        "name": name,
        "summary": summary,
    }


@tool_config(return_direct=True)
def close_context_scope(
    name: str | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """Close the active context branch and return to its parent."""
    return {
        "type": "context_scope_transition",
        "action": "close",
        "name": name,
        "summary": summary,
    }


__all__ = ["close_context_scope", "open_context_scope"]
