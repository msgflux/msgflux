from __future__ import annotations

from typing import Any, Dict, List, Optional

from msgflux.tools.types import Hidden, ToolLibraryOperator


class ToolSearchTool(ToolLibraryOperator):
    """Search and activate registered on-demand tools."""

    name = "tool_search"
    tool_kind = "tool_search"
    display_name = "Tool Search"
    description = """Find on-demand tools. `query` lists; `select` activates.

    Args:
        query: Keywords used to find tools.
        select: Exact tool names to activate.
    """
    usage_guidance = (
        "Search first; activate an exact match with `select` before calling it."
    )
    annotations = {
        "query": Optional[str],
        "select": Optional[List[str]],
        "description": bool,
        "max_results": int,
        "handle": Hidden,
        "return": dict,
    }

    def __call__(
        self,
        query: str | None = None,
        *,
        select: List[str] | None = None,
        description: bool = False,
        max_results: int = 5,
        handle,
    ) -> Dict[str, Any]:
        query, select = self._normalize_selection(query, select)
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise TypeError(f"`max_results` must be int, given `{type(max_results)}")
        if max_results <= 0:
            raise ValueError("`max_results` must be greater than 0.")

        return handle.search_tools(
            query=query,
            select=select,
            include_descriptions=description,
            max_results=max_results,
        )

    @staticmethod
    def _normalize_selection(
        query: str | None,
        select: List[str] | None,
    ) -> tuple[str | None, List[str] | None]:
        if query is not None and not isinstance(query, str):
            raise TypeError(f"`query` must be str or None, given `{type(query)}`")
        query = query.strip() if query is not None else None
        if select is not None:
            if not isinstance(select, list) or not all(
                isinstance(tool_name, str) for tool_name in select
            ):
                raise TypeError("`select` must be a list of strings or None.")
            if query:
                raise ValueError("`query` and `select` cannot be used together.")
            select = [tool_name.strip() for tool_name in select if tool_name.strip()]
            if not select:
                raise ValueError("`select` must include at least one tool name.")
        elif query and query.lower().startswith("select:"):
            select = [
                item.strip()
                for item in query.split(":", 1)[1].split(",")
                if item.strip()
            ]
            if not select:
                raise ValueError("`select` must include at least one tool name.")
        elif not query:
            raise ValueError(
                "`query` must be a non-empty string when `select` is absent."
            )
        return query, select
