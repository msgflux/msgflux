from __future__ import annotations

from typing import Any, Dict, List, Optional

from msgflux.tools.types import Hidden, ToolLibraryOperator


class ToolSearchTool(ToolLibraryOperator):
    """Search and activate registered on-demand tools."""

    name = "tool_search"
    display_name = "Tool Search"
    description = (
        "Search registered on-demand tools by keyword. Use `select:tool_a,tool_b` "
        "to activate exact tools. Set `description=true` to include tool details."
    )
    usage_guidance = (
        "Use when the current tools may not cover the task. Search first, then "
        "activate exact matches with `select:<tool_name>`."
    )
    annotations = {
        "query": str,
        "description": bool,
        "max_results": Optional[int],
        "handle": Hidden,
        "return": dict,
    }

    def __call__(
        self,
        query: str,
        *,
        description: bool = False,
        max_results: int | None = 5,
        handle,
    ) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("`query` must be a non-empty string.")
        if max_results is not None:
            if isinstance(max_results, bool) or not isinstance(max_results, int):
                raise TypeError(
                    f"`max_results` must be int or None, given `{type(max_results)}`"
                )
            if max_results <= 0:
                raise ValueError("`max_results` must be greater than 0.")

        on_demand_tool_names = handle.list_on_demand_tools()
        total = len(on_demand_tool_names)
        if total == 0:
            return self._result(
                query=query,
                matches=[],
                loaded=[],
                descriptions=[],
                total=0,
            )

        if query.lower().startswith("select:"):
            requested = [
                item.strip()
                for item in query.split(":", 1)[1].split(",")
                if item.strip()
            ]
            matches = handle.select_on_demand_tools(requested)
            loaded = handle.activate_on_demand_tools(matches)
        else:
            matches = handle.search_on_demand_tools(
                query=query,
                max_results=max_results or 5,
            )
            loaded = []

        descriptions = []
        if description:
            descriptions = [
                handle.describe_tool(tool_name)
                for tool_name in matches
            ]

        return self._result(
            query=query,
            matches=matches,
            loaded=loaded,
            descriptions=descriptions,
            total=total,
        )

    @staticmethod
    def _result(
        *,
        query: str,
        matches: List[str],
        loaded: List[str],
        descriptions: List[Dict[str, Any]],
        total: int,
    ) -> Dict[str, Any]:
        already_loaded = [
            tool_name for tool_name in matches if tool_name not in loaded
        ]
        return {
            "query": query,
            "matches": matches,
            "loaded": loaded,
            "already_loaded": already_loaded,
            "descriptions": descriptions,
            "total_on_demand_tools": total,
        }
