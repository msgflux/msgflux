from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.types import Hidden, ToolBucket, ToolLibraryOperator


class ToolSearchTool(ToolBucket, ToolLibraryOperator):
    """Search and activate registered on-demand tools."""

    name = "tool_search"
    capture = {"on_demand": True}
    expose_captured_names = True
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

        total = len(self.tools)
        if select is not None:
            matches = self._select(select)
            descriptions = self._describe_matches(matches) if description else []
            loaded = self._activate(matches, handle)
        else:
            matches = self._search(query or "", max_results)
            descriptions = self._describe_matches(matches) if description else []
            loaded = []

        if not self.tools:
            handle.remove(self.name)

        return {
            "query": query,
            "matches": matches,
            "loaded": loaded,
            "descriptions": descriptions,
            "total_on_demand_tools": total,
        }

    def validate_capture(self, metadata: ToolMetadata) -> None:
        if not self.captures(metadata):
            raise ValueError(
                f"Tool `{metadata.name}` does not match this bucket's capture rule."
            )

    def _search(self, query: str, max_results: int) -> List[str]:
        query_lower = query.strip().lower()
        terms = [term for term in query_lower.split() if term]
        if not terms:
            return []

        matches = []
        for tool_name, metadata in self.tools.items():
            name_parts = tool_name.lower().replace("__", " ").replace("_", " ")
            description = (metadata.description or "").lower()
            score = 0
            if query_lower == tool_name.lower():
                score += 100
            if query_lower in name_parts:
                score += 40
            for term in terms:
                if term in name_parts:
                    score += 15
                if description and term in description:
                    score += 5
            if score > 0:
                matches.append((score, tool_name))

        matches.sort(key=lambda item: (-item[0], item[1]))
        return [tool_name for _, tool_name in matches[:max_results]]

    def _select(self, requested: List[str]) -> List[str]:
        resolved = []
        normalized = {tool_name.lower(): tool_name for tool_name in self.tools}
        for tool_name in requested:
            match = normalized.get(tool_name.lower())
            if match is not None and match not in resolved:
                resolved.append(match)
        return resolved

    def _describe_matches(self, tool_names: List[str]) -> List[dict[str, Any]]:
        return [
            self._describe_metadata(self.tools[tool_name]) for tool_name in tool_names
        ]

    @staticmethod
    def _describe_metadata(metadata: ToolMetadata) -> dict[str, Any]:
        return {
            "name": metadata.name,
            "display_name": metadata.display_name or metadata.name,
            "description": metadata.description,
            "usage_guidance": metadata.usage_guidance,
            "tool_kind": metadata.tool_config.get("tool_kind", "tool"),
        }

    def _activate(self, tool_names: List[str], handle) -> List[str]:
        activated = []
        for tool_name in tool_names:
            metadata = self.remove(tool_name)
            promoted = replace(
                metadata,
                tool_config={**metadata.tool_config, "on_demand": False},
            )
            try:
                handle.add(promoted)
            except Exception:
                self.add(metadata)
                raise
            activated.append(tool_name)
        return activated

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
