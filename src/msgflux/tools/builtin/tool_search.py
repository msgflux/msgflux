from __future__ import annotations

from typing import Any, Dict, List, Optional

from msgflux.tools.types import (
    Hidden,
    ToolBucket,
    ToolBucketEntry,
    ToolLibraryOperator,
)


class ToolSearchTool(ToolBucket, ToolLibraryOperator):
    """Search and load deferred tools for the current thread."""

    name = "tool_search"
    catalog_role = "search"
    tool_config = {"runtime_inputs": ("handle", "messages")}
    capture = {"defer_loading": True}
    expose_captured_names = True
    display_name = "Tool Search"
    description = """Find deferred tools. `query` lists; `select` loads.

    Args:
        query: Keywords used to find tools.
        select: Exact tool names to load for the current thread.
    """
    usage_guidance = (
        "Search first; activate an exact match with `select` before calling it."
    )
    annotations = {
        "query": Optional[str],
        "select": Optional[List[str]],
        "description": bool,
        "max_results": int,
        "messages": Hidden,
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
        messages=None,
        handle,
    ) -> Dict[str, Any]:
        query, select = self._normalize_selection(query, select)
        if isinstance(max_results, bool) or not isinstance(max_results, int):
            raise TypeError(f"`max_results` must be int, given `{type(max_results)}")
        if max_results <= 0:
            raise ValueError("`max_results` must be greater than 0.")

        entries = handle.list_entries()
        total = len(entries)
        if select is not None:
            matches = self._select(select, entries)
            descriptions = (
                self._describe_matches(matches, entries) if description else []
            )
            loaded = handle.load_tools(messages, matches)
        else:
            matches = self._search(query or "", max_results, entries)
            descriptions = (
                self._describe_matches(matches, entries) if description else []
            )
            loaded = []

        return {
            "query": query,
            "matches": matches,
            "loaded": loaded,
            "descriptions": descriptions,
            "total_deferred_tools": total,
        }

    def validate_capture(self, definition: Any) -> None:
        if not self.captures(definition):
            raise ValueError(
                f"Tool `{definition.name}` does not match this bucket's capture rule."
            )

    @staticmethod
    def _search(
        query: str,
        max_results: int,
        entries: tuple[ToolBucketEntry, ...],
    ) -> List[str]:
        query_lower = query.strip().lower()
        terms = [term for term in query_lower.split() if term]
        if not terms:
            return []

        matches = []
        for entry in entries:
            tool_name = entry.name
            name_parts = tool_name.lower().replace("__", " ").replace("_", " ")
            description = (entry.description or "").lower()
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

    @staticmethod
    def _select(
        requested: List[str],
        entries: tuple[ToolBucketEntry, ...],
    ) -> List[str]:
        resolved = []
        normalized = {entry.name.lower(): entry.name for entry in entries}
        for tool_name in requested:
            match = normalized.get(tool_name.lower())
            if match is not None and match not in resolved:
                resolved.append(match)
        return resolved

    @classmethod
    def _describe_matches(
        cls,
        tool_names: List[str],
        entries: tuple[ToolBucketEntry, ...],
    ) -> List[dict[str, Any]]:
        entries_by_name = {entry.name: entry for entry in entries}
        return [
            cls._describe_entry(entries_by_name[tool_name]) for tool_name in tool_names
        ]

    @staticmethod
    def _describe_entry(entry: ToolBucketEntry) -> dict[str, Any]:
        return {
            "name": entry.name,
            "display_name": entry.display_name or entry.name,
            "description": entry.description,
            "usage_guidance": entry.usage_guidance,
            "tool_kind": entry.kind,
        }

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
