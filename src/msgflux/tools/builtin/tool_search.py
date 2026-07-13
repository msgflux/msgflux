from __future__ import annotations

import re
from dataclasses import replace
from typing import List

from msgflux.tools.dataclasses import ToolMetadata
from msgflux.tools.types import Hidden, ToolBucket, ToolLibraryOperator


class ToolSearchTool(ToolBucket, ToolLibraryOperator):
    """Search and activate registered on-demand tools."""

    name = "tool_search"
    capture = {"on_demand": True}
    expose_captured_names = True
    display_name = "Tool Search"
    description = (
        "Find tools by terms or /regex/; an exact name loads it. "
        "Append :K to limit matches."
    )
    usage_guidance = None
    tool_config = {"handle": {"tools": ["register", "remove"]}}
    annotations = {
        "query": str,
        "handle": Hidden,
        "return": str,
    }

    def __call__(
        self,
        query: str,
        handle,
    ) -> str:
        expression, max_results, pattern = self._parse_query(query)
        exact = None if pattern is not None else self._exact_match(expression)
        if exact is not None:
            self._activate([exact], handle)
            result = f"loaded={exact}"
        else:
            matches = (
                self._regex_search(pattern, max_results)
                if pattern is not None
                else self._search(expression, max_results)
            )
            result = self._format_matches(matches)

        if not self.tools:
            handle.tools.remove(self.name)
        return result

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
            description = " ".join(
                value
                for value in (metadata.description, metadata.usage_guidance)
                if isinstance(value, str)
            ).lower()
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

    def _regex_search(self, pattern: re.Pattern[str], max_results: int) -> List[str]:
        matches = []
        for tool_name, metadata in self.tools.items():
            searchable = " ".join(
                value
                for value in (
                    tool_name,
                    metadata.description,
                    metadata.usage_guidance,
                )
                if isinstance(value, str)
            )[:512]
            if pattern.search(searchable):
                matches.append(tool_name)
        return sorted(matches)[:max_results]

    def _exact_match(self, query: str) -> str | None:
        return {name.lower(): name for name in self.tools}.get(query.lower())

    def _format_matches(self, tool_names: List[str]) -> str:
        lines = []
        for tool_name in tool_names:
            metadata = self.tools[tool_name]
            detail = metadata.usage_guidance or metadata.description
            if isinstance(detail, str) and detail.strip():
                detail = " ".join(detail.split())
                if len(detail) > 160:
                    detail = detail[:157] + "..."
                lines.append(f"{tool_name}: {detail}")
            else:
                lines.append(tool_name)
        return "\n".join(lines) or "none"

    def _activate(self, tool_names: List[str], handle) -> None:
        for tool_name in tool_names:
            metadata = self.remove(tool_name)
            promoted = replace(
                metadata,
                tool_config={**metadata.tool_config, "on_demand": False},
            )
            try:
                handle.tools.register(promoted)
            except Exception:
                self.add(metadata)
                raise

    @staticmethod
    def _parse_query(query: str) -> tuple[str, int, re.Pattern[str] | None]:
        if not isinstance(query, str):
            raise TypeError(f"`query` must be str, given `{type(query)}`")
        query = query.strip()
        if not query:
            raise ValueError("`query` must be a non-empty string.")

        max_results = 5
        limit_match = re.search(r":(\d+)$", query)
        if limit_match is not None:
            max_results = int(limit_match.group(1))
            if not 1 <= max_results <= 20:
                raise ValueError("The `:K` result limit must be between 1 and 20.")
            query = query[: limit_match.start()].rstrip()
            if not query:
                raise ValueError("`query` must include text before the `:K` limit.")

        pattern = None
        if query.startswith("/") or query.endswith("/"):
            if len(query) < 3 or not (query.startswith("/") and query.endswith("/")):
                raise ValueError("Regex queries must use the `/pattern/` form.")
            expression = query[1:-1]
            if len(expression) > 128:
                raise ValueError("Regex queries cannot exceed 128 characters.")
            if (
                "(?" in expression
                or re.search(r"\\[1-9]", expression)
                or re.search(r"\)[+*{]", expression)
            ):
                raise ValueError(
                    "Regex queries do not support extensions, backreferences, "
                    "or quantified groups."
                )
            try:
                pattern = re.compile(expression, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"Invalid regex query: {exc}.") from exc
            return expression, max_results, pattern
        return query, max_results, None
