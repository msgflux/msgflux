"""Adapt Responses tool discovery items into compact observation events."""

from __future__ import annotations

import json
from typing import Any, Mapping
from uuid import uuid4

from msgflux.models.response import LMStreamEvent


def _tool_names(tools: list[Mapping[str, Any]], prefix: str = "") -> list[str]:
    names = []
    for tool in tools:
        if tool.get("type") == "namespace":
            names.extend(_tool_names(tool.get("tools", []), f"{prefix}{tool['name']}."))
        else:
            name = tool.get("name") or tool.get("function", {}).get("name")
            if isinstance(name, str):
                names.append(f"{prefix}{name}")
    return list(dict.fromkeys(names))


class ToolSearchEvents:
    """Track one response's native discovery without executing its tools."""

    def __init__(self, provider: str, api_mode: str) -> None:
        self.provider = provider
        self.api_mode = api_mode
        self.calls: dict[str, dict[str, Any]] = {}
        self.finished: set[str] = set()
        self.item_calls: dict[int, str] = {}
        self.outputs: set[int] = set()
        self.references: dict[str, str] = {}

    def observe(
        self, item: Mapping[str, Any], *, index: int, done: bool = False
    ) -> list[LMStreamEvent]:
        if item.get("type") == "tool_search_call":
            return self._observe_call(item, index=index, done=done)
        if item.get("type") == "tool_search_output" and done:
            return self._observe_output(item, index=index)
        return []

    def _observe_call(
        self, item: Mapping[str, Any], *, index: int, done: bool
    ) -> list[LMStreamEvent]:
        call_id = self.item_calls.setdefault(
            index, item.get("call_id") or item.get("id") or f"search_{uuid4().hex}"
        )
        for reference in (item.get("call_id"), item.get("id")):
            if isinstance(reference, str):
                self.references[reference] = call_id
        if call_id in self.finished:
            return []
        arguments = item.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"raw": arguments}
        data = {
            "tool_name": "tool_search",
            "tool_call_id": call_id,
            "arguments": arguments,
            "execution": "client" if item.get("execution") == "client" else "provider",
            "provider": self.provider,
            "api_mode": self.api_mode,
        }
        previous = self.calls.get(call_id)
        self.calls[call_id] = data
        events = []
        if previous is None:
            events.append(LMStreamEvent(type="tool.start", data=data))
        elif previous != data:
            events.append(LMStreamEvent(type="tool.update", data=data))
        if done and item.get("status") in {"failed", "incomplete", "cancelled"}:
            events.append(self._end(call_id, error=item.get("error") or item["status"]))
        return events

    def _observe_output(
        self, item: Mapping[str, Any], *, index: int
    ) -> list[LMStreamEvent]:
        if index in self.outputs:
            return []
        self.outputs.add(index)
        call_id = item.get("tool_search_call_id") or item.get("call_id")
        call_id = self.references.get(call_id, call_id)
        if call_id is None:
            # Hosted results have null call_id. Pair them in provider output order.
            call_id = next(
                (key for key in self.calls if key not in self.finished), None
            )
        if call_id not in self.calls or call_id in self.finished:
            return []
        error = item.get("error")
        if item.get("status") in {"failed", "incomplete", "cancelled"}:
            error = error or item["status"]
        names = [] if error else _tool_names(item.get("tools", []))
        events = []
        if names:
            events.append(
                LMStreamEvent(
                    type="tools.updated",
                    data={**self.calls[call_id], "loaded_tools": names},
                )
            )
        events.append(self._end(call_id, result={"loaded_tools": names}, error=error))
        return events

    def _end(
        self, call_id: str, *, result: Any = None, error: Any = None
    ) -> LMStreamEvent:
        self.finished.add(call_id)
        return LMStreamEvent(
            type="tool.end",
            data={**self.calls[call_id], "result": result, "error": error},
        )

    def close(self, error: str) -> list[LMStreamEvent]:
        """Settle searches lacking a result when the response finishes."""
        return [
            self._end(call_id, error=error)
            for call_id in self.calls
            if call_id not in self.finished
        ]
