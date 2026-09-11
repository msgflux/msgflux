"""Build coherent provider-neutral history items from streaming deltas."""

from __future__ import annotations

import threading
from copy import deepcopy
from typing import Any, Mapping


class ChatStreamAccumulator:
    """Accumulate stream deltas without exposing partial items to checkpoints.

    Deltas remain available through ``ModelStreamResponse`` as they arrive. This
    class only produces the ordered, coherent items committed when the stream
    reaches a completion, failure, or interruption boundary.
    """

    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._reasoning_items: dict[str, dict[str, Any]] = {}
        self._response_messages: dict[int, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def add_text(self, delta: str, *, role: str = "assistant") -> None:
        if not delta:
            return
        with self._lock:
            if self._items and self._items[-1].get("type") == "message":
                current = self._items[-1]
                if current.get("role") == role and isinstance(
                    current.get("content"), str
                ):
                    current["content"] += delta
                    return
            self._items.append({"type": "message", "role": role, "content": delta})

    def begin_response_message(
        self,
        index: int,
        *,
        role: str = "assistant",
        phase: str | None = None,
        provider: str,
        api_mode: str,
        provider_state: Mapping[str, Any] | None = None,
    ) -> None:
        """Start a native Responses message before its text deltas arrive."""
        with self._lock:
            item = self._response_messages.get(index)
            if item is None:
                item = {
                    "type": "message",
                    "role": role,
                    "content": [],
                    "provider_state": {
                        "provider": provider,
                        "api_mode": api_mode,
                        "data": deepcopy(dict(provider_state or {})),
                    },
                }
                if phase is not None:
                    item["phase"] = phase
                self._response_messages[index] = item
                self._items.append(item)

    def add_response_text(self, index: int, delta: str) -> None:
        """Append an output-text delta while retaining its message boundary."""
        if not delta:
            return
        with self._lock:
            item = self._response_messages.get(index)
            if item is None:
                self.add_text(delta)
                return
            content = item["content"]
            if content and content[-1].get("type") == "output_text":
                content[-1]["text"] += delta
            else:
                content.append({"type": "output_text", "text": delta})

    def finish_response_message(
        self,
        index: int,
        *,
        role: str = "assistant",
        phase: str | None = None,
        provider: str,
        api_mode: str,
        provider_state: Mapping[str, Any] | None = None,
        content: Any = None,
    ) -> None:
        """Finalize native message identity without duplicating streamed text."""
        self.begin_response_message(
            index,
            role=role,
            phase=phase,
            provider=provider,
            api_mode=api_mode,
            provider_state=provider_state,
        )
        with self._lock:
            item = self._response_messages[index]
            if phase is not None:
                item["phase"] = phase
            state = item["provider_state"]["data"]
            state.update(deepcopy(dict(provider_state or {})))
            if not item["content"] and content is not None:
                item["content"] = deepcopy(content)

    def add_reasoning(
        self,
        delta: str | None = None,
        *,
        summary: str | None = None,
        provider: str | None = None,
        api_mode: str | None = None,
        codec: str | None = None,
        provider_state: Any = None,
        item_id: str | None = None,
    ) -> None:
        if delta is None and summary is None and provider_state is None:
            return
        with self._lock:
            item = self._select_reasoning_item(
                item_id=item_id,
                state_only=provider_state is not None
                and delta is None
                and summary is None,
            )
            if item_id:
                self._reasoning_items[item_id] = item
            if delta:
                item["text"] = f"{item.get('text', '')}{delta}"
            if summary:
                item["summary"] = f"{item.get('summary', '')}{summary}"
            if provider_state is not None:
                existing = item.get("provider_state")
                if (
                    isinstance(existing, dict)
                    and existing.get("provider") == provider
                    and existing.get("api_mode") == api_mode
                    and existing.get("codec") == codec
                    and isinstance(existing.get("data"), list)
                    and isinstance(provider_state, list)
                ):
                    existing["data"].extend(deepcopy(provider_state))
                else:
                    item["provider_state"] = {
                        "provider": provider,
                        **({"api_mode": api_mode} if api_mode is not None else {}),
                        **({"codec": codec} if codec is not None else {}),
                        "data": deepcopy(provider_state),
                    }

    def _select_reasoning_item(
        self,
        *,
        item_id: str | None,
        state_only: bool,
    ) -> dict[str, Any]:
        item = self._reasoning_items.get(item_id) if item_id else None
        if item is not None:
            return item
        if self._items and self._items[-1].get("type") == "reasoning":
            return self._items[-1]
        if state_only:
            item = next(
                (
                    candidate
                    for candidate in reversed(self._items)
                    if candidate.get("type") == "reasoning"
                    and "provider_state" not in candidate
                ),
                None,
            )
            if item is not None:
                return item
        item = {"type": "reasoning", "role": "assistant"}
        self._items.append(item)
        return item

    def add_tool_call_delta(
        self,
        index: int,
        *,
        call_id: str | None = None,
        name: str | None = None,
        arguments: str | None = None,
        provider: str | None = None,
        api_mode: str | None = None,
        provider_state: Any = None,
    ) -> None:
        with self._lock:
            item = self._tool_calls.get(index)
            if item is None:
                item = {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": "",
                }
                self._tool_calls[index] = item
                self._items.append(item)
            elif call_id:
                item["call_id"] = call_id
            if name:
                item["name"] = name
            if arguments:
                item["arguments"] += arguments
            if provider_state is not None:
                item["provider_state"] = {
                    "provider": provider,
                    **({"api_mode": api_mode} if api_mode is not None else {}),
                    "data": deepcopy(provider_state),
                }

    def add_item(self, item: Mapping[str, Any]) -> None:
        with self._lock:
            self._items.append(deepcopy(dict(item)))

    def snapshot(
        self,
        *,
        fallback_output: Any = None,
        fallback_reasoning: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a stable copy, filling values set directly by old providers."""
        with self._lock:
            items = deepcopy(self._items)
        if fallback_reasoning and not any(
            item.get("type") == "reasoning" for item in items
        ):
            items.insert(
                0,
                {
                    "type": "reasoning",
                    "role": "assistant",
                    "text": fallback_reasoning,
                },
            )
        if isinstance(fallback_output, str) and not any(
            item.get("type") == "message" and item.get("role") == "assistant"
            for item in items
        ):
            items.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": fallback_output,
                }
            )
        return items
