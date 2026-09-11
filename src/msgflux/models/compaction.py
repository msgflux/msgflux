"""Provider-neutral contracts for model context compaction."""

from __future__ import annotations

from typing import Any, Literal

import msgspec

__all__ = ["ContextTokenEstimate", "ModelCompaction"]


class ContextTokenEstimate(msgspec.Struct, frozen=True, kw_only=True):
    """Input-token count used by an automatic compaction policy."""

    input_tokens: int
    source: Literal["provider", "heuristic"]


class ModelCompaction(msgspec.Struct, frozen=True, kw_only=True):
    """A complete model-visible context view returned by a Model."""

    format: Literal["messages", "provider"]
    items: list[dict[str, Any]]
    provider: str | None = None
    api_mode: str | None = None
    model_id: str | None = None
    usage: dict[str, Any] | None = None

    def to_view(self) -> dict[str, Any]:
        view: dict[str, Any] = {
            "format": self.format,
            "items": self.items,
        }
        if self.provider is not None:
            view["provider"] = self.provider
        if self.api_mode is not None:
            view["api_mode"] = self.api_mode
        return view
