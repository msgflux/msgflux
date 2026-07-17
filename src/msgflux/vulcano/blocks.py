from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping
from uuid import uuid4

__all__ = [
    "BlockKind",
    "BlockStatus",
    "ContentBlock",
    "new_block_id",
    "new_tool_call_id",
]


class BlockKind:
    """Built-in conversation block kinds understood by Vulcano clients."""

    TEXT = "text"
    REASONING = "reasoning"
    TOOL = "tool"
    DIFF = "diff"
    ARTIFACT = "artifact"
    ERROR = "error"


class BlockStatus:
    """Common lifecycle states for streamed conversation blocks."""

    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


def new_block_id() -> str:
    """Return a transport-safe identifier for one conversation block."""
    return f"blk_{uuid4().hex}"


def new_tool_call_id() -> str:
    """Return a transport-safe identifier for one tool execution."""
    return f"tool_{uuid4().hex}"


@dataclass(frozen=True)
class ContentBlock:
    """Serializable snapshot of one typed conversation block."""

    block_id: str
    kind: str
    content: str = ""
    status: str = BlockStatus.PENDING
    title: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.block_id.strip():
            raise ValueError("Content block id cannot be empty")
        if not self.kind.strip():
            raise ValueError("Content block kind cannot be empty")
        if not self.status.strip():
            raise ValueError("Content block status cannot be empty")

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "block_id": self.block_id,
            "kind": self.kind,
            "content": self.content,
            "status": self.status,
            "details": dict(self.details),
        }
        if self.title is not None:
            payload["title"] = self.title
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> ContentBlock:
        details = payload.get("details", {})
        if not isinstance(details, Mapping):
            raise TypeError("Content block details must be a mapping")
        title = payload.get("title")
        return cls(
            block_id=str(payload.get("block_id", "")),
            kind=str(payload.get("kind", "")),
            content=str(payload.get("content", "")),
            status=str(payload.get("status", BlockStatus.PENDING)),
            title=None if title is None else str(title),
            details=dict(details),
        )
