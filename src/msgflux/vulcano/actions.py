from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

__all__ = [
    "CancelExecution",
    "InputMode",
    "RuntimeAction",
    "StopRuntime",
    "SubmitInput",
]

InputMode = Literal["auto", "steer", "follow_up"]


def _new_correlation_id() -> str:
    return uuid4().hex


@dataclass(frozen=True)
class SubmitInput:
    """Text submitted by a client to the Vulcano runtime."""

    text: str
    mode: InputMode = "auto"
    correlation_id: str = field(default_factory=_new_correlation_id)

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "steer", "follow_up"}:
            raise ValueError(f"Unsupported input mode: {self.mode!r}")


@dataclass(frozen=True)
class CancelExecution:
    """Cancel the active execution and discard its pending input queue."""

    reason: str = "requested"
    correlation_id: str = field(default_factory=_new_correlation_id)


@dataclass(frozen=True)
class StopRuntime:
    """Request an orderly runtime shutdown."""

    reason: str = "requested"
    correlation_id: str = field(default_factory=_new_correlation_id)


RuntimeAction = CancelExecution | SubmitInput | StopRuntime
