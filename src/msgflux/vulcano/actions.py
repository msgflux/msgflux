from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

__all__ = ["RuntimeAction", "StopRuntime", "SubmitInput"]


def _new_correlation_id() -> str:
    return uuid4().hex


@dataclass(frozen=True)
class SubmitInput:
    """Text submitted by a client to the Vulcano runtime."""

    text: str
    correlation_id: str = field(default_factory=_new_correlation_id)


@dataclass(frozen=True)
class StopRuntime:
    """Request an orderly runtime shutdown."""

    reason: str = "requested"
    correlation_id: str = field(default_factory=_new_correlation_id)


RuntimeAction = SubmitInput | StopRuntime
