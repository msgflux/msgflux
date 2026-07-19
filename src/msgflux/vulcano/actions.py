from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from msgflux.vulcano.permissions import PermissionActionDecision

__all__ = [
    "ActivateSessionTab",
    "CancelExecution",
    "CloseSessionTab",
    "InputMode",
    "ResolvePermission",
    "RuntimeAction",
    "StopRuntime",
    "SubmitInput",
    "ToggleSessionPin",
]

InputMode = Literal["auto", "steer", "follow_up"]


def _new_correlation_id() -> str:
    return uuid4().hex


def _validate_thread_id(thread_id: str) -> None:
    if not thread_id.strip():
        raise ValueError("Session thread id cannot be empty")


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
class ActivateSessionTab:
    """Activate an open or persisted durable session tab."""

    thread_id: str
    correlation_id: str = field(default_factory=_new_correlation_id)

    def __post_init__(self) -> None:
        _validate_thread_id(self.thread_id)


@dataclass(frozen=True)
class CloseSessionTab:
    """Close a session tab without deleting its durable transcript."""

    thread_id: str
    correlation_id: str = field(default_factory=_new_correlation_id)

    def __post_init__(self) -> None:
        _validate_thread_id(self.thread_id)


@dataclass(frozen=True)
class ResolvePermission:
    """Return a client's decision for a pending runtime permission request."""

    request_id: str
    decision: PermissionActionDecision
    correlation_id: str = field(default_factory=_new_correlation_id)

    def __post_init__(self) -> None:
        if not self.request_id.strip():
            raise ValueError("Permission request id cannot be empty")
        if self.decision not in {"allow_once", "allow_session", "deny"}:
            raise ValueError(f"Unsupported permission decision: {self.decision!r}")


@dataclass(frozen=True)
class StopRuntime:
    """Request an orderly runtime shutdown."""

    reason: str = "requested"
    correlation_id: str = field(default_factory=_new_correlation_id)


@dataclass(frozen=True)
class ToggleSessionPin:
    """Toggle whether a session tab is restored on the next launch."""

    thread_id: str
    correlation_id: str = field(default_factory=_new_correlation_id)

    def __post_init__(self) -> None:
        _validate_thread_id(self.thread_id)


RuntimeAction = (
    ActivateSessionTab
    | CancelExecution
    | CloseSessionTab
    | ResolvePermission
    | StopRuntime
    | SubmitInput
    | ToggleSessionPin
)
