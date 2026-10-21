"""Immutable automatic live-session cache settings."""

from __future__ import annotations

import math

import msgspec


class SessionCachePolicy(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Capacity and idle-time settings for live AgentSession bindings.

    Set either field to ``None`` to disable that automatic policy. Explicit
    ``AgentService.release_session`` remains available in either case.
    """

    max_loaded: int | None = 64
    idle_timeout: float | None = 300.0

    def __post_init__(self) -> None:
        if self.max_loaded is not None and (
            type(self.max_loaded) is not int or self.max_loaded <= 0
        ):
            raise ValueError("max_loaded must be a positive integer or None")
        if self.idle_timeout is not None and (
            isinstance(self.idle_timeout, bool)
            or not isinstance(self.idle_timeout, (int, float))
            or not math.isfinite(self.idle_timeout)
            or self.idle_timeout < 0
        ):
            raise ValueError("idle_timeout must be finite and nonnegative or None")
