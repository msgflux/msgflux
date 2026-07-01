"""Abort primitives for local runtime cancellation."""

from __future__ import annotations

import threading

from msgflux.exceptions import AbortRequestedError


class AbortSignal:
    """Local, non-persistent cancellation signal for an active execution."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason: str | None = None

    def abort(self, reason: str | None = None) -> None:
        self.reason = reason
        self._event.set()

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def raise_if_aborted(self) -> None:
        if self.aborted:
            raise AbortRequestedError(self.reason)

    async def araise_if_aborted(self) -> None:
        self.raise_if_aborted()
