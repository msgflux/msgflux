"""Abort primitives for local runtime cancellation."""

from __future__ import annotations

import asyncio
import threading
from contextlib import suppress
from typing import Awaitable, TypeVar

from msgflux.exceptions import AbortRequestedError

T = TypeVar("T")


class AbortSignal:
    """Local, non-persistent cancellation signal for an active execution."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._waiters_lock = threading.Lock()
        self._waiters: set[tuple[asyncio.AbstractEventLoop, asyncio.Future]] = set()
        self.reason: str | None = None

    def abort(self, reason: str | None = None) -> None:
        self.reason = reason
        self._event.set()
        with self._waiters_lock:
            waiters = tuple(self._waiters)
        for loop, waiter in waiters:
            if not waiter.done():
                loop.call_soon_threadsafe(_resolve_waiter, waiter)

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    def raise_if_aborted(self) -> None:
        if self.aborted:
            raise AbortRequestedError(self.reason)

    async def araise_if_aborted(self) -> None:
        self.raise_if_aborted()

    async def wait(self) -> None:
        """Wait until cancellation is requested without blocking a worker thread."""
        if self.aborted:
            return
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        entry = (loop, waiter)
        with self._waiters_lock:
            if self.aborted:
                return
            self._waiters.add(entry)
        try:
            await waiter
        finally:
            with self._waiters_lock:
                self._waiters.discard(entry)


def _resolve_waiter(waiter: asyncio.Future) -> None:
    if not waiter.done():
        waiter.set_result(None)


async def await_with_abort(
    awaitable: Awaitable[T],
    abort_signal: AbortSignal | None,
) -> T:
    """Await an operation while cooperatively observing an AbortSignal."""
    if abort_signal is None:
        return await awaitable
    operation = asyncio.ensure_future(awaitable)
    if abort_signal.aborted:
        operation.cancel()
        with suppress(asyncio.CancelledError):
            await operation
        abort_signal.raise_if_aborted()
    abort_waiter = asyncio.create_task(abort_signal.wait())
    try:
        done, _ = await asyncio.wait(
            {operation, abort_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort_waiter in done:
            operation.cancel()
            with suppress(asyncio.CancelledError):
                await operation
            abort_signal.raise_if_aborted()
        result = await operation
        abort_signal.raise_if_aborted()
        return result
    finally:
        if not operation.done():
            operation.cancel()
            with suppress(asyncio.CancelledError):
                await operation
        abort_waiter.cancel()
        with suppress(asyncio.CancelledError):
            await abort_waiter
