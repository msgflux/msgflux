"""Private thread-to-loop delivery buffer with coalesced wakeups."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import Any

from msgflux.exceptions import EventBufferOverflowError


def validate_event_buffer_limit(limit: int | None) -> None:
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("event_buffer_limit must be a positive integer or None")


class _EventBuffer:
    def __init__(self, limit: int | None = None) -> None:
        validate_event_buffer_limit(limit)
        self._limit = limit
        self._loop = asyncio.get_running_loop()
        self._lock = threading.Lock()
        self._items: deque[Any] = deque()
        self._ready = asyncio.Event()
        self._closed = False
        self._overflow = False
        self._wake_pending = False

    def _notify_locked(self) -> None:
        if self._wake_pending:
            return
        self._wake_pending = True
        try:
            self._loop.call_soon_threadsafe(self._wake)
        except RuntimeError:
            # Late detached producers must not retain events for a dead loop.
            self._closed = True
            self._items.clear()
            self._wake_pending = False

    def _wake(self) -> None:
        with self._lock:
            self._wake_pending = False
            self._ready.set()

    def put(self, item: Any) -> bool:
        with self._lock:
            if self._loop.is_closed():
                self._closed = True
                self._items.clear()
                return False
            if self._closed:
                return False
            if self._limit is not None and len(self._items) >= self._limit:
                self._items.clear()
                self._overflow = True
                self._closed = True
                self._notify_locked()
                return False
            self._items.append(item)
            self._notify_locked()
            return not self._closed

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._notify_locked()

    async def get(self) -> Any:
        while True:
            with self._lock:
                if self._overflow:
                    self._overflow = False
                    raise EventBufferOverflowError(self._limit)
                if self._items:
                    return self._items.popleft()
                if self._closed:
                    return None
                self._ready.clear()
            await self._ready.wait()
