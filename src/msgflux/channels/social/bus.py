import asyncio
import time
from typing import Any

from msgflux.channels.social.types import SocialEvent


class InMemorySocialEventBus:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    async def publish(self, event: SocialEvent) -> None:
        await self._queue.put(event)

    async def get(self) -> Any:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    async def drain(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        await self._queue.put(None)


class InMemorySocialDedupStore:
    """TTL-based dedupe store for a single Python process."""

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def seen_or_mark(self, key: str, ttl_s: float) -> bool:
        now = time.monotonic()
        async with self._lock:
            expired = [
                item_key
                for item_key, expires_at in self._seen.items()
                if expires_at <= now
            ]
            for item_key in expired:
                self._seen.pop(item_key, None)

            if key in self._seen:
                return True
            self._seen[key] = now + ttl_s
            return False
