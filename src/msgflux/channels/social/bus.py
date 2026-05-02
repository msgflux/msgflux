import asyncio
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
