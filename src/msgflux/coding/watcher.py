"""Portable session observation over the service's native thread watcher."""

from msgflux.runtime.event_hub import ThreadWatcher
from msgflux.runtime.service.records import EventRecord
from msgflux.runtime.service.serialization import event_record, snapshot_record


class _SessionWatcher:
    """Project one already-subscribed watcher without owning its producer."""

    def __init__(self, watcher: ThreadWatcher):
        self._watcher = watcher
        self.snapshot = snapshot_record(watcher.snapshot)

    def __aiter__(self):
        return self

    async def __anext__(self) -> EventRecord:
        try:
            return event_record(await anext(self._watcher))
        except BaseException:
            await self.aclose()
            raise

    async def aclose(self):
        await self._watcher.aclose()
