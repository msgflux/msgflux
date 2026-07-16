from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

__all__ = [
    "DomainEvent",
    "EventDraft",
    "EventStream",
    "EventSubscription",
    "EventType",
]


class EventType:
    """Stable event names shared by runtimes and clients."""

    RUNTIME_STARTED = "runtime.started"
    RUNTIME_STOPPED = "runtime.stopped"
    MESSAGE_USER = "message.user"
    ASSISTANT_STARTED = "assistant.started"
    ASSISTANT_DELTA = "assistant.delta"
    ASSISTANT_COMPLETED = "assistant.completed"
    COMMAND_STARTED = "command.started"
    COMMAND_OUTPUT = "command.output"
    COMMAND_COMPLETED = "command.completed"
    COMMAND_ERROR = "command.error"
    CLIENT_ACTION = "client.action"
    EXTENSION_LOADED = "extension.loaded"
    EXTENSION_UNLOADED = "extension.unloaded"
    EXTENSION_FAILED = "extension.failed"
    RUNTIME_ERROR = "runtime.error"


@dataclass(frozen=True)
class EventDraft:
    """Unsequenced event returned by a runtime extension."""

    type: str
    payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class DomainEvent:
    """Serializable fact emitted by the Vulcano runtime."""

    type: str
    sequence: int
    payload: Mapping[str, object] = field(default_factory=dict)
    correlation_id: str | None = None
    occurred_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "sequence": self.sequence,
            "payload": dict(self.payload),
            "correlation_id": self.correlation_id,
            "occurred_at": self.occurred_at,
        }


_STREAM_END = object()


class EventSubscription:
    """Independent async iterator over events published after subscription."""

    def __init__(
        self,
        stream: EventStream,
        queue: asyncio.Queue[DomainEvent | object],
    ) -> None:
        self._stream = stream
        self._queue = queue
        self._closed = False

    def __aiter__(self) -> EventSubscription:
        return self

    async def __anext__(self) -> DomainEvent:
        item = await self._queue.get()
        if item is _STREAM_END:
            self._closed = True
            raise StopAsyncIteration
        if not isinstance(item, DomainEvent):
            raise RuntimeError("Vulcano event stream received an invalid item")
        return item

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            self._stream._unsubscribe(self._queue)


class EventStream:
    """In-process broadcast stream with one queue per subscriber."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[DomainEvent | object]] = set()
        self._closed = False

    def subscribe(self) -> EventSubscription:
        queue: asyncio.Queue[DomainEvent | object] = asyncio.Queue()
        if self._closed:
            queue.put_nowait(_STREAM_END)
        else:
            self._subscribers.add(queue)
        return EventSubscription(self, queue)

    async def publish(self, event: DomainEvent) -> None:
        if self._closed:
            return
        for queue in tuple(self._subscribers):
            queue.put_nowait(event)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for queue in tuple(self._subscribers):
            queue.put_nowait(_STREAM_END)
        self._subscribers.clear()

    def _unsubscribe(self, queue: asyncio.Queue[DomainEvent | object]) -> None:
        self._subscribers.discard(queue)
