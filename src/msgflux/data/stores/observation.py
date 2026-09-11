"""Run-scoped committed transitions, separate from live execution deltas."""

import asyncio
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping


class CheckpointCursorError(ValueError):
    """Cursor identity is stale, unavailable or separated by a history gap."""


@dataclass(frozen=True)
class CheckpointCursor:
    namespace: str
    thread_id: str
    run_id: str
    stream_id: str
    revision: int

    def __post_init__(self):
        for value in (self.namespace, self.thread_id, self.run_id, self.stream_id):
            if not isinstance(value, str) or not value:
                raise CheckpointCursorError("Cursor identity must be non-empty strings")
        if type(self.revision) is not int or self.revision < 0:
            raise CheckpointCursorError(
                "Cursor revision must be a non-negative integer"
            )


@dataclass(frozen=True)
class CommittedEvent:
    cursor: CheckpointCursor
    data: Mapping[str, Any]

    @property
    def event_id(self):
        return f"{self.cursor.stream_id}:{self.cursor.revision}"


@dataclass(frozen=True)
class CheckpointPage:
    cursor: CheckpointCursor
    snapshot: Mapping[str, Any] | None = None
    events: tuple[CommittedEvent, ...] = ()


def validate_read(state, namespace, thread_id, run_id, after, limit):
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    checkpoint = (state or {}).get("_checkpoint", {})
    stream_id = checkpoint.get("stream_id")
    if not stream_id:
        raise CheckpointCursorError("Run has no durable commit stream")
    latest = CheckpointCursor(
        namespace, thread_id, run_id, stream_id, checkpoint["revision"]
    )
    if after is not None:
        if not isinstance(after, CheckpointCursor):
            raise TypeError("after must be a CheckpointCursor")
        if (after.namespace, after.thread_id, after.run_id, after.stream_id) != (
            namespace,
            thread_id,
            run_id,
            stream_id,
        ) or after.revision > latest.revision:
            raise CheckpointCursorError(
                "Cursor does not belong to this stream position"
            )
    return latest


def make_page(state, latest, after, records):
    if after is None:
        return CheckpointPage(latest, snapshot=deepcopy(state))
    events = []
    revision = after.revision
    for number, data in records:
        if number != revision + 1:
            raise CheckpointCursorError("Durable event history has a gap")
        revision = number
        cursor = CheckpointCursor(
            latest.namespace, latest.thread_id, latest.run_id, latest.stream_id, number
        )
        events.append(CommittedEvent(cursor, deepcopy(data)))
    if not events and revision < latest.revision:
        raise CheckpointCursorError("Durable event history is unavailable")
    return CheckpointPage(events[-1].cursor if events else after, events=tuple(events))


async def observe_checkpoints(
    store,
    namespace,
    thread_id,
    run_id,
    *,
    after=None,
    limit=100,
    poll_interval=0.1,
):
    """Pull bounded pages; disconnecting never cancels the producing execution."""
    if (
        isinstance(poll_interval, bool)
        or not isinstance(poll_interval, (int, float))
        or not math.isfinite(poll_interval)
        or poll_interval <= 0
    ):
        raise ValueError("poll_interval must be a positive finite number")
    cursor = after
    while True:
        page = await store.aread_commits(
            namespace, thread_id, run_id, after=cursor, limit=limit
        )
        if page.snapshot is not None or page.events:
            cursor = page.cursor
            yield page
        else:
            await asyncio.sleep(poll_interval)
