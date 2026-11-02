"""Small process-local projections of the stream the service already consumes."""

import time
from collections import deque

from msgflux.runtime.service.records import RunActivity, RunActivityEvent
from msgflux.utils.time import utc_now_isoformat

_RECENT_LIMIT = 8


def _label(value, limit):
    return value[:limit] if isinstance(value, str) else None


class RunActivityTracker:
    """Own diagnostic labels only; discard them with the foreground worker."""

    __slots__ = (
        "_events",
        "_interrupt_at",
        "_interrupt_time",
        "_last_at",
        "_last_time",
        "_last_type",
    )

    def __init__(self):
        self._events = deque(maxlen=_RECENT_LIMIT)
        self._last_type = None
        self._last_at = None
        self._last_time = None
        self._interrupt_at = None
        self._interrupt_time = None

    def observe(self, event):
        event_type = _label(event.type, 64)
        if event_type is None:
            return
        self._last_type = event_type
        self._last_at = _label(event.timestamp, 64)
        self._last_time = time.monotonic()
        # Token deltas update liveness without displacing useful boundaries or
        # retaining model text. Other events contribute labels, never data.
        if event_type.endswith(".delta"):
            return
        data = event.data if type(event.data) is dict else {}
        self._events.append(
            RunActivityEvent(
                event_type=event_type,
                timestamp=self._last_at or "",
                source_path=tuple(
                    part[:128]
                    for part in event.source_path[:8]
                    if isinstance(part, str)
                ),
                tool_name=_label(data.get("tool_name"), 128),
            )
        )

    def request_interrupt(self):
        if self._interrupt_time is None:
            self._interrupt_at = utc_now_isoformat()
            self._interrupt_time = time.monotonic()

    def snapshot(self, receipt, *, interrupted):
        now = time.monotonic()
        return RunActivity(
            thread_id=receipt.thread_id,
            run_id=receipt.run_id,
            status=receipt.status,
            local_worker=True,
            last_event_type=self._last_type,
            last_event_at=self._last_at,
            seconds_since_event=(
                max(0.0, now - self._last_time) if self._last_time is not None else None
            ),
            interrupt_requested=interrupted,
            interrupt_requested_at=self._interrupt_at,
            seconds_since_interrupt=(
                max(0.0, now - self._interrupt_time)
                if self._interrupt_time is not None
                else None
            ),
            recent_events=tuple(self._events),
        )
