"""Small process-local projections of the stream the service already consumes."""

import json
import math
import sys
import time
from collections import deque
from itertools import islice

import msgspec

from msgflux.envs import envs
from msgflux.runtime.service.records import RunActivity, RunActivityEvent, RunToolCall
from msgflux.utils.time import utc_now_isoformat

_RECENT_LIMIT = 8
DEFAULT_TOOL_CALL_LIMIT = envs.service_activity_tool_call_limit
MAX_TOOL_CALLS = envs.service_activity_tool_call_capacity
ARGUMENT_PREVIEW_BYTES = envs.service_activity_argument_preview_bytes
_SENSITIVE_NAMES = (
    "apikey",
    "apikeys",
    "token",
    "password",
    "passwd",
    "secret",
    "authorization",
    "cookie",
    "credentials",
    "privatekey",
)


def validate_tool_call_limit(limit: int, *, maximum: int | None = None) -> int:
    if type(limit) is not int or limit < 1 or (maximum is not None and limit > maximum):
        bound = (
            f" between 1 and {maximum}" if maximum is not None else " greater than zero"
        )
        raise ValueError(f"tool call limit must be an integer{bound}")
    return limit


class _ActivitySettings(msgspec.Struct, frozen=True):
    capacity: int
    default_limit: int
    preview_bytes: int

    def __post_init__(self):
        for name in ("capacity", "default_limit"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"service activity {name} must be a positive integer")
        if self.capacity > sys.maxsize:
            raise ValueError("service activity capacity exceeds the platform limit")
        if type(self.preview_bytes) is not int or self.preview_bytes < 0:
            raise ValueError(
                "service activity preview bytes must be a nonnegative integer"
            )

    @classmethod
    def from_environment(cls):
        return cls(
            envs.service_activity_tool_call_capacity,
            envs.service_activity_tool_call_limit,
            envs.service_activity_argument_preview_bytes,
        )

    def limit(self, requested):
        return (
            min(self.default_limit, self.capacity)
            if requested is None
            else validate_tool_call_limit(requested, maximum=self.capacity)
        )


class _ArgumentPreview:
    """Bound work and allocations before serializing any argument data."""

    def __init__(self, preview_bytes):
        self.preview_bytes = preview_bytes
        self.truncated = False
        self.remaining = 64

    def value(self, value, depth=0):
        self.remaining -= 1
        kind = type(value)
        if self.remaining < 0:
            return self.omitted()
        if value is None or kind is bool:
            return value
        if kind is str:
            self.truncated |= len(value) > self.preview_bytes
            return value[: self.preview_bytes] + (
                "…" if len(value) > self.preview_bytes else ""
            )
        if kind is int and value.bit_length() <= 256:
            return value
        if kind is float and math.isfinite(value):
            return value
        if depth < 2 and kind is dict:
            return self.mapping(value, depth)
        if depth < 2 and kind in (list, tuple):
            self.truncated |= len(value) > 8
            return [self.value(item, depth + 1) for item in islice(value, 8)]
        return self.omitted()

    def omitted(self):
        self.truncated = True
        return "…"

    def mapping(self, value, depth):
        self.truncated |= len(value) > 8
        result = {}
        for key, item in islice(value.items(), 8):
            if type(key) is not str:
                self.truncated = True
                continue
            name = key[:64]
            self.truncated |= len(key) > 64
            normalized = key[-64:].lower().replace("_", "").replace("-", "")
            result[name] = (
                "[redacted]"
                if normalized.endswith(_SENSITIVE_NAMES)
                else self.value(item, depth + 1)
            )
        return result


def _argument_preview(arguments, preview_bytes):
    if preview_bytes == 0:
        return "", True
    preview = _ArgumentPreview(preview_bytes)
    text = json.dumps(
        preview.value(arguments), ensure_ascii=False, separators=(",", ":")
    )
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) > preview_bytes:
        suffix = "…" if preview_bytes >= 3 else "." * preview_bytes
        head = max(0, preview_bytes - len(suffix.encode("utf-8")))
        text = encoded[:head].decode("utf-8", errors="ignore") + suffix
        preview.truncated = True
    else:
        text = encoded.decode("utf-8")
    return text, preview.truncated


def _label(value, limit):
    return value[:limit] if isinstance(value, str) else None


class RunActivityTracker:
    """Own bounded diagnostic projections; discard them with the worker."""

    __slots__ = (
        "_events",
        "_interrupt_at",
        "_interrupt_time",
        "_last_at",
        "_last_time",
        "_last_type",
        "_settings",
        "_tool_calls",
    )

    def __init__(self, settings=None):
        self._settings = settings or _ActivitySettings.from_environment()
        self._events = deque(maxlen=_RECENT_LIMIT)
        self._tool_calls = deque(maxlen=self._settings.capacity)
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
        # retaining model text. Started tools additionally keep bounded previews;
        # other events contribute labels only.
        if event_type.endswith(".delta"):
            return
        data = event.data if type(event.data) is dict else {}
        if event_type == "tool.start":
            name = _label(data.get("tool_name"), 128)
            if name:
                preview, truncated = _argument_preview(
                    data.get("arguments", {}), self._settings.preview_bytes
                )
                self._tool_calls.append(
                    RunToolCall(
                        tool_name=name,
                        timestamp=self._last_at or "",
                        tool_call_id=_label(data.get("tool_call_id"), 128),
                        run_id=_label(event.run_id, 128),
                        source_path=tuple(
                            part[:128]
                            for part in event.source_path[:8]
                            if isinstance(part, str)
                        ),
                        arguments_preview=preview,
                        arguments_truncated=truncated,
                    )
                )
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

    def tool_calls(self, limit=None):
        limit = self._settings.limit(limit)
        return tuple(
            islice(self._tool_calls, max(0, len(self._tool_calls) - limit), None)
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
