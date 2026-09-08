"""Execution-local event streaming primitives."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from msgflux.runtime.context import ExecutionScope, get_execution_scope
from msgflux.runtime.event_hub import get_event_hub
from msgflux.utils.time import utc_now_isoformat

__all__ = ["ExecutionEvent", "EventType", "emit_event", "event_source"]


_DETACHED_EVENT_TASKS: set[asyncio.Task[Any]] = set()


class EventType:
    RUN_START = "run.start"
    RUN_RESUME = "run.resume"
    RUN_END = "run.end"
    RUN_ERROR = "run.error"
    RUN_INTERRUPTED = "run.interrupted"
    TURN_START = "turn.start"
    TURN_END = "turn.end"
    MODEL_REQUEST = "model.request"
    MODEL_RESPONSE = "model.response"
    MESSAGE_START = "message.start"
    MESSAGE_DELTA = "message.delta"
    MESSAGE_END = "message.end"
    REASONING_DELTA = "reasoning.delta"
    REASONING_SUMMARY_DELTA = "reasoning_summary.delta"
    TOOL_START = "tool.start"
    TOOL_UPDATE = "tool.update"
    TOOL_END = "tool.end"
    TOOL_BLOCKED = "tool.blocked"
    TOOL_PERMISSION_DENIED = "tool.permission_denied"
    TASK_START = "task.start"
    TASK_UPDATE = "task.update"
    TASK_END = "task.end"
    COMPACTION_START = "compaction.start"
    COMPACTION_END = "compaction.end"
    CHECKPOINT_SAVED = "checkpoint.saved"
    HANDLER_ERROR = "handler.error"


@dataclass(frozen=True)
class ExecutionEvent:
    """One ordered event emitted by an execution."""

    type: str
    timestamp: str
    data: Mapping[str, Any] = field(default_factory=dict)
    run_id: str | None = None
    source_path: tuple[str, ...] = ()


@dataclass(frozen=True)
class _EventSource:
    name: str
    type: str


_CURRENT_EVENT_SOURCES: contextvars.ContextVar[tuple[_EventSource, ...]] = (
    contextvars.ContextVar("msgflux_event_sources", default=())
)


@contextmanager
def event_source(name: str, source_type: str):
    """Attach a logical event origin to all emissions in this context."""
    source = _EventSource(name=str(name), type=str(source_type))
    current = _CURRENT_EVENT_SOURCES.get()
    if current and current[-1] == source:
        yield
        return
    token = _CURRENT_EVENT_SOURCES.set((*current, source))
    try:
        yield
    finally:
        _CURRENT_EVENT_SOURCES.reset(token)


class _EventSink:
    def __init__(
        self,
        publish: Callable[[ExecutionEvent], None],
        *,
        root_module: Any = None,
    ) -> None:
        self._publish = publish
        self._lock = threading.Lock()
        self._scope = ExecutionScope()
        self.root_module = root_module

    def emit(
        self,
        event_type: str,
        data: Mapping[str, Any] | None = None,
        *,
        scope: ExecutionScope | None = None,
    ) -> None:
        current = scope or get_execution_scope()
        if current.run_id is not None or current.thread_id is not None:
            self._scope = current
        scope = current if current.run_id is not None else self._scope
        sources = _CURRENT_EVENT_SOURCES.get()
        event_data = dict(data or {})
        if event_type == EventType.RUN_START:
            event_data = {
                **event_data,
                **{
                    key: value
                    for key, value in {
                        "thread_id": scope.thread_id,
                        "namespace": scope.namespace,
                        "parent_run_id": scope.parent_run_id,
                        "root_run_id": scope.root_run_id,
                    }.items()
                    if value is not None
                },
            }
        event = ExecutionEvent(
            type=event_type,
            timestamp=utc_now_isoformat(),
            data=event_data,
            run_id=scope.run_id,
            source_path=tuple(f"{item.type}:{item.name}" for item in sources),
        )
        with self._lock:
            get_event_hub().publish(scope.thread_id, event)
            self._publish(event)


_CURRENT_EVENT_SINK: contextvars.ContextVar[_EventSink | None] = contextvars.ContextVar(
    "msgflux_event_sink", default=None
)


@contextmanager
def _capture_events(sink: _EventSink):
    token = _CURRENT_EVENT_SINK.set(sink)
    try:
        yield
    finally:
        _CURRENT_EVENT_SINK.reset(token)


def _hub_event_sink(*, root_module: Any = None) -> _EventSink:
    """Create a sink that updates the shared hub without a direct subscriber."""
    return _EventSink(lambda _event: None, root_module=root_module)


def _track_event_task(task: asyncio.Task[Any]) -> None:
    """Retain a detached event consumer until it settles."""
    _DETACHED_EVENT_TASKS.add(task)
    task.add_done_callback(_DETACHED_EVENT_TASKS.discard)


def emit_event(
    event_type: str,
    data: Mapping[str, Any] | None = None,
    *,
    scope: ExecutionScope | None = None,
) -> None:
    """Emit an event when the current execution is being observed."""
    sink = _CURRENT_EVENT_SINK.get()
    if sink is not None:
        sink.emit(event_type, data, scope=scope)


def _is_event_stream_root(module: Any) -> bool:
    sink = _CURRENT_EVENT_SINK.get()
    return sink is not None and sink.root_module is module


def _is_capturing_events() -> bool:
    return _CURRENT_EVENT_SINK.get() is not None


_CLOSED = object()


class _AsyncEventChannel:
    def __init__(self, *, root_module: Any = None) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[ExecutionEvent | object] = asyncio.Queue()
        self.sink = _EventSink(self._publish, root_module=root_module)

    def _publish(self, event: ExecutionEvent) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, event)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, _CLOSED)

    async def get(self) -> ExecutionEvent | None:
        item = await self._queue.get()
        return None if item is _CLOSED else item
