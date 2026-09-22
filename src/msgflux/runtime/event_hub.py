"""Process-local execution event distribution and live thread projections."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Mapping

from msgflux.exceptions import EventBufferOverflowError
from msgflux.runtime.event_buffer import _EventBuffer, validate_event_buffer_limit

if TYPE_CHECKING:
    from msgflux.runtime.approvals.records import ApprovalRecord
    from msgflux.runtime.events import ExecutionEvent


@dataclass(frozen=True)
class LiveRunSnapshot:
    """Current presentation state for one active run source."""

    run_id: str | None
    source_path: tuple[str, ...]
    namespace: str | None = None
    streaming_message: Any = None
    reasoning: str | None = None
    reasoning_summary: str | None = None


@dataclass(frozen=True)
class RunningToolSnapshot:
    """One tool invocation that has started but not settled."""

    run_id: str | None
    source_path: tuple[str, ...]
    tool_call_id: str | None
    tool_name: str | None
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackgroundTaskSnapshot:
    """One process-local background task that has not settled."""

    task_id: str
    tool_name: str | None
    status: str | None
    progress: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ThreadSnapshot:
    """Durable conversation state combined with process-local live state."""

    thread_id: str
    namespace: str | None = None
    messages: Any = None
    active_runs: tuple[LiveRunSnapshot, ...] = ()
    running_tools: tuple[RunningToolSnapshot, ...] = ()
    background_tasks: tuple[BackgroundTaskSnapshot, ...] = ()
    approvals: tuple[ApprovalRecord, ...] = ()

    @property
    def active_run(self) -> LiveRunSnapshot | None:
        """Return the most recently started active run, when one exists."""
        return self.active_runs[-1] if self.active_runs else None

    @property
    def streaming_message(self) -> Any:
        """Return the presentation buffer for the most recent active run."""
        active = self.active_run
        return active.streaming_message if active is not None else None


@dataclass
class _LiveRun:
    run_id: str | None
    source_path: tuple[str, ...]
    namespace: str | None = None
    streaming_message: Any = None
    reasoning: str | None = None
    reasoning_summary: str | None = None
    message_chunks: list[Any] = field(default_factory=list)
    reasoning_chunks: list[Any] = field(default_factory=list)
    reasoning_summary_chunks: list[Any] = field(default_factory=list)

    @staticmethod
    def _combine(chunks: list[Any], settled: Any) -> Any:
        if not chunks:
            return settled
        if len(chunks) == 1:
            return chunks[0]
        if all(isinstance(chunk, str) for chunk in chunks):
            combined = "".join(chunks)
        elif all(isinstance(chunk, bytes) for chunk in chunks):
            combined = b"".join(chunks)
        else:
            return chunks[-1]
        # Snapshots are built under the hub lock. Keep the joined immutable value
        # instead of retaining both it and all original fragments; reconnects
        # without new deltas can reuse it. Other consumers may still own chunks.
        chunks[:] = [combined]
        return combined

    def snapshot(self) -> LiveRunSnapshot:
        return LiveRunSnapshot(
            run_id=self.run_id,
            source_path=self.source_path,
            namespace=self.namespace,
            streaming_message=self._combine(
                self.message_chunks,
                self.streaming_message,
            ),
            reasoning=self._combine(self.reasoning_chunks, self.reasoning),
            reasoning_summary=self._combine(
                self.reasoning_summary_chunks,
                self.reasoning_summary,
            ),
        )


@dataclass
class _ThreadLiveState:
    runs: dict[tuple[str | None, tuple[str, ...]], _LiveRun] = field(
        default_factory=dict
    )
    tools: dict[tuple[str | None, tuple[str, ...], str | None], RunningToolSnapshot] = (
        field(default_factory=dict)
    )
    background_tasks: dict[str, BackgroundTaskSnapshot] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return bool(self.runs or self.tools or self.background_tasks)


class ThreadWatcher:
    """Async context manager yielding live events for one existing thread."""

    def __init__(
        self,
        hub: EventHub,
        *,
        thread_id: str,
        namespace: str | None,
        load_messages: Callable[[], Any] | None,
        load_approvals: Callable[[], tuple[Any, ...]] | None = None,
        event_buffer_limit: int | None = None,
    ) -> None:
        validate_event_buffer_limit(event_buffer_limit)
        self._hub = hub
        self.thread_id = thread_id
        self.namespace = namespace
        self._load_messages = load_messages
        self._load_approvals = load_approvals
        self._buffer: _EventBuffer | None = None
        self._event_buffer_limit = event_buffer_limit
        self._closed = False
        self._snapshot: ThreadSnapshot | None = None

    @property
    def snapshot(self) -> ThreadSnapshot:
        if self._snapshot is None:
            raise RuntimeError("Enter the watcher context before reading its snapshot.")
        return self._snapshot

    async def __aenter__(self) -> ThreadWatcher:
        if self._buffer is not None or self._closed:
            raise RuntimeError("A ThreadWatcher cannot be entered more than once.")
        self._buffer = _EventBuffer(self._event_buffer_limit)
        self._snapshot = self._hub._subscribe(self)
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        await self.aclose()

    def __aiter__(self) -> ThreadWatcher:
        if self._buffer is None:
            raise RuntimeError("Enter the watcher context before iterating it.")
        return self

    async def __anext__(self) -> ExecutionEvent:
        if self._buffer is None:
            raise RuntimeError("Enter the watcher context before iterating it.")
        try:
            item = await self._buffer.get()
        except EventBufferOverflowError:
            await self.aclose()
            raise
        if item is None:
            raise StopAsyncIteration
        return item

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._hub._unsubscribe(self)
        if self._buffer is not None:
            self._buffer.close()

    def _enqueue(self, event: ExecutionEvent) -> None:
        if self._closed or self._buffer is None:
            return
        if not self._buffer.put(event):
            self._closed = True
            self._hub._unsubscribe(self)


class EventHub:
    """Thread-safe process-local hub with no durable event log."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._threads: dict[str, _ThreadLiveState] = {}
        self._watchers: dict[str, set[ThreadWatcher]] = {}

    def watch(
        self,
        thread_id: str,
        *,
        namespace: str | None = None,
        load_messages: Callable[[], Any] | None = None,
        load_approvals: Callable[[], tuple[Any, ...]] | None = None,
        event_buffer_limit: int | None = None,
    ) -> ThreadWatcher:
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("`thread_id` must be a non-empty string.")
        return ThreadWatcher(
            self,
            thread_id=thread_id,
            namespace=namespace,
            load_messages=load_messages,
            load_approvals=load_approvals,
            event_buffer_limit=event_buffer_limit,
        )

    def publish(self, thread_id: str | None, event: ExecutionEvent) -> None:
        if not isinstance(thread_id, str) or not thread_id:
            return
        with self._lock:
            state = self._threads.setdefault(thread_id, _ThreadLiveState())
            self._reduce(state, event)
            watchers = tuple(self._watchers.get(thread_id, ()))
            for watcher in watchers:
                watcher._enqueue(event)
            if not state.active and not watchers:
                self._threads.pop(thread_id, None)

    def _subscribe(self, watcher: ThreadWatcher) -> ThreadSnapshot:
        with self._lock:
            messages = (
                watcher._load_messages() if watcher._load_messages is not None else None
            )
            state = self._threads.get(watcher.thread_id)
            snapshot = self._snapshot(
                watcher.thread_id,
                namespace=watcher.namespace,
                messages=messages,
                state=state,
            )
            snapshot = replace(
                snapshot,
                approvals=watcher._load_approvals() if watcher._load_approvals else (),
            )
            self._watchers.setdefault(watcher.thread_id, set()).add(watcher)
            return snapshot

    def _unsubscribe(self, watcher: ThreadWatcher) -> None:
        with self._lock:
            watchers = self._watchers.get(watcher.thread_id)
            if watchers is not None:
                watchers.discard(watcher)
                if not watchers:
                    self._watchers.pop(watcher.thread_id, None)
            state = self._threads.get(watcher.thread_id)
            if state is not None and not state.active:
                self._threads.pop(watcher.thread_id, None)

    @staticmethod
    def _snapshot(
        thread_id: str,
        *,
        namespace: str | None,
        messages: Any,
        state: _ThreadLiveState | None,
    ) -> ThreadSnapshot:
        if state is None:
            return ThreadSnapshot(
                thread_id=thread_id,
                namespace=namespace,
                messages=messages,
            )
        return ThreadSnapshot(
            thread_id=thread_id,
            namespace=namespace,
            messages=messages,
            active_runs=tuple(run.snapshot() for run in state.runs.values()),
            running_tools=tuple(state.tools.values()),
            background_tasks=tuple(state.background_tasks.values()),
        )

    @staticmethod
    def _reduce(  # noqa: C901
        state: _ThreadLiveState, event: ExecutionEvent
    ) -> None:
        run_key = (event.run_id, event.source_path)
        event_type = event.type
        if event_type == "run.start":
            state.runs[run_key] = _LiveRun(
                run_id=event.run_id,
                source_path=event.source_path,
                namespace=event.data.get("namespace"),
            )
            return

        if event_type == "task.start":
            task_id = event.data.get("task_id")
            if isinstance(task_id, str) and task_id:
                state.background_tasks[task_id] = BackgroundTaskSnapshot(
                    task_id=task_id,
                    tool_name=event.data.get("tool_name"),
                    status=event.data.get("status"),
                    progress=dict(event.data.get("progress") or {}),
                )
            return
        if event_type == "task.update":
            task_id = event.data.get("task_id")
            if isinstance(task_id, str) and task_id:
                current = state.background_tasks.get(task_id)
                state.background_tasks[task_id] = BackgroundTaskSnapshot(
                    task_id=task_id,
                    tool_name=event.data.get("tool_name")
                    or (current.tool_name if current is not None else None),
                    status=event.data.get("status")
                    or (current.status if current is not None else None),
                    progress=dict(
                        event.data.get("progress")
                        or (current.progress if current is not None else {})
                    ),
                )
            return
        if event_type == "task.end":
            task_id = event.data.get("task_id")
            if isinstance(task_id, str) and task_id:
                state.background_tasks.pop(task_id, None)
            return

        run = state.runs.get(run_key)
        if (
            run is None
            and event_type
            not in {
                "run.end",
                "run.error",
                "run.interrupted",
                "run.paused",
            }
            and not event_type.startswith("tool.")
        ):
            run = _LiveRun(run_id=event.run_id, source_path=event.source_path)
            state.runs[run_key] = run

        if event_type == "message.start" and run is not None:
            run.streaming_message = None
            run.message_chunks.clear()
        elif event_type == "message.delta" and run is not None:
            delta = event.data.get("delta")
            if delta is not None:
                run.message_chunks.append(delta)
        elif event_type == "message.end" and run is not None:
            run.streaming_message = event.data.get("content")
            run.message_chunks.clear()
        elif event_type == "reasoning.delta" and run is not None:
            delta = event.data.get("delta")
            if delta is not None:
                run.reasoning_chunks.append(delta)
        elif event_type == "reasoning_summary.delta" and run is not None:
            delta = event.data.get("delta")
            if delta is not None:
                run.reasoning_summary_chunks.append(delta)
        elif event_type == "tool.start":
            tool_call_id = event.data.get("tool_call_id")
            tool_key = (event.run_id, event.source_path, tool_call_id)
            state.tools[tool_key] = RunningToolSnapshot(
                run_id=event.run_id,
                source_path=event.source_path,
                tool_call_id=tool_call_id,
                tool_name=event.data.get("tool_name"),
                arguments=dict(event.data.get("arguments") or {}),
            )
        elif event_type == "tool.end":
            tool_call_id = event.data.get("tool_call_id")
            state.tools.pop((event.run_id, event.source_path, tool_call_id), None)
        elif event_type in {"run.end", "run.error", "run.interrupted", "run.paused"}:
            # A run can emit events while nested modules/tools extend its
            # source_path. Its terminal event is emitted after those nested
            # contexts unwind, so remove the whole descendant projection too.
            # Otherwise one stale _LiveRun survives per completed thread.
            source_prefix = event.source_path
            for candidate in tuple(state.runs):
                candidate_run_id, candidate_source = candidate
                if (
                    candidate_run_id == event.run_id
                    and candidate_source[: len(source_prefix)] == source_prefix
                ):
                    state.runs.pop(candidate, None)
            for tool_key in tuple(state.tools):
                tool_run_id, tool_source, _tool_call_id = tool_key
                if (
                    tool_run_id == event.run_id
                    and tool_source[: len(source_prefix)] == source_prefix
                ):
                    state.tools.pop(tool_key, None)

    def _reset(self) -> None:
        """Clear process-local state for isolated tests."""
        with self._lock:
            self._threads.clear()
            watchers = [
                watcher
                for thread_watchers in self._watchers.values()
                for watcher in thread_watchers
            ]
            self._watchers.clear()
        for watcher in watchers:
            watcher._closed = True
            if watcher._buffer is not None:
                watcher._buffer.close()


_EVENT_HUB = EventHub()


def get_event_hub() -> EventHub:
    """Return the process-local runtime event hub."""
    return _EVENT_HUB


__all__ = [
    "BackgroundTaskSnapshot",
    "EventHub",
    "LiveRunSnapshot",
    "RunningToolSnapshot",
    "ThreadSnapshot",
    "ThreadWatcher",
    "get_event_hub",
]
