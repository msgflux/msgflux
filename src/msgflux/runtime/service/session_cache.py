"""Private per-thread cache for live AgentSession bindings."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from types import MappingProxyType
from typing import Any, Callable, Mapping

import msgspec

from msgflux.runtime.service.records import (
    ServiceBusyError,
    ServiceRecoveryRequiredError,
)

_LOADING_SESSIONS: contextvars.ContextVar[frozenset[tuple[int, str]]] = (
    contextvars.ContextVar("msgflux_loading_service_sessions", default=frozenset())
)


class _SessionLoadCleanupError(RuntimeError):
    """Keep a partially initialized binding reachable when cleanup fails."""

    def __init__(self, session: Any, error: BaseException) -> None:
        super().__init__(f"Session initialization cleanup failed: {error}")
        self.session = session
        self.error = error


class _Entry(msgspec.Struct, forbid_unknown_fields=True):
    thread: Any
    key: str
    generation: int
    session: Any
    pins: int = 0
    close_task: asyncio.Task | None = None
    close_error: BaseException | None = None


class SessionLease:
    """A pin on one live AgentSession generation."""

    def __init__(self, cache: _SessionCache, entry: _Entry) -> None:
        self._session = entry.session
        self._cache: _SessionCache | None = cache
        self._key = entry.key
        self._generation = entry.generation
        self._release_task: asyncio.Task | None = None
        self._closed = False

    @property
    def session(self) -> Any:
        if self._closed:
            raise RuntimeError("Session lease is closed")
        return self._session

    async def aclose(self) -> None:
        if self._closed:
            return
        if self._release_task is None:
            self._release_task = asyncio.create_task(self._release())
        await asyncio.shield(self._release_task)

    async def _release(self) -> None:
        cache = self._cache
        try:
            if cache is not None:
                await cache._unpin(self._key, self._generation)
        finally:
            self._closed = True
            self._cache = None
            self._session = None
            self._release_task = None

    async def __aenter__(self) -> SessionLease:
        return self

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        await self.aclose()


class _SessionCache:
    """Single-flight live session cache with explicit pins and close quarantine.

    ``load`` and ``idle_reason`` are host/service callbacks. Loading and closing
    happen outside the cache lock; per-thread tasks serialize each generation.
    """

    def __init__(
        self,
        load: Callable[[Any], Any],
        idle_reason: Callable[[Any, Any], str | None],
    ) -> None:
        if not callable(load) or not callable(idle_reason):
            raise TypeError("load and idle_reason must be callable")
        self._load = load
        self._idle_reason = idle_reason
        self._lock = asyncio.Lock()
        self._entries: dict[str, _Entry] = {}
        self._loads: dict[str, asyncio.Task] = {}
        self._generation = 0
        self._closed = False
        self._shutdown_task: asyncio.Task | None = None

    @property
    def sessions(self) -> Mapping[str, Any]:
        """Read-only snapshot of loaded, closing, and quarantined sessions."""
        return MappingProxyType(
            {key: entry.session for key, entry in self._entries.items()}
        )

    async def acquire(self, thread: Any) -> SessionLease:
        key = self._thread_key(thread)
        if (id(self), key) in _LOADING_SESSIONS.get():
            raise RuntimeError(f"Session factory re-entered thread {key!r}")

        while True:
            async with self._lock:
                if self._closed:
                    raise ServiceRecoveryRequiredError(
                        "The AgentService session cache is closing"
                    )
                entry = self._entries.get(key)
                if entry is not None:
                    if entry.close_error is not None:
                        raise self._quarantined_error(entry)
                    if entry.close_task is not None:
                        wait_for = entry.close_task
                    else:
                        entry.pins += 1
                        return SessionLease(self, entry)
                else:
                    wait_for = self._loads.get(key)
                    if wait_for is None:
                        self._generation += 1
                        generation = self._generation
                        wait_for = self._create_task(
                            self._load_and_publish(thread, key, generation)
                        )
                        self._loads[key] = wait_for
            # A canceled caller must not cancel shared load/close work.
            await asyncio.shield(wait_for)

    async def release(self, thread: Any) -> bool:
        """Close an idle unpinned generation; absent threads are a no-op."""
        key = self._thread_key(thread)
        while True:
            async with self._lock:
                entry = self._entries.get(key)
                if entry is None:
                    load_task = self._loads.get(key)
                    if load_task is None:
                        return False
                    close_task = None
                else:
                    if entry.close_error is not None:
                        raise self._quarantined_error(entry)
                    if entry.close_task is not None:
                        close_task = entry.close_task
                        load_task = None
                    else:
                        if entry.pins:
                            raise ServiceBusyError(
                                f"Session {key!r} has {entry.pins} active lease(s)"
                            )
                        reason = self._idle_reason(entry.thread, entry.session)
                        if reason:
                            raise ServiceBusyError(reason)
                        close_task = self._start_close_locked(entry)
                        load_task = None
            task = close_task or load_task
            if task is None:  # Defensive: the entry/load changed unexpectedly.
                raise RuntimeError(f"Session {key!r} changed during release")
            await asyncio.shield(task)
            if close_task is not None:
                return True

    async def close_all(self) -> list[Exception]:
        """Stop new loads, close every generation, and return close failures.

        Failed generations remain in ``sessions`` so their partially closed
        resources stay reachable and cannot be replaced by a new factory result.
        """
        async with self._lock:
            if self._shutdown_task is None:
                self._closed = True
                self._shutdown_task = self._create_task(self._close_all_impl())
            shutdown_task = self._shutdown_task
        return await asyncio.shield(shutdown_task)

    async def _close_all_impl(self) -> list[Exception]:
        async with self._lock:
            load_tasks = tuple(self._loads.values())

        if load_tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in load_tasks),
                return_exceptions=True,
            )

        async with self._lock:
            close_tasks = []
            for entry in self._entries.values():
                if entry.close_error is not None:
                    continue
                task = entry.close_task or self._start_close_locked(entry)
                close_tasks.append(task)

        if close_tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in close_tasks),
                return_exceptions=True,
            )

        async with self._lock:
            return [
                self._quarantined_error(entry)
                for entry in self._entries.values()
                if entry.close_error is not None
            ]

    async def _load_and_publish(self, thread: Any, key: str, generation: int):
        token = _LOADING_SESSIONS.set(_LOADING_SESSIONS.get() | {(id(self), key)})
        try:
            session = self._load(thread)
            if inspect.isawaitable(session):
                session = await session
            entry = _Entry(thread, key, generation, session)
            async with self._lock:
                if self._loads.get(key) is asyncio.current_task():
                    self._loads.pop(key, None)
                if self._closed:
                    entry.close_task = self._create_task(self._close_late_entry(entry))
                    close_task = entry.close_task
                else:
                    self._entries[key] = entry
                    close_task = None
            if close_task is not None:
                await asyncio.shield(close_task)
                raise ServiceRecoveryRequiredError(
                    "Session factory completed after the service began closing"
                )
            return session
        except _SessionLoadCleanupError as error:
            entry = _Entry(thread, key, generation, error.session)
            entry.close_error = error.error
            async with self._lock:
                self._entries[key] = entry
                if self._loads.get(key) is asyncio.current_task():
                    self._loads.pop(key, None)
            raise self._quarantined_error(entry) from error.error
        except BaseException:
            async with self._lock:
                if self._loads.get(key) is asyncio.current_task():
                    self._loads.pop(key, None)
            raise
        finally:
            _LOADING_SESSIONS.reset(token)

    async def _close_late_entry(self, entry: _Entry) -> None:
        """Dispose a late factory result without publishing it while closing."""
        try:
            await self._invoke_close(entry)
        except BaseException as error:
            entry.close_error = error
            async with self._lock:
                self._entries[entry.key] = entry
            raise self._quarantined_error(entry) from error

    def _start_close_locked(self, entry: _Entry) -> asyncio.Task:
        if entry.close_task is None:
            entry.close_task = self._create_task(self._close_entry(entry))
        return entry.close_task

    @staticmethod
    def _observe_task(task: asyncio.Task) -> None:
        """Retrieve exceptions even when every shielded waiter was cancelled."""
        if not task.cancelled():
            task.exception()

    @classmethod
    def _create_task(cls, awaitable: Any) -> asyncio.Task:
        task = asyncio.create_task(awaitable)
        task.add_done_callback(cls._observe_task)
        return task

    async def _close_entry(self, entry: _Entry) -> None:
        try:
            await self._invoke_close(entry)
        except BaseException as error:
            entry.close_error = error
            raise self._quarantined_error(entry) from error
        async with self._lock:
            if self._entries.get(entry.key) is entry:
                self._entries.pop(entry.key, None)

    @staticmethod
    async def _invoke_close(entry: _Entry) -> None:
        callback = getattr(entry.session, "on_close", None)
        if callback is not None:
            result = callback()
            if inspect.isawaitable(result):
                await result

    @staticmethod
    def _quarantined_error(entry: _Entry) -> ServiceRecoveryRequiredError:
        reason = f": {entry.close_error}" if entry.close_error is not None else ""
        error = ServiceRecoveryRequiredError(
            f"Session {entry.key!r} could not be closed; it is quarantined{reason}"
        )
        if entry.close_error is not None:
            error.__cause__ = entry.close_error
        return error

    async def _unpin(self, key: str, generation: int) -> None:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.generation != generation:
                return
            if entry.pins <= 0:
                return
            entry.pins -= 1

    @staticmethod
    def _thread_key(thread: Any) -> str:
        key = thread if isinstance(thread, str) else getattr(thread, "thread_id", None)
        if not isinstance(key, str) or not key:
            raise ValueError("thread must be a non-empty id or have thread_id")
        return key
