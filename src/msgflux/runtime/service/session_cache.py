"""Private per-thread cache for live AgentSession bindings."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

import msgspec

from msgflux.logger import logger
from msgflux.runtime.service.cache_policy import SessionCachePolicy
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
    pending_leases: int = 0
    last_used: float = 0.0
    idle_since: float | None = None
    next_probe_at: float | None = None
    close_task: asyncio.Task | None = None
    probing: bool = False
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
        *,
        policy: SessionCachePolicy | None = None,
        busy_reason: Callable[[Any, Any], str | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(load) or not callable(idle_reason) or not callable(clock):
            raise TypeError("load, idle_reason, and clock must be callable")
        if busy_reason is not None and not callable(busy_reason):
            raise TypeError("busy_reason must be callable or None")
        self._load = load
        self._idle_reason = idle_reason
        self._busy_reason = busy_reason
        self._policy = policy
        self._clock = clock
        self._lock = asyncio.Lock()
        self._entries: dict[str, _Entry] = {}
        self._loads: dict[str, asyncio.Task] = {}
        self._load_waiters: dict[str, int] = {}
        self._generation = 0
        self._closed = False
        self._shutdown_task: asyncio.Task | None = None
        self._cleaner_task: asyncio.Task | None = None
        self._cleaner_wakeup = asyncio.Event()

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
        skipped: set[int] = set()
        while True:
            load_task = None
            close_task = None
            capacity_eviction = False
            close_generation = -1
            async with self._lock:
                if self._closed:
                    raise ServiceRecoveryRequiredError(
                        "The AgentService session cache is closing"
                    )
                lease, close_task, close_generation = self._cached_action_locked(key)
                if lease is not None:
                    return lease
                if close_task is None:
                    (
                        load_task,
                        close_task,
                        capacity_eviction,
                        close_generation,
                    ) = self._new_load_action_locked(thread, key, skipped)

            if load_task is not None:
                return await self._await_loaded_lease(key, load_task)

            if close_task is not None:
                should_skip = await self._await_close_task(
                    key,
                    close_task,
                    capacity_eviction=capacity_eviction,
                )
                if should_skip:
                    skipped.add(close_generation)
                continue

    async def _await_loaded_lease(
        self, key: str, load_task: asyncio.Task
    ) -> SessionLease:
        try:
            loaded = await asyncio.shield(load_task)
        except BaseException:
            await self._cancel_load_waiter(key, load_task)
            raise
        try:
            async with self._lock:
                if self._closed:
                    raise ServiceRecoveryRequiredError(
                        "The AgentService session cache is closing"
                    )
                if self._entries.get(key) is not loaded or loaded.pending_leases <= 0:
                    raise ServiceRecoveryRequiredError(
                        "The loaded session is no longer available"
                    )
                loaded.pending_leases -= 1
                loaded.last_used = self._clock()
                loaded.idle_since = None
                loaded.next_probe_at = None
                return SessionLease(self, loaded)
        except BaseException:
            await self._drop_pending_lease(loaded)
            raise

    def _cached_action_locked(
        self, key: str
    ) -> tuple[SessionLease | None, asyncio.Task | None, int]:
        entry = self._entries.get(key)
        if entry is None:
            return None, None, -1
        if entry.close_error is not None:
            raise self._quarantined_error(entry)
        if entry.close_task is not None:
            return None, entry.close_task, entry.generation
        entry.pins += 1
        entry.last_used = self._clock()
        entry.idle_since = None
        entry.next_probe_at = None
        return SessionLease(self, entry), None, -1

    def _new_load_action_locked(
        self, thread: Any, key: str, skipped: set[int]
    ) -> tuple[asyncio.Task | None, asyncio.Task | None, bool, int]:
        load_task = self._loads.get(key)
        if load_task is not None:
            self._load_waiters[key] = self._load_waiters.get(key, 0) + 1
            return load_task, None, False, -1
        if self._at_capacity_locked():
            victim = self._oldest_candidate_locked(skipped)
            if victim is not None:
                close_task = self._start_eviction_locked(victim, "probe")
                return None, close_task, True, victim.generation
            closing = self._oldest_closing_locked()
            if closing is not None:
                return None, closing.close_task, True, closing.generation
            raise ServiceBusyError(self._capacity_busy_reason_locked())

        self._generation += 1
        generation = self._generation
        self._load_waiters[key] = 1
        load_task = self._create_task(self._load_and_publish(thread, key, generation))
        self._loads[key] = load_task
        return load_task, None, False, -1

    async def _await_close_task(
        self,
        key: str,
        close_task: asyncio.Task,
        *,
        capacity_eviction: bool,
    ) -> bool:
        try:
            await asyncio.shield(close_task)
        except ServiceBusyError:
            return capacity_eviction
        except ServiceRecoveryRequiredError:
            if capacity_eviction or key in self._entries:
                return capacity_eviction
            raise
        return False

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
                        is_probe = entry.probing
                        load_task = None
                    else:
                        if entry.pins:
                            raise ServiceBusyError(
                                f"Session {key!r} has {entry.pins} active lease(s)"
                            )
                        close_task = self._start_eviction_locked(entry, "fence")
                        is_probe = False
                        load_task = None
            task = close_task or load_task
            if task is None:  # Defensive: the entry/load changed unexpectedly.
                raise RuntimeError(f"Session {key!r} changed during release")
            await asyncio.shield(task)
            if close_task is not None:
                if not is_probe:
                    return True

    async def close_all(self) -> list[Exception]:
        """Stop new loads, close every generation, and return close failures.

        Failed generations remain in ``sessions`` so their partially closed
        resources stay reachable and cannot be replaced by a new factory result.
        """
        async with self._lock:
            if self._shutdown_task is None:
                self._closed = True
                self._cleaner_wakeup.set()
                self._shutdown_task = self._create_task(self._close_all_impl())
            shutdown_task = self._shutdown_task
        return await asyncio.shield(shutdown_task)

    async def _close_all_impl(self) -> list[Exception]:
        cleaner = self._cleaner_task
        if cleaner is not None and cleaner is not asyncio.current_task():
            await asyncio.gather(asyncio.shield(cleaner), return_exceptions=True)

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
                task = entry.close_task or self._start_force_close_locked(entry)
                close_tasks.append(task)

        if close_tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in close_tasks),
                return_exceptions=True,
            )

        async with self._lock:
            forced_tasks = []
            for entry in self._entries.values():
                if entry.close_error is not None:
                    continue
                if entry.close_task is None:
                    entry.close_task = self._create_task(self._close_entry(entry))
                forced_tasks.append(entry.close_task)
        if forced_tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in forced_tasks),
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
            entry = None
            async with self._lock:
                waiters = self._load_waiters.pop(key, 0)
                is_closing = self._closed
                if not is_closing and self._loads.get(key) is asyncio.current_task():
                    self._loads.pop(key, None)
                if is_closing:
                    entry = _Entry(thread, key, generation, session)
                    entry.close_task = self._create_task(self._close_late_entry(entry))
                    close_task = entry.close_task
                else:
                    now = self._clock()
                    entry = _Entry(
                        thread,
                        key,
                        generation,
                        session,
                        pins=waiters,
                        pending_leases=waiters,
                        last_used=now,
                        idle_since=None,
                        next_probe_at=now if waiters == 0 else None,
                    )
                    self._entries[key] = entry
                    self._ensure_cleaner_locked()
                    self._cleaner_wakeup.set()
                    close_task = None
            if close_task is not None:
                await asyncio.shield(close_task)
                raise ServiceRecoveryRequiredError(
                    "Session factory completed after the service began closing"
                )
            return entry
        except _SessionLoadCleanupError as error:
            entry = _Entry(
                thread,
                key,
                generation,
                error.session,
                last_used=self._clock(),
            )
            entry.close_error = error.error
            async with self._lock:
                self._load_waiters.pop(key, None)
                self._entries[key] = entry
                if self._loads.get(key) is asyncio.current_task():
                    self._loads.pop(key, None)
            raise self._quarantined_error(entry) from error.error
        except BaseException:
            async with self._lock:
                self._load_waiters.pop(key, None)
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
            logger.error(
                "Late session %s cleanup failed and is quarantined: %s",
                entry.key,
                error,
            )
            async with self._lock:
                self._entries[entry.key] = entry
            raise self._quarantined_error(entry) from error

    def _at_capacity_locked(self) -> bool:
        limit = self._policy.max_loaded if self._policy is not None else None
        return limit is not None and len(self._entries) + len(self._loads) >= limit

    def _oldest_candidate_locked(self, skipped: set[int]) -> _Entry | None:
        candidates = (
            entry
            for entry in self._entries.values()
            if entry.close_error is None
            and entry.close_task is None
            and entry.pins == 0
            and entry.pending_leases == 0
            and entry.generation not in skipped
        )
        return min(candidates, key=lambda item: item.last_used, default=None)

    def _oldest_closing_locked(self) -> _Entry | None:
        candidates = (
            entry
            for entry in self._entries.values()
            if entry.close_error is None and entry.close_task is not None
        )
        return min(candidates, key=lambda item: item.last_used, default=None)

    def _capacity_busy_reason_locked(self) -> str:
        slots = len(self._entries) + len(self._loads)
        limit = self._policy.max_loaded if self._policy is not None else slots
        return (
            f"Session cache is full ({slots}/{limit} resource slots); "
            "all loaded bindings are busy, loading, or quarantined"
        )

    def _start_eviction_locked(self, entry: _Entry, mode: str) -> asyncio.Task:
        if entry.close_task is None:
            entry.close_task = self._create_task(self._evict_entry(entry, mode))
        return entry.close_task

    def _start_force_close_locked(self, entry: _Entry) -> asyncio.Task:
        if entry.close_task is None:
            entry.close_task = self._create_task(self._close_entry(entry))
        return entry.close_task

    async def _evict_entry(self, entry: _Entry, mode: str) -> None:
        task = asyncio.current_task()
        try:
            if mode == "probe" and self._busy_reason is not None:
                reason = self._busy_reason(entry.thread, entry.session)
                if reason:
                    raise ServiceBusyError(reason)
            reason = self._idle_reason(entry.thread, entry.session)
            if reason:
                raise ServiceBusyError(reason)
        except ServiceBusyError:
            await self._reset_busy_eviction(entry, task)
            raise
        except Exception as error:
            await self._reset_busy_eviction(entry, task)
            reason = f"Session idle state could not be confirmed: {error}"
            logger.warning("%s (%s)", reason, entry.key)
            raise ServiceBusyError(reason) from error

        await self._close_entry(entry)

    async def _reset_busy_eviction(
        self, entry: _Entry, task: asyncio.Task | None
    ) -> None:
        async with self._lock:
            if entry.close_task is task:
                entry.close_task = None
            entry.idle_since = None
            entry.next_probe_at = self._clock() + self._probe_interval()
            self._cleaner_wakeup.set()

    _CLEANER_RETRY_SECONDS = 0.5

    def _ensure_cleaner_locked(self) -> None:
        if (
            self._policy is not None
            and self._policy.idle_timeout is not None
            and self._cleaner_task is None
            and not self._closed
        ):
            self._cleaner_task = self._create_task(self._cleaner_loop())

    async def _cleaner_loop(self) -> None:
        while True:
            async with self._lock:
                if self._closed:
                    return
                self._cleaner_wakeup.clear()
                delay = self._next_cleaner_delay_locked()
            try:
                if delay is None:
                    await self._cleaner_wakeup.wait()
                else:
                    await asyncio.wait_for(
                        self._cleaner_wakeup.wait(), timeout=max(0.0, delay)
                    )
            except TimeoutError:
                pass
            if self._closed:
                return
            try:
                await self._sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Session cache idle sweep failed")
                await asyncio.sleep(self._CLEANER_RETRY_SECONDS)

    def _next_cleaner_delay_locked(self) -> float | None:
        timeout = self._policy.idle_timeout if self._policy is not None else None
        if timeout is None:
            return None
        now = self._clock()
        deadlines = []
        for entry in self._entries.values():
            if (
                entry.close_error is not None
                or entry.close_task is not None
                or entry.pins
                or entry.pending_leases
            ):
                continue
            if entry.idle_since is None:
                deadline = (
                    entry.next_probe_at if entry.next_probe_at is not None else now
                )
            else:
                deadline = entry.idle_since + timeout
                if entry.next_probe_at is not None:
                    deadline = max(deadline, entry.next_probe_at)
            deadlines.append(deadline)
        return max(0.0, min(deadlines) - now) if deadlines else None

    async def _sweep_once(self) -> None:
        timeout = self._policy.idle_timeout if self._policy is not None else None
        if timeout is None:
            return
        while True:
            async with self._lock:
                now = self._clock()
                due = [
                    entry
                    for entry in self._entries.values()
                    if entry.close_error is None
                    and entry.close_task is None
                    and entry.pins == 0
                    and entry.pending_leases == 0
                    and (entry.idle_since is None or now >= entry.idle_since + timeout)
                    and (entry.next_probe_at is None or entry.next_probe_at <= now)
                ]
                if self._closed:
                    return
                if not due:
                    task = next(
                        (
                            entry.close_task
                            for entry in self._entries.values()
                            if entry.probing and entry.close_task is not None
                        ),
                        None,
                    )
                    if task is None:
                        return
                else:
                    entry = min(due, key=lambda item: item.last_used)
                    if entry.idle_since is None:
                        task = self._create_task(self._probe_idle(entry))
                        entry.close_task = task
                        entry.probing = True
                    else:
                        task = self._start_eviction_locked(entry, "probe")
            try:
                await asyncio.shield(task)
            except (ServiceBusyError, ServiceRecoveryRequiredError):
                continue

    async def _probe_idle(self, entry: _Entry) -> None:
        task = asyncio.current_task()
        try:
            if self._busy_reason is not None:
                reason = self._busy_reason(entry.thread, entry.session)
            else:
                # Test/standalone caches may provide only a pure idle probe.
                reason = self._idle_reason(entry.thread, entry.session)
        except Exception as error:
            reason = f"Session idle state could not be confirmed: {error}"
            logger.warning("%s (%s)", reason, entry.key)
        async with self._lock:
            if (
                self._entries.get(entry.key) is not entry
                or entry.close_task is not task
            ):
                return
            entry.close_task = None
            entry.probing = False
            now = self._clock()
            if reason:
                entry.idle_since = None
                entry.next_probe_at = now + self._probe_interval()
            else:
                entry.idle_since = now
                entry.next_probe_at = None
            self._cleaner_wakeup.set()

    def _probe_interval(self) -> float:
        timeout = self._policy.idle_timeout if self._policy is not None else None
        if timeout == 0:
            return self._CLEANER_RETRY_SECONDS
        return min(timeout, 30.0) if timeout is not None else 30.0

    async def _cancel_load_waiter(self, key: str, task: asyncio.Task) -> None:
        cleanup = self._create_task(self._cancel_load_waiter_impl(key, task))
        await asyncio.shield(cleanup)

    async def _cancel_load_waiter_impl(self, key: str, task: asyncio.Task) -> None:
        async with self._lock:
            if self._loads.get(key) is task:
                waiters = self._load_waiters.get(key, 0)
                if waiters <= 1:
                    self._load_waiters.pop(key, None)
                else:
                    self._load_waiters[key] = waiters - 1
                return
            if task.done() and not task.cancelled():
                try:
                    entry = task.result()
                except BaseException:
                    return
                if isinstance(entry, _Entry) and self._entries.get(key) is entry:
                    self._drop_pending_lease_locked(entry)

    async def _drop_pending_lease(self, entry: _Entry) -> None:
        cleanup = self._create_task(self._drop_pending_lease_impl(entry))
        await asyncio.shield(cleanup)

    async def _drop_pending_lease_impl(self, entry: _Entry) -> None:
        async with self._lock:
            self._drop_pending_lease_locked(entry)

    def _drop_pending_lease_locked(self, entry: _Entry) -> None:
        if self._entries.get(entry.key) is not entry or entry.pending_leases <= 0:
            return
        entry.pending_leases -= 1
        entry.pins = max(0, entry.pins - 1)
        if entry.pins == 0:
            entry.idle_since = None
            entry.next_probe_at = self._clock()
            self._cleaner_wakeup.set()

    def activity_finished(self, thread_id: str) -> None:
        """Reset idle time when foreground finalization has fully completed."""
        entry = self._entries.get(thread_id)
        if entry is None:
            return
        now = self._clock()
        entry.last_used = now
        entry.idle_since = None
        entry.next_probe_at = now if entry.pins == 0 else None
        self._cleaner_wakeup.set()

    @staticmethod
    def _observe_task(task: asyncio.Task) -> None:
        """Retrieve exceptions even when every shielded waiter was cancelled."""
        if not task.cancelled():
            task.exception()

    @classmethod
    def _create_task(cls, awaitable: Any) -> asyncio.Task:
        task = asyncio.create_task(cls._deferred(awaitable))
        task.add_done_callback(cls._observe_task)
        return task

    @staticmethod
    async def _deferred(awaitable: Any) -> Any:
        """Let the caller publish its reservation before task work can run."""
        started = False
        try:
            await asyncio.sleep(0)
            started = True
            return await awaitable
        finally:
            if not started and inspect.iscoroutine(awaitable):
                awaitable.close()

    async def _close_entry(self, entry: _Entry) -> None:
        try:
            await self._invoke_close(entry)
        except BaseException as error:
            entry.close_error = error
            logger.error(
                "Session %s cleanup failed and is quarantined: %s",
                entry.key,
                error,
            )
            raise self._quarantined_error(entry) from error
        async with self._lock:
            if self._entries.get(entry.key) is entry:
                self._entries.pop(entry.key, None)
            self._cleaner_wakeup.set()

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
            if entry.pins == 0:
                now = self._clock()
                entry.idle_since = None
                entry.last_used = now
                entry.next_probe_at = now
                self._cleaner_wakeup.set()

    @staticmethod
    def _thread_key(thread: Any) -> str:
        key = thread if isinstance(thread, str) else getattr(thread, "thread_id", None)
        if not isinstance(key, str) or not key:
            raise ValueError("thread must be a non-empty id or have thread_id")
        return key
