"""One process-wide heartbeat thread for active background task leases."""

from __future__ import annotations

import time
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING

from msgflux.logger import logger

if TYPE_CHECKING:
    from msgflux.tasks.handle import TaskHandle


class TaskLeaseHeartbeats:
    _lock = Lock()
    _wake = Event()
    _active: dict[int, tuple[TaskHandle, float, float]] = {}
    _thread: Thread | None = None

    @classmethod
    def register(cls, handle: TaskHandle, *, lease_seconds: float) -> None:
        interval = lease_seconds / 3
        with cls._lock:
            cls._active[id(handle)] = (
                handle,
                lease_seconds,
                time.monotonic() + interval,
            )
            if cls._thread is None:
                cls._thread = Thread(
                    target=cls._run,
                    name="msgflux-task-lease-heartbeat",
                    daemon=True,
                )
                cls._thread.start()
            cls._wake.set()

    @classmethod
    def unregister(cls, handle: TaskHandle) -> None:
        with cls._lock:
            cls._active.pop(id(handle), None)
            cls._wake.set()

    @classmethod
    def _run(cls) -> None:
        while True:
            with cls._lock:
                if not cls._active:
                    cls._thread = None
                    cls._wake.clear()
                    return
                now = time.monotonic()
                due = [
                    (key, handle, lease_seconds)
                    for key, (handle, lease_seconds, next_at) in cls._active.items()
                    if next_at <= now
                ]
                if not due:
                    next_at = min(item[2] for item in cls._active.values())
                    timeout = max(0.001, next_at - now)
                    cls._wake.clear()
                else:
                    timeout = 0.0
            if not due:
                cls._wake.wait(timeout)
                continue
            for key, handle, lease_seconds in due:
                try:
                    renewed = handle.renew_worker(lease_seconds=lease_seconds)
                except Exception:
                    logger.exception("Background task lease heartbeat failed")
                    renewed = True  # Retry on the next tick until the lease expires.
                with cls._lock:
                    current = cls._active.get(key)
                    if current is None or current[0] is not handle:
                        continue
                    if not renewed:
                        cls._active.pop(key, None)
                    else:
                        cls._active[key] = (
                            handle,
                            lease_seconds,
                            time.monotonic() + lease_seconds / 3,
                        )
