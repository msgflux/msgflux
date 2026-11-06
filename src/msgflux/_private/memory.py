"""Return freed heap pages to the OS after agent runs.

Long agent runs allocate and free many short-lived buffers (model payloads,
checkpoint JSON). glibc keeps freed arenas mapped, so RSS stays at its peak
long after the run ends; with dozens of subagents per process that idle
footprint adds up. ``malloc_trim`` hands those pages back.

The trim is process-wide, so it is throttled: concurrent subagents finishing
together trigger at most one trim per :data:`TRIM_INTERVAL_SECONDS`. Set
``MSGFLUX_MALLOC_TRIM=0`` to disable it. It is a no-op outside glibc.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from threading import Lock, Thread
from typing import Any, Optional

TRIM_INTERVAL_SECONDS = 1.0

_lock = Lock()
_last_trim = float("-inf")
_malloc_trim: Optional[Any] = None
_resolved = False


def _resolve() -> Optional[Any]:
    global _malloc_trim, _resolved  # noqa: PLW0603
    if not _resolved:
        _resolved = True
        if sys.platform.startswith("linux") and os.environ.get(
            "MSGFLUX_MALLOC_TRIM", "1"
        ).strip().lower() not in {"0", "false", "no", "off"}:
            try:
                libc = ctypes.CDLL(None)
                trim = libc.malloc_trim
            except (OSError, AttributeError):
                trim = None
            if trim is not None:
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
            _malloc_trim = trim
    return _malloc_trim


def _run_trim(trim: Any) -> None:
    try:
        trim(0)
    finally:
        _lock.release()


def release_freed_memory(*, background: bool = True) -> bool:
    """Trim the C heap unless a trim ran recently; return if one started.

    A trim walks the whole heap (tens of ms on a large fragmented one), so by
    default it runs on a daemon thread; ctypes drops the GIL for the call and
    the caller (often an event loop) pays only the throttle check.
    """
    global _last_trim  # noqa: PLW0603
    trim = _resolve()
    if trim is None:
        return False
    now = time.monotonic()
    if now - _last_trim < TRIM_INTERVAL_SECONDS or not _lock.acquire(blocking=False):
        return False
    if now - _last_trim < TRIM_INTERVAL_SECONDS:
        _lock.release()
        return False
    _last_trim = now
    if not background:
        _run_trim(trim)
        return True
    try:
        Thread(
            target=_run_trim, args=(trim,), name="msgflux-malloc-trim", daemon=True
        ).start()
    except RuntimeError:
        _lock.release()
        return False
    return True
