"""Monotonic timing helpers for model requests."""

import threading
import time
from collections.abc import Callable
from typing import Literal


class ModelRequestTimer:
    """Measure end-to-end latency and the first observable stream output."""

    def __init__(
        self,
        *,
        source: Literal["provider", "cache"] = "provider",
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.source = source
        self._clock_ns = clock_ns
        self._started_ns = clock_ns()
        self._first_output_ns: int | None = None
        self._finished_ns: int | None = None
        self._lock = threading.Lock()

    def mark_first_output(self) -> None:
        """Record the first non-empty output, once."""
        with self._lock:
            if self._first_output_ns is None and self._finished_ns is None:
                self._first_output_ns = self._clock_ns()

    def finish(self) -> dict[str, float | str]:
        """Return a stable metadata snapshot, finalizing the timer if needed."""
        with self._lock:
            if self._finished_ns is None:
                self._finished_ns = self._clock_ns()
            finished_ns = self._finished_ns
            first_output_ns = self._first_output_ns

        timing: dict[str, float | str] = {
            "source": self.source,
            "latency_ms": (finished_ns - self._started_ns) / 1_000_000,
        }
        if first_output_ns is not None:
            timing["ttft_ms"] = (first_output_ns - self._started_ns) / 1_000_000
        return timing
