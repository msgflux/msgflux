"""Offline retention benchmark; no models, credentials, disk spooling or shell tools.

Uses private runtime primitives intentionally to isolate retention layers.
Run from the repository with ``uv run python scripts/benchmark_event_memory.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import sys
import time
import tracemalloc

from msgflux.exceptions import EventBufferOverflowError
from msgflux.runtime.event_buffer import _EventBuffer
from msgflux.runtime.event_hub import EventHub
from msgflux.runtime.events import ExecutionEvent


def event(kind: str, text: str = "") -> ExecutionEvent:
    return ExecutionEvent(
        type=kind,
        timestamp="benchmark",
        run_id="run",
        data={"result" if kind == "tool.end" else "delta": text},
    )


async def measure(  # noqa: C901 - keep allocation and cleanup in one tracing interval
    scenario: str, *, events: int, payload_bytes: int, limit: int
) -> dict:
    """Measure one isolated scenario; no concurrent tracemalloc users supported."""
    if scenario not in {
        "unlimited_queue",
        "bounded_queue",
        "oversized_event",
        "live_projection",
        "reconnect_snapshot",
    }:
        raise ValueError("Unknown scenario")
    if any(
        type(value) is not int or value <= 0 for value in (events, payload_bytes, limit)
    ):
        raise ValueError("events, payload_bytes and limit must be positive integers")
    if tracemalloc.is_tracing():
        raise RuntimeError("Benchmark requires exclusive use of tracemalloc")
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    try:
        buffer = None
        hub = None
        watcher = None
        snapshot = None
        overflow = False
        snapshot_chars = 0
        pending = 0
        count = 1 if scenario == "oversized_event" else events
        size = (
            events * payload_bytes if scenario == "oversized_event" else payload_bytes
        )
        if scenario in {"live_projection", "reconnect_snapshot"}:
            hub = EventHub()
            hub.publish("thread", event("run.start"))
        else:
            buffer = _EventBuffer(None if scenario == "unlimited_queue" else limit)
        for _ in range(count):
            # Decode creates a fresh string even for identical content; no list
            # of payloads is kept by the benchmark itself.
            item = event("message.delta" if hub else "tool.end", (b"x" * size).decode())
            if hub is not None:
                hub.publish("thread", item)
            else:
                buffer.put(item)
        del item
        if scenario == "reconnect_snapshot":
            watcher = hub.watch("thread", event_buffer_limit=limit)
            await watcher.__aenter__()
            snapshot = watcher.snapshot
            snapshot_chars = len(snapshot.streaming_message)
        if buffer is not None:
            pending = len(buffer._items)
        retained, peak = tracemalloc.get_traced_memory()
        elapsed = time.perf_counter() - started
        if buffer is not None:
            buffer.close()
            try:
                while await buffer.get() is not None:
                    pass
            except EventBufferOverflowError:
                overflow = True
        if watcher is not None:
            await watcher.aclose()
        if hub is not None:
            hub.publish("thread", event("run.end"))
        # Release the consumer-owned snapshot as well as runtime ownership.
        buffer = hub = watcher = snapshot = None
        await asyncio.sleep(0)  # let the single coalesced wake release its buffer
        gc.collect()
        after_cleanup, _ = tracemalloc.get_traced_memory()
        return {
            "scenario": scenario,
            "events": count,
            "emitted_payload_bytes": count * size,
            "pending_events": pending,
            "overflow": overflow,
            "snapshot_chars": snapshot_chars,
            "retained_python_bytes": retained,
            "peak_python_bytes": peak,
            "after_cleanup_python_bytes": after_cleanup,
            "publish_seconds": elapsed,
        }
    finally:
        tracemalloc.stop()


async def run(args: argparse.Namespace) -> list[dict]:
    return [
        await measure(
            scenario,
            events=args.events,
            payload_bytes=args.payload_bytes,
            limit=args.limit,
        )
        for scenario in (
            "unlimited_queue",
            "bounded_queue",
            "oversized_event",
            "live_projection",
            "reconnect_snapshot",
        )
    ]


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=positive, default=20000)
    parser.add_argument("--payload-bytes", type=positive, default=1024)
    parser.add_argument("--limit", type=positive, default=256)
    args = parser.parse_args()
    sys.stdout.write(
        json.dumps({"measurements": asyncio.run(run(args))}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
