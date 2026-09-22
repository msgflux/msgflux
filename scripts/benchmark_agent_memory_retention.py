"""Measure Python memory retention across repeated offline Agent event streams.

Run with ``uv run python scripts/benchmark_agent_memory_retention.py``. The
script uses a scripted model, a local tool and SQLite checkpoints; it makes no
provider, credential, or network calls. Measurements are sampled only after
each batch has completed and garbage collection has run.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import statistics
import sys
import tempfile
import tracemalloc
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.event_hub import get_event_hub
from msgflux.runtime.events import ExecutionEvent


def _response_with_tool() -> ModelResponse:
    calls = ToolCallAggregator()
    calls.process(0, "memory-call", "lookup", '{"query":"status"}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    response.reasoning = None
    return response


def _text_response() -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add("status: local")
    response.reasoning = None
    return response


def _slope(samples: list[int]) -> float:
    """Least-squares bytes-per-batch slope, without requiring numpy."""
    count = len(samples)
    x_mean = (count - 1) / 2
    y_mean = statistics.fmean(samples)
    denominator = sum((index - x_mean) ** 2 for index in range(count))
    numerator = sum(
        (index - x_mean) * (value - y_mean) for index, value in enumerate(samples)
    )
    return numerator / denominator


async def measure(  # noqa: C901 - workload and lifecycle intentionally stay together
    *, batches: int = 8, iterations_per_batch: int = 10, thread_mode: str = "fresh"
) -> dict:
    """Return post-GC retained-memory and live-event samples for a repeated workload."""
    if type(batches) is not int or batches < 3:
        raise ValueError("batches must be an integer of at least 3")
    if type(iterations_per_batch) is not int or iterations_per_batch < 1:
        raise ValueError("iterations_per_batch must be a positive integer")
    if thread_mode not in {"fresh", "shared"}:
        raise ValueError("thread_mode must be 'fresh' or 'shared'")
    if tracemalloc.is_tracing():
        raise RuntimeError("Benchmark requires exclusive use of tracemalloc")

    def lookup(query: str) -> str:
        return f"local:{query}"

    thread_prefix = f"memory-retention-{uuid.uuid4().hex}"

    async def run_batch(database_path: Path, batch: int) -> tuple[int, int, int, int]:
        store = SQLiteCheckpointStore(str(database_path))
        model = Mock()
        model.model_type = "chat_completion"
        agent = Agent(
            name="memory_retention_agent",
            model=model,
            tools=[lookup],
            checkpoint_store=store,
        )
        calls = 0

        def scripted_response(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return _response_with_tool() if calls % 2 else _text_response()

        agent.generator.aforward = AsyncMock(side_effect=scripted_response)
        completed = emitted = 0
        try:
            for offset in range(iterations_per_batch):
                index = batch * iterations_per_batch + offset
                scope = ExecutionScope(
                    namespace="memory_retention_agent",
                    thread_id=(
                        f"{thread_prefix}-{index}"
                        if thread_mode == "fresh"
                        else f"{thread_prefix}-shared"
                    ),
                    run_id=f"{thread_prefix}-run-{index}",
                )
                async for _event in agent.stream_events(
                    f"Check status {index}", scope=scope
                ):
                    emitted += 1
            for offset in range(iterations_per_batch):
                index = batch * iterations_per_batch + offset
                checkpoint = store.load_state(
                    "memory_retention_agent",
                    f"{thread_prefix}-{index}"
                    if thread_mode == "fresh"
                    else f"{thread_prefix}-shared",
                    f"{thread_prefix}-run-{index}",
                )
                completed += (
                    checkpoint is not None and checkpoint.get("status") == "completed"
                )
        finally:
            store.close()
            sqlite_bytes = database_path.stat().st_size
            # Explicitly release the Agent, model mock, and all per-run locals
            # before sampling: the samples represent process-level survivors.
            del agent, model, store
        return completed, emitted, calls, sqlite_bytes

    retained_samples: list[int] = []
    live_event_samples: list[int] = []
    completed = emitted = model_calls = sqlite_bytes = 0
    with tempfile.TemporaryDirectory(prefix="msgflux-agent-retention-") as temp_dir:
        if gc.isenabled():
            gc.collect()
        tracemalloc.start()
        first_snapshot = None
        last_snapshot = None
        try:
            for batch in range(batches):
                counts = await run_batch(
                    Path(temp_dir) / f"checkpoints-{batch}.sqlite3", batch
                )
                completed += counts[0]
                emitted += counts[1]
                model_calls += counts[2]
                sqlite_bytes += counts[3]
                gc.collect()
                retained, _peak = tracemalloc.get_traced_memory()
                retained_samples.append(retained)
                live_event_samples.append(
                    sum(
                        isinstance(obj, ExecutionEvent)
                        and isinstance(obj.run_id, str)
                        and obj.run_id.startswith(f"{thread_prefix}-run-")
                        for obj in gc.get_objects()
                    )
                )
                snapshot = tracemalloc.take_snapshot()
                if first_snapshot is None:
                    first_snapshot = snapshot
                last_snapshot = snapshot
        finally:
            tracemalloc.stop()

    # The initial sample includes one-time lazy initialization. Compare fitted
    # slopes after warm-up; report full samples so a reviewer can inspect noise.
    warm_samples = retained_samples[1:]
    window = max(2, len(warm_samples) // 2)
    early_slope = _slope(warm_samples[:window])
    late_slope = _slope(warm_samples[-window:])
    growth_by_file = last_snapshot.compare_to(first_snapshot, "filename")[:8]
    return {
        "batches": batches,
        "iterations_per_batch": iterations_per_batch,
        "thread_mode": thread_mode,
        "completed_checkpoints": completed,
        "events_emitted": emitted,
        "model_calls": model_calls,
        # SQLite growth is expected durable state and is reported separately
        # from Python heap retention.
        "sqlite_bytes": sqlite_bytes,
        "retained_python_bytes_after_gc": retained_samples,
        "live_execution_events_after_gc": live_event_samples,
        "early_retained_slope_bytes_per_batch": early_slope,
        "late_retained_slope_bytes_per_batch": late_slope,
        "max_live_events_after_gc": max(live_event_samples),
        "global_event_hub_threads_after_gc": sum(
            thread_id.startswith(thread_prefix)
            for thread_id in get_event_hub()._threads
        ),
        "global_event_hub_watchers_after_gc": sum(
            len(watchers)
            for thread_id, watchers in get_event_hub()._watchers.items()
            if thread_id.startswith(thread_prefix)
        ),
        "global_event_hub_live_state_after_gc": {
            thread_id: {
                "runs": [str(key) for key in state.runs],
                "tools": len(state.tools),
                "background_tasks": len(state.background_tasks),
            }
            for thread_id, state in get_event_hub()._threads.items()
            if thread_id.startswith(thread_prefix)
        },
        "largest_retained_growth_by_file": [
            {
                "file": str(stat.traceback[0].filename),
                "bytes": stat.size_diff,
                "blocks": stat.count_diff,
            }
            for stat in growth_by_file
        ],
    }


def positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=positive, default=8)
    parser.add_argument("--iterations-per-batch", type=positive, default=10)
    parser.add_argument("--thread-mode", choices=("fresh", "shared"), default="fresh")
    args = parser.parse_args()
    if args.batches < 3:
        parser.error("--batches must be at least 3")
    sys.stdout.write(
        json.dumps(
            asyncio.run(
                measure(
                    batches=args.batches,
                    iterations_per_batch=args.iterations_per_batch,
                    thread_mode=args.thread_mode,
                )
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
