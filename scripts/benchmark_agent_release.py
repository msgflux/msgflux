"""Opt-in offline Agent regression benchmark with scripted model responses.

The synthetic workload stresses EventHub and in-memory checkpoints. The Agent
scenario additionally runs stream_events, a tool, and SQLite checkpoints. No
provider, credentials, or network calls are involved. Use --baseline and
--max-regression for a machine-specific release gate.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import math
import statistics
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from unittest.mock import AsyncMock, Mock

from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.event_hub import EventHub
from msgflux.runtime.events import ExecutionEvent


def _percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


async def measure(*, iterations: int = 2000, payload_bytes: int = 512) -> dict:
    """Run a deterministic local workload and return JSON-safe measurements."""
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if type(payload_bytes) is not int or payload_bytes < 1:
        raise ValueError("payload_bytes must be a positive integer")
    if tracemalloc.is_tracing():
        raise RuntimeError("Benchmark requires exclusive use of tracemalloc")

    gc.collect()
    tracemalloc.start()
    hub = EventHub()
    store = InMemoryCheckpointStore()
    payload = "x" * payload_bytes
    event_latencies: list[float] = []
    checkpoint_latencies: list[float] = []
    started = time.perf_counter()
    try:
        watcher = hub.watch("release-benchmark", event_buffer_limit=max(32, iterations))
        await watcher.__aenter__()
        for index in range(iterations):
            event_started = time.perf_counter()
            hub.publish(
                "release-benchmark",
                ExecutionEvent(
                    type="message.delta",
                    timestamp="benchmark",
                    run_id="release-benchmark",
                    data={"delta": payload, "index": index},
                ),
            )
            event_latencies.append(time.perf_counter() - event_started)

            checkpoint_started = time.perf_counter()
            store.save_state(
                "agent:benchmark",
                "release-benchmark",
                "run-1",
                {"status": "running", "step": index, "context": payload},
            )
            checkpoint_latencies.append(time.perf_counter() - checkpoint_started)

        # Drain through the public watcher queue; the projection remains bounded
        # by runtime policy and validates that consumers can keep up.
        received = 0
        while received < iterations:
            await watcher.__anext__()
            received += 1
        await watcher.aclose()
        restored = store.load_state("agent:benchmark", "release-benchmark", "run-1")
        retained, peak = tracemalloc.get_traced_memory()
        elapsed = time.perf_counter() - started
        return {
            "schema_version": 1,
            "workload": "offline_agent_event_checkpoint",
            "iterations": iterations,
            "payload_bytes": payload_bytes,
            "events_published": iterations,
            "events_received": received,
            "checkpoint_step_restored": restored["step"],
            "elapsed_seconds": elapsed,
            "operations_per_second": (iterations * 2 / elapsed) if elapsed else 0,
            "event_publish_seconds_p50": statistics.median(event_latencies),
            "event_publish_seconds_p95": _percentile(event_latencies, 0.95),
            "checkpoint_save_seconds_p50": statistics.median(checkpoint_latencies),
            "checkpoint_save_seconds_p95": _percentile(checkpoint_latencies, 0.95),
            "retained_python_bytes": retained,
            "peak_python_bytes": peak,
        }
    finally:
        tracemalloc.stop()


async def measure_agent(*, iterations: int = 100) -> dict:  # noqa: C901
    """Run the public Agent event stream with scripted model/tool work and SQLite."""
    if type(iterations) is not int or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if tracemalloc.is_tracing():
        raise RuntimeError("Benchmark requires exclusive use of tracemalloc")

    def lookup(query: str) -> str:
        return f"local:{query}"

    def tool_response():
        calls = ToolCallAggregator()
        calls.process(0, "lookup-call", "lookup", '{"query":"status"}')
        response = ModelResponse()
        response.set_response_type("tool_call")
        response.add(calls)
        response.reasoning = None
        return response

    def text_response():
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("status: local:status")
        response.reasoning = None
        return response

    with tempfile.TemporaryDirectory(prefix="msgflux-agent-release-") as temp_dir:
        store = SQLiteCheckpointStore(str(Path(temp_dir) / "checkpoints.sqlite3"))
        model = Mock()
        model.model_type = "chat_completion"
        agent = Agent(
            name="release_benchmark_agent",
            model=model,
            tools=[lookup],
            checkpoint_store=store,
        )
        # Every run consumes precisely these scripted responses. No provider or
        # network client is constructed, while Agent still owns orchestration.
        response_index = 0

        def next_response(*_args, **_kwargs):
            nonlocal response_index
            response_index += 1
            return tool_response() if response_index % 2 else text_response()

        agent.generator.aforward = AsyncMock(side_effect=next_response)
        durations: list[float] = []
        event_count = 0
        tool_events = 0
        started = time.perf_counter()
        database_path = Path(temp_dir) / "checkpoints.sqlite3"
        gc.collect()
        tracemalloc.start()
        try:
            for index in range(iterations):
                scope = ExecutionScope(
                    namespace="release_benchmark_agent",
                    thread_id=f"thread-{index}",
                    run_id=f"run-{index}",
                )
                run_started = time.perf_counter()
                async for event in agent.stream_events(
                    f"Check status {index}", scope=scope
                ):
                    event_count += 1
                    if event.type.startswith("tool."):
                        tool_events += 1
                durations.append(time.perf_counter() - run_started)
            completed = 0
            for index in range(iterations):
                state = store.load_state(
                    "release_benchmark_agent", f"thread-{index}", f"run-{index}"
                )
                completed += state is not None and state.get("status") == "completed"
        finally:
            store.close()
            _retained, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        elapsed = time.perf_counter() - started
        return {
            "iterations": iterations,
            "completed_checkpoints": completed,
            "events_emitted": event_count,
            "tool_events": tool_events,
            "elapsed_seconds": elapsed,
            "runs_per_second": iterations / elapsed if elapsed else 0,
            "run_seconds_p50": statistics.median(durations),
            "run_seconds_p95": _percentile(durations, 0.95),
            "peak_python_bytes": peak,
            "sqlite_bytes": database_path.stat().st_size,
        }


def compare(  # noqa: C901 - validate compatibility and gate in one clear pass
    measurement: dict, baseline: dict, max_regression: float
) -> list[str]:
    """Return gate violations for comparable numeric performance metrics."""
    if not math.isfinite(max_regression) or max_regression < 0:
        raise ValueError("max_regression must be non-negative")
    for key in ("schema_version", "workload", "iterations", "payload_bytes"):
        if measurement.get(key) != baseline.get(key):
            raise ValueError(f"Baseline {key} does not match measurement")
    old_agent = baseline.get("agent_sqlite_scenario")
    new_agent = measurement.get("agent_sqlite_scenario")
    if (old_agent is None) != (new_agent is None):
        raise ValueError("Baseline Agent scenario does not match measurement")
    if old_agent is not None and old_agent.get("iterations") != new_agent.get(
        "iterations"
    ):
        raise ValueError("Baseline Agent iterations do not match measurement")
    keys = (
        "elapsed_seconds",
        "event_publish_seconds_p95",
        "checkpoint_save_seconds_p95",
        "peak_python_bytes",
    )
    agent_keys = ()
    if old_agent is not None:
        agent_keys = (
            "run_seconds_p95",
            "elapsed_seconds",
            "peak_python_bytes",
            "sqlite_bytes",
        )

    def numeric_finite(value) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    for label, data, metric_keys in (
        ("baseline", baseline, keys),
        ("measurement", measurement, keys),
        ("baseline Agent scenario", old_agent, agent_keys),
        ("measurement Agent scenario", new_agent, agent_keys),
    ):
        for key in metric_keys:
            if data is None or not numeric_finite(data.get(key)):
                raise ValueError(f"{label} metric {key} must be a finite number")

    violations = []
    for key in keys:
        before, after = baseline[key], measurement[key]
        if before > 0 and after > before * (1 + max_regression):
            violations.append(key)
    if old_agent is not None:
        for key in agent_keys:
            before, after = old_agent[key], new_agent[key]
            label = f"agent_sqlite_scenario.{key}"
            if before > 0 and after > before * (1 + max_regression):
                violations.append(label)
    return violations


def validate_measurement(measurement: dict) -> None:
    """Fail if the run did not complete its expected event/checkpoint workload."""
    iterations = measurement["iterations"]
    if measurement["events_received"] != iterations:
        raise RuntimeError("Event watcher received an unexpected event count")
    if measurement["checkpoint_step_restored"] != iterations - 1:
        raise RuntimeError("Synthetic checkpoint was not restored at the final step")
    agent = measurement["agent_sqlite_scenario"]
    if agent["completed_checkpoints"] != iterations:
        raise RuntimeError("Agent did not complete every SQLite checkpoint")
    if agent["tool_events"] < iterations * 2:
        raise RuntimeError("Agent emitted too few tool lifecycle events")


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=positive, default=2000)
    parser.add_argument("--payload-bytes", type=positive, default=512)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression", type=float, default=0.20)
    args = parser.parse_args()
    result = asyncio.run(
        measure(iterations=args.iterations, payload_bytes=args.payload_bytes)
    )
    result["agent_sqlite_scenario"] = asyncio.run(
        measure_agent(iterations=args.iterations)
    )
    validate_measurement(result)
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        violations = compare(result, baseline, args.max_regression)
        result["gate"] = {
            "passed": not violations,
            "max_regression": args.max_regression,
            "violations": violations,
        }
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    if args.baseline and result["gate"]["violations"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
