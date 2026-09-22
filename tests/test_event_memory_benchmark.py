"""Semantic checks for memory measurements, without machine-specific thresholds."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_event_memory.py"
    spec = importlib.util.spec_from_file_location("event_memory_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario,pending,overflow,snapshot",
    [
        ("unlimited_queue", 20, False, 0),
        ("bounded_queue", 0, True, 0),
        ("oversized_event", 1, False, 0),
        ("live_projection", 0, False, 0),
        ("reconnect_snapshot", 0, False, 1280),
    ],
)
async def test_measurement_exercises_distinct_retention_layers(
    benchmark, scenario, pending, overflow, snapshot
):
    result = await benchmark.measure(scenario, events=20, payload_bytes=64, limit=4)
    assert result["emitted_payload_bytes"] == 1280
    assert result["events"] == (1 if scenario == "oversized_event" else 20)
    assert result["pending_events"] == pending
    assert result["overflow"] is overflow
    assert result["snapshot_chars"] == snapshot
    assert result["peak_python_bytes"] >= result["retained_python_bytes"] > 0
    assert result["publish_seconds"] >= 0


@pytest.mark.asyncio
async def test_measurement_rejects_invalid_inputs_without_starting_tracing(benchmark):
    with pytest.raises(ValueError, match="positive integers"):
        await benchmark.measure("bounded_queue", events=0, payload_bytes=64, limit=4)
    with pytest.raises(ValueError, match="Unknown scenario"):
        await benchmark.measure("unknown", events=1, payload_bytes=64, limit=4)


@pytest.mark.asyncio
async def test_measurement_preserves_external_tracing_session(benchmark):
    import tracemalloc

    tracemalloc.start()
    try:
        with pytest.raises(RuntimeError, match="exclusive"):
            await benchmark.measure(
                "bounded_queue", events=1, payload_bytes=64, limit=4
            )
        assert tracemalloc.is_tracing()
    finally:
        tracemalloc.stop()
