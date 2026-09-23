"""Regression checks for retained objects in repeated Agent event streams."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_agent_memory_retention.py"
    spec = importlib.util.spec_from_file_location("agent_memory_retention", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_mode", ["fresh", "shared"])
async def test_repeated_agent_streams_release_live_events_and_hub_state(
    benchmark, thread_mode
):
    result = await benchmark.measure(
        batches=4, iterations_per_batch=2, thread_mode=thread_mode
    )

    assert result["completed_checkpoints"] == 8
    assert result["model_calls"] == 16
    assert result["events_emitted"] >= 8 * 6
    assert result["max_live_events_after_gc"] == 0
    assert result["global_event_hub_watchers_after_gc"] == 0
    # A completed stream has no live run/tool state. This count is a direct
    # retention invariant and catches one retained state per finished thread.
    assert result["global_event_hub_threads_after_gc"] == 0
    assert result["sqlite_bytes"] > 0
    assert len(result["retained_python_bytes_after_gc"]) == 4
    assert all(
        isinstance(value, int) and value >= 0
        for value in result["retained_python_bytes_after_gc"]
    )
    assert isinstance(result["late_retained_slope_bytes_per_batch"], float)


@pytest.mark.asyncio
async def test_memory_benchmark_validates_parameters_without_tracing(benchmark):
    import tracemalloc

    with pytest.raises(ValueError, match="at least 3"):
        await benchmark.measure(batches=2)
    with pytest.raises(ValueError, match="positive integer"):
        await benchmark.measure(batches=3, iterations_per_batch=0)
    assert not tracemalloc.is_tracing()
