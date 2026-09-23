"""Semantic coverage for the opt-in release benchmark."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def benchmark():
    path = Path(__file__).parents[1] / "scripts" / "benchmark_agent_release.py"
    spec = importlib.util.spec_from_file_location("agent_release_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_offline_workload_exercises_events_and_checkpoints(benchmark):
    result = await benchmark.measure(iterations=12, payload_bytes=32)
    assert result["workload"] == "offline_agent_event_checkpoint"
    assert result["events_published"] == result["events_received"] == 12
    assert result["checkpoint_step_restored"] == 11
    assert result["peak_python_bytes"] >= result["retained_python_bytes"] > 0
    assert result["elapsed_seconds"] > 0
    assert result["event_publish_seconds_p95"] >= 0
    assert result["checkpoint_save_seconds_p95"] >= 0


@pytest.mark.asyncio
async def test_measurement_rejects_bad_inputs(benchmark):
    with pytest.raises(ValueError, match="iterations"):
        await benchmark.measure(iterations=0)
    with pytest.raises(ValueError, match="payload_bytes"):
        await benchmark.measure(payload_bytes=0)


def test_configurable_regression_gate(benchmark):
    identity = {
        "schema_version": 1,
        "workload": "offline_agent_event_checkpoint",
        "iterations": 4,
        "payload_bytes": 8,
    }
    metrics = {
        "elapsed_seconds": 1.0,
        "event_publish_seconds_p95": 0.1,
        "checkpoint_save_seconds_p95": 0.1,
        "peak_python_bytes": 100,
    }
    baseline = {**identity, **metrics}
    current = {**identity, **metrics, "elapsed_seconds": 1.3, "peak_python_bytes": 110}
    assert benchmark.compare(current, baseline, 0.2) == ["elapsed_seconds"]
    assert benchmark.compare(current, baseline, 0.3) == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 2),
        ("workload", "other"),
        ("iterations", 5),
        ("payload_bytes", 9),
    ],
)
def test_regression_gate_rejects_incompatible_baseline(benchmark, key, value):
    identity = {
        "schema_version": 1,
        "workload": "offline_agent_event_checkpoint",
        "iterations": 4,
        "payload_bytes": 8,
    }
    measurement = {**identity, "elapsed_seconds": 1.0}
    baseline = {**identity, "elapsed_seconds": 1.0, key: value}
    with pytest.raises(ValueError, match="does not match"):
        benchmark.compare(measurement, baseline, 0.2)


@pytest.mark.parametrize(
    "key,value",
    [
        ("elapsed_seconds", None),
        ("peak_python_bytes", float("nan")),
        ("event_publish_seconds_p95", True),
    ],
)
def test_regression_gate_requires_finite_numeric_metrics(benchmark, key, value):
    identity = {
        "schema_version": 1,
        "workload": "offline_agent_event_checkpoint",
        "iterations": 4,
        "payload_bytes": 8,
        "elapsed_seconds": 1.0,
        "event_publish_seconds_p95": 0.1,
        "checkpoint_save_seconds_p95": 0.1,
        "peak_python_bytes": 100,
    }
    measurement = {**identity}
    baseline = {**identity, key: value}
    with pytest.raises(ValueError, match="finite number"):
        benchmark.compare(measurement, baseline, 0.2)


def test_regression_gate_requires_all_agent_metrics(benchmark):
    identity = {
        "schema_version": 1,
        "workload": "offline_agent_event_checkpoint",
        "iterations": 1,
        "payload_bytes": 8,
        "elapsed_seconds": 1.0,
        "event_publish_seconds_p95": 0.1,
        "checkpoint_save_seconds_p95": 0.1,
        "peak_python_bytes": 100,
    }
    agent = {
        "iterations": 1,
        "run_seconds_p95": 0.1,
        "elapsed_seconds": 0.1,
        "peak_python_bytes": 10,
        "sqlite_bytes": 1024,
    }
    current = {**identity, "agent_sqlite_scenario": agent}
    baseline = {**identity, "agent_sqlite_scenario": {**agent}}
    del baseline["agent_sqlite_scenario"]["sqlite_bytes"]
    with pytest.raises(ValueError, match=r"sqlite_bytes.*finite number"):
        benchmark.compare(current, baseline, 0.2)


def test_measurement_validation_fails_closed_on_functional_mismatch(benchmark):
    result = {
        "iterations": 3,
        "events_received": 3,
        "checkpoint_step_restored": 2,
        "agent_sqlite_scenario": {"completed_checkpoints": 3, "tool_events": 6},
    }
    benchmark.validate_measurement(result)
    result["agent_sqlite_scenario"]["tool_events"] = 5
    with pytest.raises(RuntimeError, match="too few tool"):
        benchmark.validate_measurement(result)


@pytest.mark.asyncio
async def test_real_agent_scripted_tool_stream_and_sqlite_checkpoints(benchmark):
    result = await benchmark.measure_agent(iterations=2)
    assert result["completed_checkpoints"] == 2
    assert result["events_emitted"] > 0
    assert result["tool_events"] >= 4
    assert result["run_seconds_p95"] >= 0
    assert result["peak_python_bytes"] > 0
    assert result["sqlite_bytes"] > 0
