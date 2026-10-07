"""Lifecycle contracts for the foreground local service runner."""

import asyncio
from pathlib import Path

import pytest
from unittest.mock import Mock

from msgflux.nn import Agent
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http.client import AgentServiceClient
from msgflux.runtime.service.local.files import process_lock, read_record
from msgflux.runtime.service.local.runner import serve_local_service

_closed_sessions: list[str] = []


def local_factory(runtime_dir: Path) -> AgentService:
    return AgentService(store=SQLiteServiceStore(Path(runtime_dir) / "service.sqlite3"))


def invalid_factory(_runtime_dir: Path) -> object:
    return object()


def resource_factory(runtime_dir: Path) -> AgentService:
    service = AgentService(
        store=SQLiteServiceStore(Path(runtime_dir) / "resource-service.sqlite3")
    )
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="local-resource-test", model=model)
    service.register(
        "agent",
        lambda thread_id: AgentSession(
            agent, on_close=lambda: _closed_sessions.append(thread_id)
        ),
    )
    return service


async def _wait_for_record(runtime_dir: Path):
    for _ in range(500):
        record = read_record(runtime_dir)
        if record is not None:
            return record
        await asyncio.sleep(0.01)
    raise AssertionError("local runner did not publish its ready record")


@pytest.mark.asyncio
async def test_runner_publishes_after_health_and_cleans_its_record_on_cancel(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runner = asyncio.create_task(
        serve_local_service(
            "tests.test_local_service_runner:local_factory",
            runtime_dir=runtime_dir,
            cwd=Path.cwd(),
        )
    )
    record = await _wait_for_record(runtime_dir)

    assert record.factory == "tests.test_local_service_runner:local_factory"
    assert record.cwd == str(Path.cwd().resolve())
    client = AgentServiceClient(record.url, token=record.token)
    try:
        health = await client.health()
    finally:
        await client.aclose()
    assert health.instance_id == record.instance_id

    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner
    assert read_record(runtime_dir) is None

    # The process lock is released on every exit path.
    with process_lock(runtime_dir / "daemon.lock") as lock:
        assert lock is not None


@pytest.mark.asyncio
async def test_runner_rejects_competing_daemon_without_overwriting_identity(tmp_path):
    runtime_dir = tmp_path / "runtime"
    runner = asyncio.create_task(
        serve_local_service(
            "tests.test_local_service_runner:local_factory",
            runtime_dir=runtime_dir,
            cwd=Path.cwd(),
        )
    )
    record = await _wait_for_record(runtime_dir)

    with pytest.raises(RuntimeError, match="already holds the lock"):
        await serve_local_service(
            "tests.test_local_service_runner:invalid_factory",
            runtime_dir=runtime_dir,
            cwd=Path.cwd(),
        )
    assert read_record(runtime_dir) == record

    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner


@pytest.mark.asyncio
async def test_service_session_resources_close_once_on_runner_shutdown(tmp_path):
    _closed_sessions.clear()
    runtime_dir = tmp_path / "runtime"
    runner = asyncio.create_task(
        serve_local_service(
            "tests.test_local_service_runner:resource_factory",
            runtime_dir=runtime_dir,
            cwd=Path.cwd(),
        )
    )
    record = await _wait_for_record(runtime_dir)
    client = AgentServiceClient(record.url, token=record.token)
    try:
        thread = await client.open_thread("agent", thread_id="session-thread")
        await client.snapshot(thread.thread_id)
    finally:
        await client.aclose()

    runner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await runner

    assert _closed_sessions == ["session-thread"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "error"),
    [
        ("tests.test_local_service_runner:missing_factory", AttributeError),
        ("tests.test_local_service_runner:invalid_factory", TypeError),
        ("malformed-factory", ValueError),
    ],
)
async def test_factory_startup_failure_leaves_no_record_and_releases_lock(
    tmp_path, factory, error
):
    runtime_dir = tmp_path / "runtime"

    with pytest.raises(error):
        await serve_local_service(factory, runtime_dir=runtime_dir, cwd=Path.cwd())

    assert read_record(runtime_dir) is None
    with process_lock(runtime_dir / "daemon.lock") as lock:
        assert lock is not None
