"""Repeated runner cancellation must not release a live shutdown's owner lock."""

import asyncio
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("uvicorn")

from msgflux.nn import Agent
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http import factory as http_factory
from msgflux.runtime.service.local import runner
from msgflux.runtime.service.local.files import process_lock, read_record
from msgflux.runtime.service.local.leases import LocalOwnerLease
from msgflux.runtime.service.records import ServiceRecoveryRequiredError
from tests.test_local_service_runner import _wait_for_record


class _Server:
    """Keep transport out of this test while running the actual runner cleanup."""

    def __init__(self, config):
        self.config = config
        self.started = False
        self.should_exit = False

    async def serve(self, *, sockets):
        self.started = True
        await asyncio.Event().wait()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX lifetime locks")
async def test_runner_retains_owner_lease_and_store_until_cancelled_shutdown_settles(
    tmp_path, monkeypatch
):
    import uvicorn

    runtime = tmp_path / "runtime"
    store = SQLiteServiceStore(tmp_path / "service.sqlite3")
    service = AgentService(store=store)
    closing = asyncio.Event()
    release = asyncio.Event()
    close_calls = []

    async def close_session():
        close_calls.append("started")
        closing.set()
        await release.wait()
        # The journal must remain usable until the last callback is complete.
        assert store.thread("retained-thread").agent_id == "agent"
        close_calls.append("finished")

    agent = Agent(name="lease-lifecycle", model=Mock(model_type="chat_completion"))
    service.register(
        "agent", lambda _thread: AgentSession(agent, on_close=close_session)
    )
    await service.open_thread("agent", thread_id="retained-thread")
    await service.snapshot("retained-thread")

    async def make_service(_factory, _runtime):
        return service

    async def verify_health(*_args):
        pass

    monkeypatch.setattr(runner, "_make_service", make_service)
    monkeypatch.setattr(runner, "_verify_health", verify_health)
    monkeypatch.setattr(uvicorn, "Server", _Server)
    monkeypatch.setattr(http_factory, "create_service_app", lambda *_a, **_kw: object())
    serving = asyncio.create_task(
        runner.serve_local_service(
            "tests.fixture:create_service", runtime_dir=runtime, cwd=Path.cwd()
        )
    )
    observer_store = None
    observer_lease = None
    try:
        await _wait_for_record(runtime)
        observer_store = SQLiteServiceStore(tmp_path / "service.sqlite3")
        observer_lease = LocalOwnerLease(AgentService(store=observer_store))
        serving.cancel()
        await asyncio.wait_for(closing.wait(), timeout=5)

        for _ in range(3):
            serving.cancel()
            await asyncio.sleep(0)
            assert not serving.done()
            assert store.thread("retained-thread").agent_id == "agent"
            with pytest.raises(ServiceRecoveryRequiredError, match="still held"):
                with observer_lease.reclaim(service._owner_id):
                    pytest.fail("the owner still has an unfinished close callback")
            with process_lock(runtime / "daemon.lock") as lock:
                assert lock is None

        assert close_calls == ["started"]
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(serving, timeout=5)
        assert close_calls == ["started", "finished"]
        with observer_lease.reclaim(service._owner_id):
            assert observer_store.thread("retained-thread").agent_id == "agent"
        with process_lock(runtime / "daemon.lock") as lock:
            assert lock is not None
        assert read_record(runtime) is None
    finally:
        release.set()
        if not serving.done():
            serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)
        if service._shutdown_task is not None:
            await asyncio.gather(service._shutdown_task, return_exceptions=True)
        if observer_lease is not None:
            observer_lease.close()
        if observer_store is not None:
            observer_store.close()
