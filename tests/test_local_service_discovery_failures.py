from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from msgflux.runtime.service.local import discovery


class _Child:
    def __init__(self):
        self.returncode = None
        self.terminate_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = 0


@pytest.mark.asyncio
async def test_connect_preserves_startup_error_when_cleanup_fails_after_cancellation(
    tmp_path, monkeypatch
):
    child = _Child()
    startup_error = TimeoutError("primary startup timeout")
    cleanup_error = OSError("secondary child cleanup failure")
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()

    monkeypatch.setattr(discovery, "_start_child", lambda *_args, **_kwargs: child)

    async def fail_startup(*_args, **_kwargs):
        raise startup_error

    async def fail_cleanup(_child):
        cleanup_started.set()
        await allow_cleanup.wait()
        raise cleanup_error

    monkeypatch.setattr(discovery, "_wait_for_child", fail_startup)
    monkeypatch.setattr(discovery, "_stop_owned_child", fail_cleanup)
    operation = asyncio.create_task(
        discovery.connect_local_service(
            "tests.test_local_service_runner:local_factory",
            runtime_dir=tmp_path / "runtime",
            cwd=Path.cwd(),
        )
    )

    await asyncio.wait_for(cleanup_started.wait(), timeout=2)
    operation.cancel()
    await asyncio.sleep(0)
    operation.cancel()
    allow_cleanup.set()
    with pytest.raises(TimeoutError) as raised:
        await operation

    assert raised.value is startup_error
    assert any(
        "Local daemon child cleanup failed: secondary child cleanup failure" in note
        for note in getattr(raised.value, "__notes__", ())
    )


@pytest.mark.asyncio
async def test_connect_retains_child_ownership_after_log_close_error(
    tmp_path, monkeypatch, caplog
):
    child = _Child()
    startup_error = TimeoutError("primary startup timeout")
    close_error = OSError("injected log descriptor close failure")
    log_descriptors = set()
    closed_log_descriptors = []
    original_open = discovery.os.open
    original_close = discovery.os.close

    def track_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if Path(path).name == "daemon.log":
            log_descriptors.add(descriptor)
        return descriptor

    def close_log_with_error(descriptor):
        original_close(descriptor)
        if descriptor in log_descriptors:
            log_descriptors.remove(descriptor)
            closed_log_descriptors.append(descriptor)
            raise close_error

    monkeypatch.setattr(discovery.os, "open", track_open)
    monkeypatch.setattr(discovery.os, "close", close_log_with_error)
    monkeypatch.setattr(discovery.subprocess, "Popen", lambda *_a, **_kw: child)

    async def fail_startup(process, *_args, **_kwargs):
        assert process is child
        raise startup_error

    monkeypatch.setattr(discovery, "_wait_for_child", fail_startup)

    with pytest.raises(TimeoutError) as raised:
        await discovery.connect_local_service(
            "tests.test_local_service_runner:local_factory",
            runtime_dir=tmp_path / "runtime",
            cwd=Path.cwd(),
        )

    assert raised.value is startup_error
    assert child.terminate_calls == 1
    assert len(closed_log_descriptors) == 1
    assert any(
        "Local daemon log descriptor cleanup failed" in record.message
        and "injected log descriptor close failure" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_connect_preserves_spawn_error_when_log_close_also_fails(
    tmp_path, monkeypatch
):
    spawn_error = OSError("primary process spawn failure")
    close_error = OSError("secondary log descriptor close failure")
    log_descriptors = set()
    original_open = discovery.os.open
    original_close = discovery.os.close

    def track_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if Path(path).name == "daemon.log":
            log_descriptors.add(descriptor)
        return descriptor

    def close_log_with_error(descriptor):
        original_close(descriptor)
        if descriptor in log_descriptors:
            log_descriptors.remove(descriptor)
            raise close_error

    monkeypatch.setattr(discovery.os, "open", track_open)
    monkeypatch.setattr(discovery.os, "close", close_log_with_error)

    def fail_spawn(*_args, **_kwargs):
        raise spawn_error

    monkeypatch.setattr(discovery.subprocess, "Popen", fail_spawn)

    with pytest.raises(OSError) as raised:
        await discovery.connect_local_service(
            "tests.test_local_service_runner:local_factory",
            runtime_dir=tmp_path / "runtime",
            cwd=Path.cwd(),
        )

    assert raised.value is spawn_error
    assert any(
        "Local daemon log descriptor cleanup failed: secondary log descriptor close failure"
        in note
        for note in getattr(raised.value, "__notes__", ())
    )
