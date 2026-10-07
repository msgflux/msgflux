"""Discover or start the per-user local AgentService daemon."""

from __future__ import annotations

import asyncio
import math
import os
import stat
import subprocess
import sys
from pathlib import Path

from msgflux.runtime.service import (
    ServiceConflictError,
    ServiceRecoveryRequiredError,
)
from msgflux.runtime.service.http.client import AgentServiceClient
from msgflux.runtime.service.local.files import (
    prepare_runtime_dir,
    process_lock,
    read_record,
    remove_record,
)
from msgflux.runtime.service.local.records import LocalServiceRecord

_reapers: set[asyncio.Task[None]] = set()
_cleanups: set[asyncio.Task[None]] = set()
_STARTUP_LOCK = "startup.lock"
_DAEMON_LOCK = "daemon.lock"
_LOG_FILE = "daemon.log"


async def connect_local_service(
    factory: str,
    *,
    runtime_dir: Path | None = None,
    cwd: Path | None = None,
    startup_timeout: float = 30,
) -> AgentServiceClient:
    """Connect to a matching local daemon, starting one when no owner exists.

    The daemon is independent of this client and remains running after the
    returned HTTP client is closed.
    """
    _validate_factory(factory)
    if (
        isinstance(startup_timeout, bool)
        or not isinstance(startup_timeout, (int, float))
        or not math.isfinite(startup_timeout)
        or startup_timeout <= 0
    ):
        raise ValueError("startup_timeout must be a finite positive number")
    selected_runtime = prepare_runtime_dir(
        Path(runtime_dir)
        if runtime_dir is not None
        else Path.home() / ".msgflux" / "runtime"
    )
    selected_cwd = (Path(cwd) if cwd is not None else Path.cwd()).expanduser().resolve()
    if not selected_cwd.is_dir():
        raise NotADirectoryError("cwd must name an existing directory")
    deadline = asyncio.get_running_loop().time() + startup_timeout
    async with _StartupLock(selected_runtime, deadline):
        client = await _reuse_running(selected_runtime, factory, selected_cwd, deadline)
        if client is not None:
            return client

        process = _start_child(selected_runtime, factory, selected_cwd)
        try:
            return await _wait_for_child(
                process,
                selected_runtime,
                factory,
                selected_cwd,
                deadline,
            )
        except BaseException:
            cleanup = asyncio.create_task(_stop_owned_child(process))
            _cleanups.add(cleanup)
            cleanup.add_done_callback(_cleanups.discard)
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
            cleanup.result()
            raise


async def _reuse_running(
    runtime_dir: Path, factory: str, cwd: Path, deadline: float
) -> AgentServiceClient | None:
    while True:
        record = read_record(runtime_dir)
        if record is not None:
            client = await _probe(record)
            if client is not None:
                try:
                    _require_match(record, factory, cwd)
                except BaseException:
                    await client.aclose()
                    raise
                return client
            if not _daemon_is_stopped(runtime_dir):
                raise ServiceRecoveryRequiredError(
                    "The local daemon owns its lifetime lock but failed "
                    "its identity health check"
                )
            remove_record(runtime_dir, record.instance_id)
            return None
        if _daemon_is_stopped(runtime_dir):
            return None
        # Foreground startup or old-owner shutdown can hold the lock without
        # publishing metadata. Wait without starting a second process.
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for the local daemon owner")
        await asyncio.sleep(min(0.05, remaining))


class _StartupLock:
    """Async context wrapper for polling a process lock without blocking asyncio."""

    def __init__(self, runtime_dir: Path, deadline: float) -> None:
        self.runtime_dir = runtime_dir
        self.deadline = deadline
        self._context = None

    async def __aenter__(self):
        while True:
            context = process_lock(self.runtime_dir / _STARTUP_LOCK)
            handle = context.__enter__()
            if handle is not None:
                self._context = context
                return handle
            context.__exit__(None, None, None)
            remaining = self.deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for local service startup lock")
            await asyncio.sleep(min(0.05, remaining))

    async def __aexit__(self, exc_type, exc, tb):
        return self._context.__exit__(exc_type, exc, tb)


async def _probe(record: LocalServiceRecord) -> AgentServiceClient | None:
    client = AgentServiceClient(record.url, token=record.token, timeout=0.75)
    try:
        health = await asyncio.wait_for(client.health(), timeout=1.0)
        if health.instance_id != record.instance_id or health.version != 1:
            await client.aclose()
            return None
        await client.aclose()
        return AgentServiceClient(record.url, token=record.token)
    except asyncio.CancelledError:
        await client.aclose()
        raise
    except Exception:
        await client.aclose()
        return None


def _daemon_is_stopped(runtime_dir: Path) -> bool:
    with process_lock(runtime_dir / _DAEMON_LOCK) as handle:
        return handle is not None


def _start_child(runtime_dir: Path, factory: str, cwd: Path) -> subprocess.Popen[bytes]:
    log_path = runtime_dir / _LOG_FILE
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(log_path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise PermissionError("Local daemon log must be an owned regular file")
        os.fchmod(descriptor, 0o600)
        # argv is passed directly; no shell interprets configured values.
        return subprocess.Popen(  # noqa: S603
            [
                sys.executable,
                "-m",
                "msgflux.runtime.service.local.cli",
                "--factory",
                factory,
                "--runtime-dir",
                str(runtime_dir),
                "--cwd",
                str(cwd),
            ],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
    finally:
        os.close(descriptor)


async def _wait_for_child(
    process: subprocess.Popen[bytes],
    runtime_dir: Path,
    factory: str,
    cwd: Path,
    deadline: float,
) -> AgentServiceClient:
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                "Local AgentService daemon exited during startup; inspect daemon.log"
            )
        record = read_record(runtime_dir)
        if record is not None:
            _require_match(record, factory, cwd)
            client = await _probe(record)
            if client is not None:
                _schedule_reaper(process)
                return client
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(
                "Timed out waiting for local AgentService health; inspect daemon.log"
            )
        await asyncio.sleep(min(0.1, remaining))


def _schedule_reaper(process: subprocess.Popen[bytes]) -> None:
    async def reap() -> None:
        while process.poll() is None:
            await asyncio.sleep(0.25)

    task = asyncio.create_task(reap())
    _reapers.add(task)
    task.add_done_callback(_reapers.discard)


async def _stop_owned_child(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    deadline = asyncio.get_running_loop().time() + 2.0
    while process.poll() is None and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            return
        while process.poll() is None:
            await asyncio.sleep(0.05)


def _require_match(record: LocalServiceRecord, factory: str, cwd: Path) -> None:
    if record.factory != factory or record.cwd != str(cwd):
        raise ServiceConflictError(
            "A local AgentService is already configured for a different "
            "factory or working directory"
        )


def _validate_factory(factory: str) -> None:
    if not isinstance(factory, str) or ":" not in factory:
        raise ValueError("factory must use MODULE:CALLABLE syntax")
    module, callable_name = factory.split(":", 1)
    if (
        not module
        or not callable_name
        or any(character.isspace() for character in factory)
    ):
        raise ValueError("factory must use MODULE:CALLABLE syntax")
