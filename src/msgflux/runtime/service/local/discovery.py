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
from msgflux.runtime.service.http.client import (
    AgentServiceClient,
    AgentServiceHTTPError,
)
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
    returned HTTP client is closed. ``cwd`` selects its initial launch directory;
    a healthy daemon with the same factory is reused from other directories.
    Select an Agent's workspace separately when opening its service thread.
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
        client = await _reuse_running(selected_runtime, factory, deadline)
        if client is not None:
            return client

        return await _start_and_wait(selected_runtime, factory, selected_cwd, deadline)


async def restart_local_service(  # noqa: C901
    factory: str | None = None,
    *,
    runtime_dir: Path | None = None,
    cwd: Path | None = None,
    restart_timeout: float = 30,
) -> AgentServiceClient:
    """Gracefully replace the authenticated daemon and return its new client.

    Stored factory and launch directory are reused when omitted. The startup
    lock spans shutdown and replacement, so concurrent connectors cannot start
    a second owner in the transition.
    """
    if factory is not None:
        _validate_factory(factory)
    if (cwd is not None) and not isinstance(cwd, (str, Path)):
        raise TypeError("cwd must be a path")
    if (
        isinstance(restart_timeout, bool)
        or not isinstance(restart_timeout, (int, float))
        or not math.isfinite(restart_timeout)
        or restart_timeout <= 0
    ):
        raise ValueError("restart_timeout must be a finite positive number")
    selected_runtime = prepare_runtime_dir(
        Path(runtime_dir)
        if runtime_dir is not None
        else Path.home() / ".msgflux" / "runtime"
    )
    deadline = asyncio.get_running_loop().time() + restart_timeout
    async with _StartupLock(selected_runtime, deadline):
        record = read_record(selected_runtime)
        if record is None and factory is None:
            if not _daemon_is_stopped(selected_runtime):
                raise ServiceRecoveryRequiredError(
                    "The local daemon has no trusted factory metadata; "
                    "pass factory explicitly"
                )
            raise ServiceRecoveryRequiredError(
                "No stored local service factory is available; pass factory explicitly"
            )

        selected_factory = factory or record.factory
        if cwd is not None:
            selected_cwd = Path(cwd).expanduser().resolve()
        elif record is not None:
            selected_cwd = Path(record.cwd)
        else:
            selected_cwd = Path.cwd().resolve()
        if not selected_cwd.is_dir():
            raise NotADirectoryError("cwd must name an existing directory")
        if record is None:
            # An owner can be between acquiring its lifetime lock and writing
            # metadata. There is no identity to authenticate or stop, so wait
            # for that owner to finish before starting the requested factory.
            await _wait_for_daemon_stop(selected_runtime, deadline)
        elif _daemon_is_stopped(selected_runtime):
            # A dead owner left stale metadata. Preserve its launch settings and
            # start a replacement without contacting an unowned endpoint.
            remove_record(selected_runtime, record.instance_id)
        elif record is not None:
            client = await _probe(record)
            if client is None:
                raise ServiceRecoveryRequiredError(
                    "The local daemon owns its lifetime lock but failed "
                    "its identity health check"
                )
            try:
                _require_match(record, selected_factory)
                try:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError(
                            "Timed out before requesting local daemon shutdown"
                        )
                    accepted = await asyncio.wait_for(
                        client.shutdown(expected_instance_id=record.instance_id),
                        timeout=remaining,
                    )
                except AgentServiceHTTPError as exc:
                    if exc.status_code == 404:
                        raise ServiceRecoveryRequiredError(
                            "The local daemon does not support authenticated "
                            "graceful shutdown; restart it manually"
                        ) from exc
                    raise
                if accepted.instance_id != record.instance_id or not accepted.accepted:
                    raise ServiceRecoveryRequiredError(
                        "The local daemon accepted shutdown for an unexpected instance"
                    )
            finally:
                await client.aclose()
            await _wait_for_daemon_stop(selected_runtime, deadline)
            remove_record(selected_runtime, record.instance_id)
        return await _start_and_wait(
            selected_runtime, selected_factory, selected_cwd, deadline
        )


async def _start_and_wait(
    runtime_dir: Path, factory: str, cwd: Path, deadline: float
) -> AgentServiceClient:
    if deadline - asyncio.get_running_loop().time() <= 0:
        raise TimeoutError("Timed out before starting the replacement local daemon")
    process = _start_child(runtime_dir, factory, cwd)
    try:
        return await _wait_for_child(process, runtime_dir, factory, deadline)
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


async def _wait_for_daemon_stop(runtime_dir: Path, deadline: float) -> None:
    while not _daemon_is_stopped(runtime_dir):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(
                "Timed out waiting for the local daemon to finish graceful shutdown"
            )
        await asyncio.sleep(min(0.05, remaining))


async def _reuse_running(
    runtime_dir: Path, factory: str, deadline: float
) -> AgentServiceClient | None:
    while True:
        record = read_record(runtime_dir)
        if record is not None:
            client = await _probe(record)
            if client is not None:
                try:
                    _require_match(record, factory)
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
    deadline: float,
) -> AgentServiceClient:
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                "Local AgentService daemon exited during startup; inspect daemon.log"
            )
        record = read_record(runtime_dir)
        if record is not None:
            _require_match(record, factory)
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


def _require_match(record: LocalServiceRecord, factory: str) -> None:
    if record.factory != factory:
        raise ServiceConflictError(
            "A local AgentService is already configured for a different factory"
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
