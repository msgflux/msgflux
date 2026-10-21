"""Discover or start the per-user local AgentService daemon."""

from __future__ import annotations

import asyncio
import logging
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
from msgflux.runtime.service.local.manifest import (
    ServiceManifest,
    default_service_config_path,
    initialize_service_config,
    load_service_manifest,
    validate_factory_path,
)
from msgflux.runtime.service.local.records import LocalServiceRecord, LocalServiceStatus
from msgflux.utils.errors import exception_message

_logger = logging.getLogger(__name__)
_reapers: set[asyncio.Task[None]] = set()
_cleanups: set[asyncio.Task[None]] = set()
_STARTUP_LOCK = "startup.lock"
_DAEMON_LOCK = "daemon.lock"
_LOG_FILE = "daemon.log"


async def inspect_local_service(
    *, runtime_dir: Path | None = None
) -> LocalServiceStatus:
    """Inspect the recorded daemon without starting it or creating local files."""
    directory = (
        (
            Path(runtime_dir)
            if runtime_dir is not None
            else Path.home() / ".msgflux" / "runtime"
        )
        .expanduser()
        .absolute()
    )
    if directory.is_symlink():
        raise ValueError("Runtime directory must be a real directory, not a symlink")
    record = read_record(directory)
    if record is None:
        return LocalServiceStatus("stopped", str(directory))
    if await _probe(record) is None:
        return LocalServiceStatus("unreachable", str(directory))
    return LocalServiceStatus(
        "running", str(directory), record.url, record.instance_id, record.pid
    )


async def connect_local_service(
    factory: str | None = None,
    *,
    config: Path | None = None,
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
    record = await _ensure_local_service(
        factory,
        config=config,
        runtime_dir=runtime_dir,
        cwd=cwd,
        startup_timeout=startup_timeout,
    )
    return AgentServiceClient(record.url, token=record.token)


async def _ensure_local_service(
    factory: str | None,
    *,
    config: Path | None,
    runtime_dir: Path | None,
    cwd: Path | None,
    startup_timeout: float,
) -> LocalServiceRecord:
    """Return trusted metadata for a healthy matching owner, starting one if absent."""
    if (factory is None) == (config is None):
        if factory is None and config is None:
            config = initialize_service_config(default_service_config_path())
        else:
            raise ValueError("provide exactly one of factory or config")
    manifest = None
    if config is not None:
        selected_cwd = (
            (Path(cwd) if cwd is not None else Path.cwd()).expanduser().resolve()
        )
        manifest = load_service_manifest(config, cwd=selected_cwd)
    else:
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
        record = await _reuse_running(selected_runtime, factory, manifest, deadline)
        if record is not None:
            return record

        return await _start_and_wait(
            selected_runtime, factory, selected_cwd, deadline, manifest=manifest
        )


async def restart_local_service(  # noqa: C901
    factory: str | None = None,
    *,
    config: Path | None = None,
    runtime_dir: Path | None = None,
    cwd: Path | None = None,
    restart_timeout: float = 30,
) -> AgentServiceClient:
    """Gracefully replace the authenticated daemon and return its new client.

    Stored factory and launch directory are reused when omitted. The startup
    lock spans shutdown and replacement, so concurrent connectors cannot start
    a second owner in the transition.
    """
    if factory is not None and config is not None:
        raise ValueError("provide only one of factory or config")
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
        if record is None and factory is None and config is None:
            if not _daemon_is_stopped(selected_runtime):
                raise ServiceRecoveryRequiredError(
                    "The local daemon has no trusted factory metadata; "
                    "pass factory explicitly"
                )
            raise ServiceRecoveryRequiredError(
                "No stored local service factory is available; pass factory explicitly"
            )

        manifest = None
        if factory is None and config is None and record is not None:
            factory = record.factory
            if record.config_path is not None:
                config = Path(record.config_path)
        if config is not None:
            selected_cwd_for_config = (
                Path(cwd).expanduser().resolve()
                if cwd is not None
                else Path(record.cwd)
                if record is not None
                else Path.cwd().resolve()
            )
            manifest = load_service_manifest(config, cwd=selected_cwd_for_config)
        selected_factory = factory
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
            healthy = await _probe(record)
            if healthy is None:
                raise ServiceRecoveryRequiredError(
                    "The local daemon owns its lifetime lock but failed "
                    "its identity health check"
                )
            _require_match(record, selected_factory, manifest, allow_digest_change=True)
            await _shutdown_running_daemon(record, selected_runtime, deadline)
        record = await _start_and_wait(
            selected_runtime,
            selected_factory,
            selected_cwd,
            deadline,
            manifest=manifest,
        )
        return AgentServiceClient(record.url, token=record.token)


async def stop_local_service(
    *, runtime_dir: Path | None = None, stop_timeout: float = 30
) -> LocalServiceStatus:
    """Stop a verified daemon, wait for cleanup, and never start a replacement."""
    if (
        isinstance(stop_timeout, bool)
        or not isinstance(stop_timeout, (int, float))
        or not math.isfinite(stop_timeout)
        or stop_timeout <= 0
    ):
        raise ValueError("stop_timeout must be a finite positive number")
    directory = (
        (
            Path(runtime_dir)
            if runtime_dir is not None
            else Path.home() / ".msgflux" / "runtime"
        )
        .expanduser()
        .absolute()
    )
    if directory.is_symlink():
        raise ValueError("Runtime directory must be a real directory, not a symlink")
    if not directory.exists():
        return LocalServiceStatus("stopped", str(directory))
    directory = prepare_runtime_dir(directory)
    deadline = asyncio.get_running_loop().time() + stop_timeout
    async with _StartupLock(directory, deadline):
        record = read_record(directory)
        if _daemon_is_stopped(directory):
            if record is not None:
                remove_record(directory, record.instance_id)
            return LocalServiceStatus("stopped", str(directory))
        async with asyncio.timeout_at(deadline):
            if record is None:
                # Shutdown removes metadata before releasing the lifetime lock.
                # Wait for that owner to finish; there is no verified target to signal.
                await _wait_for_daemon_stop(directory, deadline)
                return LocalServiceStatus("stopped", str(directory))
            if await _probe(record) is None:
                raise ServiceRecoveryRequiredError(
                    "Cannot verify the active local daemon identity; "
                    "stop it through the trusted host"
                )
            await _shutdown_running_daemon(record, directory, deadline)
    return LocalServiceStatus("stopped", str(directory))


async def _shutdown_running_daemon(record, runtime_dir: Path, deadline: float) -> None:
    client = AgentServiceClient(record.url, token=record.token)
    try:
        try:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Timed out before requesting local daemon shutdown")
            accepted = await asyncio.wait_for(
                client.shutdown(expected_instance_id=record.instance_id),
                timeout=remaining,
            )
        except AgentServiceHTTPError as exc:
            if exc.status_code == 404:
                raise ServiceRecoveryRequiredError(
                    "The local daemon does not support authenticated "
                    "graceful shutdown; stop it through the trusted host"
                ) from exc
            raise
        if accepted.instance_id != record.instance_id or not accepted.accepted:
            raise ServiceRecoveryRequiredError(
                "The local daemon accepted shutdown for an unexpected instance"
            )
    finally:
        await client.aclose()
    await _wait_for_daemon_stop(runtime_dir, deadline)
    remove_record(runtime_dir, record.instance_id)


async def _start_and_wait(
    runtime_dir: Path,
    factory: str | None,
    cwd: Path,
    deadline: float,
    *,
    manifest: ServiceManifest | None = None,
) -> LocalServiceRecord:
    if deadline - asyncio.get_running_loop().time() <= 0:
        raise TimeoutError("Timed out before starting the replacement local daemon")
    process = _start_child(runtime_dir, factory, cwd, manifest=manifest)
    try:
        return await _wait_for_child(process, runtime_dir, factory, manifest, deadline)
    except BaseException as error:
        cleanup = asyncio.create_task(_stop_owned_child(process))
        _cleanups.add(cleanup)
        cleanup.add_done_callback(_cleanups.discard)
        while not cleanup.done():
            try:
                await asyncio.wait((cleanup,))
            except asyncio.CancelledError:
                continue
        try:
            cleanup.result()
        except BaseException as cleanup_error:
            error.add_note(
                f"Local daemon child cleanup failed: {exception_message(cleanup_error)}"
            )
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
    runtime_dir: Path,
    factory: str | None,
    manifest: ServiceManifest | None,
    deadline: float,
) -> LocalServiceRecord | None:
    while True:
        record = read_record(runtime_dir)
        if record is not None:
            healthy = await _probe(record)
            if healthy is not None:
                _require_match(record, factory, manifest)
                return healthy
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


async def _probe(record: LocalServiceRecord) -> LocalServiceRecord | None:
    client = AgentServiceClient(record.url, token=record.token, timeout=0.75)
    try:
        health = await asyncio.wait_for(client.health(), timeout=1.0)
        if health.instance_id != record.instance_id or health.version != 1:
            return None
        return record
    except asyncio.CancelledError:
        raise
    except Exception:
        return None
    finally:
        await client.aclose()


def _daemon_is_stopped(runtime_dir: Path) -> bool:
    with process_lock(runtime_dir / _DAEMON_LOCK) as handle:
        return handle is not None


def _start_child(
    runtime_dir: Path,
    factory: str | None,
    cwd: Path,
    *,
    manifest: ServiceManifest | None = None,
) -> subprocess.Popen[bytes]:
    log_path = runtime_dir / _LOG_FILE
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(log_path, flags, 0o600)
    process: subprocess.Popen[bytes]
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise PermissionError("Local daemon log must be an owned regular file")
        os.fchmod(descriptor, 0o600)
        # argv is passed directly; no shell interprets configured values.
        command = [
            sys.executable,
            "-m",
            "msgflux.runtime.service.local.cli",
            "--runtime-dir",
            str(runtime_dir),
            "--cwd",
            str(cwd),
        ]
        if manifest is not None:
            command.extend(("--config", str(manifest.path)))
        else:
            if factory is None:
                raise ValueError("provide exactly one of factory or config")
            command.extend(("--factory", factory))
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
    except BaseException as error:
        try:
            os.close(descriptor)
        except OSError as close_error:
            error.add_note(
                "Local daemon log descriptor cleanup failed: "
                f"{exception_message(close_error)}"
            )
        raise
    try:
        os.close(descriptor)
    except OSError as close_error:
        _logger.warning(
            "Local daemon log descriptor cleanup failed: %s",
            exception_message(close_error),
        )
    return process


async def _wait_for_child(
    process: subprocess.Popen[bytes],
    runtime_dir: Path,
    factory: str | None,
    manifest: ServiceManifest | None,
    deadline: float,
) -> LocalServiceRecord:
    while True:
        if process.poll() is not None:
            raise RuntimeError(
                "Local AgentService daemon exited during startup; inspect daemon.log"
            )
        record = read_record(runtime_dir)
        if record is not None:
            _require_match(record, factory, manifest)
            healthy = await _probe(record)
            if healthy is not None:
                _schedule_reaper(process)
                return healthy
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


def _require_match(
    record: LocalServiceRecord,
    factory: str | None,
    manifest: ServiceManifest | None,
    *,
    allow_digest_change: bool = False,
) -> None:
    matches = (
        record.factory == factory
        if manifest is None
        else record.config_path == str(manifest.path)
        and (allow_digest_change or record.config_digest == manifest.digest)
    )
    if not matches:
        raise ServiceConflictError(
            "A local AgentService is already configured for a different "
            "factory or config source"
        )


def _validate_factory(factory: str) -> None:
    if not isinstance(factory, str):
        raise ValueError("factory must use MODULE:CALLABLE syntax")
    validate_factory_path(factory)
