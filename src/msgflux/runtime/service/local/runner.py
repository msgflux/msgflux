"""Foreground process runner for one authenticated local AgentService."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import secrets
import signal
import socket
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from msgflux.runtime.service import AgentService
from msgflux.runtime.service.local.files import (
    prepare_runtime_dir,
    process_lock,
    remove_record,
    write_record,
)
from msgflux.runtime.service.local.records import LocalServiceRecord


def _load_factory(path: str) -> Callable[[Path], Any]:
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("factory must use the form 'module:attribute'")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute)
    if not callable(factory):
        raise TypeError("configured factory attribute must be callable")
    return factory


async def _make_service(factory_path: str, runtime_dir: Path) -> AgentService:
    factory = _load_factory(factory_path)
    service = factory(runtime_dir)
    if inspect.isawaitable(service):
        service = await service
    if not isinstance(service, AgentService):
        raise TypeError("configured factory must return an AgentService")
    return service


def _validated_cwd(cwd: Path | None) -> Path:
    current = Path.cwd().resolve()
    if cwd is None:
        return current
    requested = Path(cwd).expanduser().resolve()
    if requested != current:
        raise ValueError(
            "cwd must match the current process directory; use the CLI --cwd option"
        )
    return requested


async def _wait_until_started(server, server_task: asyncio.Task) -> None:
    while not server.started:
        if server_task.done():
            await asyncio.shield(server_task)
            raise RuntimeError("local AgentService server stopped during startup")
        await asyncio.sleep(0.01)


async def _verify_health(url: str, token: str, instance_id: str) -> None:
    from msgflux.runtime.service.http.client import AgentServiceClient  # noqa: PLC0415

    client = AgentServiceClient(url, token=token, timeout=2)
    try:
        health = await client.health()
    finally:
        await client.aclose()
    if health.instance_id != instance_id:
        raise RuntimeError("local AgentService health identity did not match")


async def _stop_server(server, server_task: asyncio.Task | None) -> None:
    """Stop Uvicorn within a bound and consume any final server exception."""
    if server is not None:
        server.should_exit = True
    if server_task is None:
        return
    try:
        await asyncio.wait_for(asyncio.shield(server_task), timeout=4)
    except TimeoutError:
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)


async def serve_local_service(  # noqa: C901
    factory: str,
    *,
    runtime_dir: Path | None = None,
    cwd: Path | None = None,
) -> None:
    """Serve one trusted local factory until shutdown.

    The daemon lock is held for this entire function. The record becomes
    discoverable only after Uvicorn starts and the authenticated health endpoint
    confirms the generated instance identity.
    """
    if os.name != "posix":
        raise OSError("the local service runner currently requires POSIX locks")
    if not isinstance(factory, str) or not factory:
        raise ValueError("factory must use the form 'module:attribute'")
    working_directory = _validated_cwd(cwd)
    directory = prepare_runtime_dir(
        Path(runtime_dir).expanduser()
        if runtime_dir is not None
        else Path.home() / ".msgflux" / "runtime"
    )

    lock_path = directory / "daemon.lock"
    with process_lock(lock_path, blocking=False) as lock:
        if lock is None:
            raise RuntimeError("a local AgentService daemon already holds the lock")

        service: AgentService | None = None
        server_task: asyncio.Task | None = None
        server = None
        listening_socket: socket.socket | None = None
        instance_id = uuid4().hex
        try:
            service = await _make_service(factory, directory)

            import uvicorn  # noqa: PLC0415

            from msgflux.runtime.service.http.factory import (  # noqa: PLC0415
                create_service_app,
            )

            @contextmanager
            def capture_signals_without_replay(server):
                # Uvicorn normally re-raises captured SIGTERM/SIGINT after
                # serve() completes. In a managed runner that bypasses this
                # function's cleanup block, so restore handlers without replay.
                if threading.current_thread() is not threading.main_thread():
                    yield
                    return
                handled_signals = (signal.SIGINT, signal.SIGTERM)
                previous = {
                    item: signal.signal(item, server.handle_exit)
                    for item in handled_signals
                }
                try:
                    yield
                finally:
                    for item, handler in previous.items():
                        signal.signal(item, handler)

            class _RunnerServer(uvicorn.Server):
                capture_signals = capture_signals_without_replay

            # Keep the generated credential for the health probe and record.
            auth_token = secrets.token_urlsafe(32)
            app = create_service_app(
                service,
                token=auth_token,
                close_service=True,
                instance_id=instance_id,
            )

            listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listening_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listening_socket.bind(("127.0.0.1", 0))
            listening_socket.listen(socket.SOMAXCONN)
            listening_socket.setblocking(False)
            port = listening_socket.getsockname()[1]
            url = f"http://127.0.0.1:{port}"

            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                workers=1,
                ws="none",
                lifespan="on",
                access_log=False,
                log_level="warning",
                timeout_graceful_shutdown=2,
            )
            server = _RunnerServer(config)
            server_task = asyncio.create_task(server.serve(sockets=[listening_socket]))
            await _wait_until_started(server, server_task)
            await _verify_health(url, auth_token, instance_id)

            write_record(
                directory,
                LocalServiceRecord(
                    instance_id=instance_id,
                    factory=factory,
                    cwd=str(working_directory),
                    url=url,
                    pid=os.getpid(),
                    token=auth_token,
                ),
            )
            await server_task
        finally:
            try:
                stop_task = asyncio.create_task(_stop_server(server, server_task))
                while not stop_task.done():
                    try:
                        await asyncio.shield(stop_task)
                    except asyncio.CancelledError:
                        # Finish cleanup even under repeated cancellation.
                        continue
                await stop_task
            finally:
                if listening_socket is not None:
                    listening_socket.close()
                try:
                    remove_record(directory, instance_id)
                finally:
                    if service is not None:
                        try:
                            await service.aclose()
                        finally:
                            service.store.close()
