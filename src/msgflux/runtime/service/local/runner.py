"""Foreground process runner for one authenticated local AgentService."""

from __future__ import annotations

import asyncio
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

from msgflux.runtime.service import AgentService, SQLiteServiceStore
from msgflux.runtime.service.local.agent_inspection import manifest_agent_configuration
from msgflux.runtime.service.local.extension_factories import with_extensions
from msgflux.runtime.service.local.files import (
    prepare_runtime_dir,
    process_lock,
    remove_record,
    write_record,
)
from msgflux.runtime.service.local.manifest import (
    ServiceManifest,
    load_service_manifest,
)
from msgflux.runtime.service.local.records import LocalServiceRecord
from msgflux.utils.imports import _factory_import_root, load_factory


def _load_factory(path: str, cwd: Path) -> Callable[[Path], Any]:
    with _factory_import_root(cwd):
        return load_factory(path)


def _with_factory_import_root(
    factory: Callable[..., Any], cwd: Path
) -> Callable[..., Any]:
    def invoke(*args: Any, **kwargs: Any) -> Any:
        with _factory_import_root(cwd):
            return factory(*args, **kwargs)

    return invoke


async def _make_service(
    factory_path: str, runtime_dir: Path, cwd: Path
) -> AgentService:
    factory = _load_factory(factory_path, cwd)
    with _factory_import_root(cwd):
        service = factory(runtime_dir)
    if inspect.isawaitable(service):
        service = await service
    if not isinstance(service, AgentService):
        raise TypeError("configured factory must return an AgentService")
    return service


def _make_manifest_service(
    config_path: Path,
    runtime_dir: Path,
    cwd: Path,
    *,
    manifest: ServiceManifest | None = None,
) -> tuple[AgentService, str, str]:
    manifest = manifest or load_service_manifest(config_path, cwd=cwd)
    coding_builder = None
    accounts = None
    agent_dir = None
    if manifest.coding_catalog.agents:
        from msgflux.coding.config import (  # noqa: PLC0415
            _resolve_coding_agent_dir,
        )
        from msgflux.coding.factory import (  # noqa: PLC0415
            create_coding_session_factory,
        )

        agent_dir = _resolve_coding_agent_dir(
            runtime_dir,
            Path(manifest.defaults.agent_dir)
            if manifest.defaults.agent_dir is not None
            else None,
        )
        from msgflux.accounts.store import Accounts  # noqa: PLC0415

        accounts = Accounts(manifest.defaults.accounts_dir)
        coding_builder = create_coding_session_factory
    factories = {}
    for agent_id, config in manifest.agents.items():
        try:
            if config.harness == "coding":
                if coding_builder is None:
                    raise RuntimeError("Coding session factory builder is unavailable")
                factories[agent_id] = coding_builder(
                    options=manifest.coding_catalog.agents[agent_id],
                    catalog=manifest.coding_catalog,
                    config_path=manifest.path,
                    runtime_dir=runtime_dir,
                    agent_dir=agent_dir,
                    accounts=accounts,
                    extension_catalog=manifest.extension_catalog,
                )
            else:
                if config.factory is None:
                    raise ValueError(
                        f"Agent {agent_id!r} has no coding harness or factory"
                    )
                factories[agent_id] = with_extensions(
                    _with_factory_import_root(_load_factory(config.factory, cwd), cwd),
                    config.extensions,
                    manifest.extension_catalog,
                    import_root=cwd,
                )
        except (ValueError, TypeError, ImportError, AttributeError) as exc:
            raise ValueError(f"Invalid agent factory for {agent_id!r}: {exc}") from exc
    store = SQLiteServiceStore(runtime_dir / "service.sqlite3")
    try:
        service = AgentService(
            store=store,
            default_agent=manifest.defaults.agent,
            default_account=manifest.defaults.account,
        )
        for agent_id, factory in factories.items():
            service.register(
                agent_id,
                factory,
                configuration=manifest_agent_configuration(manifest, agent_id),
            )
    except BaseException:
        store.close()
        raise
    return service, str(manifest.path), manifest.digest


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


async def _finish_cleanup(task: asyncio.Task) -> None:
    """Join cleanup despite repeated cancellation before releasing host locks."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    await task


async def serve_local_service(  # noqa: C901
    factory: str | None = None,
    *,
    config: Path | None = None,
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
    if (factory is None) == (config is None):
        raise ValueError("provide exactly one of factory or config")
    if factory is not None and (not isinstance(factory, str) or not factory):
        raise ValueError("factory must use the form 'module:attribute'")
    working_directory = _validated_cwd(cwd)
    requested_runtime = (
        Path(runtime_dir).expanduser()
        if runtime_dir is not None
        else Path.home() / ".msgflux" / "runtime"
    )
    manifest = None
    if config is not None:
        manifest = load_service_manifest(config, cwd=working_directory)
        if manifest.coding_catalog.agents:
            from msgflux.coding.config import (  # noqa: PLC0415
                _resolve_coding_agent_dir,
            )

            _resolve_coding_agent_dir(
                requested_runtime,
                Path(manifest.defaults.agent_dir)
                if manifest.defaults.agent_dir is not None
                else None,
            )
    directory = prepare_runtime_dir(requested_runtime)

    lock_path = directory / "daemon.lock"
    with process_lock(lock_path, blocking=False) as lock:
        if lock is None:
            raise RuntimeError("a local AgentService daemon already holds the lock")

        service: AgentService | None = None
        server_task: asyncio.Task | None = None
        server = None
        listening_socket: socket.socket | None = None
        instance_id = uuid4().hex
        config_path: str | None = None
        config_digest: str | None = None
        http_config = None
        try:
            if config is None:
                if factory is None:
                    raise ValueError("provide exactly one of factory or config")
                service = await _make_service(factory, directory, working_directory)
            else:
                if manifest is None:
                    raise RuntimeError("service config was not loaded during preflight")
                http_config = manifest.http
                service, config_path, config_digest = _make_manifest_service(
                    config, directory, working_directory, manifest=manifest
                )

            if service.store._path is not None:
                from msgflux.runtime.service.local.leases import (  # noqa: PLC0415
                    LocalOwnerLease,
                )

                service._owner_lease = LocalOwnerLease(service)

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

            def request_shutdown() -> None:
                if server is not None:
                    server.should_exit = True

            app = create_service_app(
                service,
                token=auth_token,
                http_config=http_config,
                close_service=True,
                instance_id=instance_id,
                on_shutdown=request_shutdown,
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
                    cwd=str(working_directory),
                    url=url,
                    pid=os.getpid(),
                    token=auth_token,
                    factory=factory,
                    config_path=config_path,
                    config_digest=config_digest,
                ),
            )
            await server_task
        finally:
            try:
                stop_task = asyncio.create_task(_stop_server(server, server_task))
                await _finish_cleanup(stop_task)
            finally:
                if listening_socket is not None:
                    listening_socket.close()
                try:
                    remove_record(directory, instance_id)
                finally:
                    if service is not None:
                        try:
                            await _finish_cleanup(asyncio.create_task(service.aclose()))
                        finally:
                            try:
                                if service._owner_lease is not None:
                                    service._owner_lease.close()
                            finally:
                                service.store.close()
