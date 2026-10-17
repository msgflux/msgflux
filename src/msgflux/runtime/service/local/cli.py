"""Command line entry point for the local AgentService daemon."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Sequence

from msgflux.utils.msgspec import msgspec_dumps


def _help_for(resource: str, text: str, *resources: str) -> str:
    return text if resource in resources else argparse.SUPPRESS


def _parser(resource: str = "service") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"msgflux {resource}",
        description={
            "service": "Run, inspect, or restart a local authenticated AgentService.",
            "agents": "List agents or inspect their loaded registration defaults.",
            "threads": "List, watch, steer, interrupt, or read thread results.",
            "runs": "List runs, inspect recovery, resume, or read final results.",
        }[resource],
    )
    actions = {
        "service": ("serve", "restart", "status", "stop", "down", "inspect", "init"),
        "runs": ("inspect", "resume", "ls", "result"),
        "threads": ("ls", "watch", "steer", "interrupt", "stop", "result"),
        "agents": ("ls", "inspect"),
    }
    parser.add_argument(
        "action", nargs="?", choices=actions[resource], default=actions[resource][0]
    )
    parser.set_defaults(resource=resource, agent_id=None, thread_id=None, run_id=None)
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON output for operation commands",
    )
    parser.add_argument(
        "agent_id" if resource == "agents" else "thread_id",
        nargs="?",
        help="agent ID"
        if resource == "agents"
        else _help_for(resource, "thread ID", "runs", "threads"),
    )
    parser.add_argument("run_id", nargs="?", help=_help_for(resource, "run ID", "runs"))
    parser.add_argument(
        "--stop-timeout",
        type=float,
        help=_help_for(resource, "graceful shutdown deadline in seconds", "service"),
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help=_help_for(
            resource,
            "wait for the selected run to leave accepted/running state (result only)",
            "runs",
            "threads",
        ),
    )
    parser.add_argument(
        "--active",
        action="store_true",
        help=_help_for(
            resource, "list only active resources", "runs", "agents", "threads"
        ),
    )
    parser.add_argument(
        "--agent",
        help=_help_for(resource, "filter active runs by registered agent ID", "runs"),
    )
    parser.add_argument(
        "-m",
        "--message",
        help=_help_for(resource, "message for the active run (steer only)", "threads"),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--factory",
        help=_help_for(
            resource,
            "trusted factory as module:attribute; it receives runtime_dir",
            "service",
        ),
    )
    source.add_argument(
        "--config",
        type=Path,
        help=_help_for(resource, "TOML service manifest", "service"),
    )
    parser.add_argument(
        "--restart-timeout",
        type=float,
        help=_help_for(
            resource,
            "maximum seconds to wait for graceful restart and replacement startup",
            "service",
        ),
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        help="owner-only directory for daemon metadata and local runtime state",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        help=_help_for(
            resource,
            "working directory used to import the factory and resolve its files",
            "service",
        ),
    )
    parser.add_argument(
        "--model",
        help=_help_for(
            resource,
            "provider/model-id to set in a newly initialized config (init only)",
            "service",
        ),
    )
    return parser


def _initialize_config(args: argparse.Namespace) -> None:
    if args.factory is not None:
        raise SystemExit("init accepts --config, not --factory")
    from msgflux.runtime.service.local.manifest import (  # noqa: PLC0415
        default_service_config_path,
        initialize_service_config,
    )

    config = args.config or default_service_config_path()
    if args.config is not None and not config.is_absolute():
        base = args.cwd.expanduser().resolve() if args.cwd is not None else Path.cwd()
        config = base / config
    path = initialize_service_config(config, model=args.model)
    sys.stdout.write(f"Initialized service config at {path}\n")


def _show_status(args: argparse.Namespace) -> None:
    if any(
        value is not None
        for value in (args.factory, args.config, args.cwd, args.restart_timeout)
    ):
        raise SystemExit("status accepts only --runtime-dir and --json")
    from msgflux.runtime.service.local import inspect_local_service  # noqa: PLC0415

    result = asyncio.run(inspect_local_service(runtime_dir=args.runtime_dir))
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
    elif result.status == "running":
        sys.stdout.write(f"AgentService running at {result.url} (PID {result.pid})\n")
    else:
        sys.stdout.write(
            f"AgentService {result.status}; runtime: {result.runtime_dir}\n"
        )
    if result.status != "running":
        raise SystemExit(1)


def _validate_options(args: argparse.Namespace) -> None:
    if args.json and args.action not in {
        "status",
        "stop",
        "down",
        "inspect",
        "resume",
        "interrupt",
        "watch",
        "steer",
        "result",
        "ls",
    }:
        raise SystemExit(
            "--json is only accepted with operation and observation commands"
        )
    if args.model is not None and args.action != "init":
        raise SystemExit("--model is only accepted with the init action")

    if args.wait and not (
        args.resource in {"runs", "threads"} and args.action == "result"
    ):
        raise SystemExit("--wait is only accepted with threads result or runs result")

    _validate_target(args)
    _validate_message(args)
    if args.agent is not None and not (args.resource == "runs" and args.action == "ls"):
        raise SystemExit("--agent is only accepted with runs ls")
    if args.active and not (
        args.resource in {"runs", "agents", "threads"} and args.action == "ls"
    ):
        raise SystemExit(
            "--active is only accepted with runs ls, agents ls or threads ls"
        )

    if args.stop_timeout is not None and not (
        args.resource == "service" and args.action in {"stop", "down"}
    ):
        raise SystemExit("--stop-timeout is only accepted with service stop")
    if (
        args.resource != "service" or args.action in {"stop", "down", "inspect"}
    ) and any(
        value is not None
        for value in (args.factory, args.config, args.cwd, args.restart_timeout)
    ):
        raise SystemExit(
            f"{args.action} does not accept factory, config, cwd or restart timeout"
        )


def _validate_message(args: argparse.Namespace) -> None:
    if args.resource == "threads" and args.action == "steer":
        if args.message is None or not args.message.strip():
            raise SystemExit("threads steer requires a non-empty --message")
    elif args.message is not None:
        raise SystemExit("--message is only accepted with threads steer")


def _validate_listing_target(args: argparse.Namespace) -> None:
    if args.resource == "runs":
        if not args.active:
            raise SystemExit("runs ls requires --active")
        if args.run_id is not None:
            raise SystemExit("runs ls accepts at most one THREAD_ID")
    elif any(
        value is not None for value in (args.agent_id, args.thread_id, args.run_id)
    ):
        raise SystemExit(f"{args.resource} ls does not accept thread or run IDs")


def _validate_target(args: argparse.Namespace) -> None:
    if args.action == "ls":
        _validate_listing_target(args)
    elif args.resource == "agents":
        if args.agent_id is None or args.run_id is not None:
            raise SystemExit("agents inspect requires exactly one AGENT_ID")
    elif args.resource == "runs":
        if args.thread_id is None or args.run_id is None:
            raise SystemExit(f"{args.action} requires THREAD_ID and RUN_ID")
    elif args.resource == "threads":
        if args.thread_id is None or args.run_id is not None:
            raise SystemExit(f"{args.action} requires exactly one THREAD_ID")
    elif args.thread_id is not None or args.run_id is not None:
        raise SystemExit("THREAD_ID and RUN_ID require msgflux runs or threads")


async def _stop(args) -> None:
    from msgflux.runtime.service.local import stop_local_service  # noqa: PLC0415

    try:
        result = await stop_local_service(
            runtime_dir=args.runtime_dir,
            stop_timeout=args.stop_timeout if args.stop_timeout is not None else 30,
        )
    except TimeoutError as error:
        raise TimeoutError(
            "Graceful shutdown deadline exceeded; the service may still be closing"
        ) from error
    sys.stdout.write(
        msgspec_dumps(result) + "\n" if args.json else "AgentService stopped.\n"
    )


async def _inspect_service(args) -> None:
    from msgflux.runtime.service.http.client import AgentServiceClient  # noqa: PLC0415
    from msgflux.runtime.service.local import inspect_local_service  # noqa: PLC0415
    from msgflux.runtime.service.local.files import read_record  # noqa: PLC0415

    status = await inspect_local_service(runtime_dir=args.runtime_dir)
    result = {
        "service": status,
        "factory": None,
        "config_path": None,
        "cwd": None,
        "http": None,
    }
    if status.status == "running":
        record = read_record(Path(status.runtime_dir))
        if record is None or record.instance_id != status.instance_id:
            raise RuntimeError("Local daemon identity changed during inspection")
        client = AgentServiceClient(record.url, token=record.token)
        try:
            inspection = await client.inspect_service()
            if inspection.instance_id != record.instance_id:
                raise RuntimeError("Local daemon identity changed during inspection")
            current = read_record(Path(status.runtime_dir))
            if current is None or current.instance_id != record.instance_id:
                raise RuntimeError("Local daemon identity changed during inspection")
            result.update(
                factory=record.factory,
                config_path=record.config_path,
                cwd=record.cwd,
                http=inspection.http,
            )
        finally:
            await client.aclose()
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
        return
    sys.stdout.write(f"AgentService {status.status}\nRuntime: {status.runtime_dir}\n")
    if status.status == "running":
        sys.stdout.write(
            f"Address: {status.url}\nPID: {status.pid}\n"
            f"Instance: {status.instance_id}\n"
        )
        sys.stdout.write(
            f"Source: {result['config_path'] or result['factory']}\n"
            f"Cwd: {result['cwd']}\n"
        )
        origins = result["http"].cors_allow_origins
        sys.stdout.write(f"CORS: {', '.join(origins) if origins else 'disabled'}\n")


async def _run_operation(args) -> None:
    from msgflux.runtime.service.http.client import AgentServiceClient  # noqa: PLC0415
    from msgflux.runtime.service.local import inspect_local_service  # noqa: PLC0415
    from msgflux.runtime.service.local.files import read_record  # noqa: PLC0415

    status = await inspect_local_service(runtime_dir=args.runtime_dir)
    if status.status != "running":
        raise RuntimeError(
            f"AgentService is {status.status}; {args.action} requires an active service"
        )
    record = read_record(Path(status.runtime_dir))
    if record is None or record.instance_id != status.instance_id:
        raise RuntimeError("Local daemon identity changed during inspection")
    client = AgentServiceClient(record.url, token=record.token)
    try:
        health = await client.health()
        if health.instance_id != record.instance_id:
            raise RuntimeError("Local daemon identity changed during inspection")
        if args.action == "ls":
            from msgflux.runtime.service.local.listing import (  # noqa: PLC0415
                list_resources,
            )

            await list_resources(client, args)
            return
        if args.resource == "agents":
            from msgflux.runtime.service.local.agent_inspection import (  # noqa: PLC0415
                inspect_registered_agent,
            )

            await inspect_registered_agent(client, args)
            return
        if args.action == "result":
            from msgflux.runtime.service.local.results import (  # noqa: PLC0415
                show_result,
            )

            await show_result(client, args)
            return
        if args.resource == "threads":
            await _thread_operation(client, args)
            return
        if args.action == "resume":
            result = await client.resume_checkpoint(args.thread_id, args.run_id)
        else:
            result = await client.inspect_run(args.thread_id, args.run_id)
    finally:
        await client.aclose()
    _show_run_result(args, result)


async def _thread_operation(client, args) -> None:
    if args.action == "watch":
        from msgflux.runtime.service.local.watch import (  # noqa: PLC0415
            watch_thread,
        )

        await watch_thread(client, args.thread_id, as_json=args.json)
    elif args.action == "steer":
        await _steer_thread(client, args)
    else:
        await _interrupt_thread(client, args)


def _show_run_result(args, result) -> None:
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
        return
    if args.action == "resume":
        sys.stdout.write(f"Run {result.run_id}: {result.status}\n")
        return
    run_status = (
        result.receipt.status
        if result.receipt is not None
        else result.checkpoint_status
    )
    sys.stdout.write(f"Run {result.run_id}: {run_status or 'unknown'}\n")
    sys.stdout.write(
        f"Thread: {args.thread_id}\nCheckpoint: {result.checkpoint_status or 'none'}\n"
    )
    sys.stdout.write(
        f"Approvals: {result.approval_request_count}; "
        f"background tasks: {len(result.background_tasks)}\n"
    )
    if result.requires_quiescence:
        sys.stdout.write(
            "Recovery requires trusted-host confirmation of worker quiescence.\n"
        )
    for reason in result.reasons:
        sys.stdout.write(f"- {reason}\n")


async def _active_thread_run(client, thread_id: str) -> str | None:
    runs = tuple(
        run for run in await client.active_runs() if run.thread_id == thread_id
    )
    if len(runs) > 1:
        raise RuntimeError(
            "Multiple active runs for the thread; cannot select a target"
        )
    return runs[0].run_id if runs else None


async def _steer_thread(client, args) -> None:
    run_id = await _active_thread_run(client, args.thread_id)
    if run_id is None:
        raise RuntimeError(f"No active run for thread {args.thread_id}")
    notice = await client.steer(args.thread_id, run_id, args.message)
    result = {
        "thread_id": args.thread_id,
        "run_id": run_id,
        "notification_id": notice["notification_id"],
        "status": "queued",
    }
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
    else:
        sys.stdout.write(
            f"Message queued for thread {args.thread_id}, run {run_id} "
            f"(notification_id={notice['notification_id']}).\n"
        )


async def _interrupt_thread(client, args) -> None:
    run_id = await _active_thread_run(client, args.thread_id)
    requested = await client.interrupt(args.thread_id, run_id) if run_id else False
    if run_id is None and not any(
        thread.thread_id == args.thread_id for thread in await client.threads()
    ):
        raise KeyError(f"Unknown thread: {args.thread_id}")
    status = (
        "interrupt_requested"
        if requested
        else "not_active"
        if run_id is not None
        else "no_active_run"
    )
    result = {"thread_id": args.thread_id, "run_id": run_id, "status": status}
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
    elif requested:
        sys.stdout.write(f"Interruption requested for run {run_id}.\n")
    else:
        sys.stdout.write(f"No active run to interrupt for thread {args.thread_id}.\n")


def _main(argv: Sequence[str] | None = None, *, resource: str = "service") -> None:
    args = _parser(resource).parse_args(argv)
    _validate_options(args)
    actions = {
        "stop": _run_operation if resource == "threads" else _stop,
        "down": _stop,
        "inspect": _run_operation
        if resource in {"runs", "agents"}
        else _inspect_service,
        "resume": _run_operation,
        "interrupt": _run_operation,
        "watch": _run_operation,
        "steer": _run_operation,
        "result": _run_operation,
        "ls": _run_operation,
    }
    if args.action in actions:
        asyncio.run(actions[args.action](args))
        return
    if args.action == "status":
        _show_status(args)
        return
    if args.action == "init":
        _initialize_config(args)
        return
    if args.action == "restart":
        if args.restart_timeout is not None and args.restart_timeout <= 0:
            raise SystemExit("--restart-timeout must be positive")
        from msgflux.runtime.service.local.discovery import (  # noqa: PLC0415
            restart_local_service,
        )

        async def restart() -> None:
            client = await restart_local_service(
                args.factory,
                config=args.config,
                runtime_dir=args.runtime_dir,
                cwd=args.cwd,
                restart_timeout=args.restart_timeout or 30,
            )
            try:
                health = await client.health()
                sys.stdout.write(
                    f"Restarted AgentService {health.instance_id} "
                    f"at {client.base_url}\n"
                )
            finally:
                await client.aclose()

        asyncio.run(restart())
        return
    if args.cwd is not None:
        os.chdir(args.cwd.expanduser().resolve())
    if args.factory is None and args.config is None:
        from msgflux.runtime.service.local.manifest import (  # noqa: PLC0415
            default_service_config_path,
            initialize_service_config,
        )

        args.config = initialize_service_config(default_service_config_path())
    current_directory = str(Path.cwd().resolve())
    if current_directory not in sys.path:
        sys.path.insert(0, current_directory)

    from msgflux.runtime.service.local.runner import (  # noqa: PLC0415
        serve_local_service,
    )

    asyncio.run(
        serve_local_service(
            args.factory,
            config=args.config,
            runtime_dir=args.runtime_dir,
            cwd=Path.cwd(),
        )
    )


def main(argv: Sequence[str] | None = None, *, resource: str = "service") -> None:
    """Run administration commands with concise operational errors."""
    try:
        _main(argv, resource=resource)
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted.\n")
        raise SystemExit(130) from None
    except Exception as error:
        sys.stderr.write(f"Error: {error}\n")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
