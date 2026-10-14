"""Command line entry point for the local AgentService daemon."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Sequence

from msgflux.utils.msgspec import msgspec_dumps


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msgflux service",
        description="Run, inspect, or restart a local authenticated AgentService.",
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=("serve", "restart", "status", "init"),
        default="serve",
    )
    parser.add_argument(
        "--json", action="store_true", help="JSON status output (status only)"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--factory",
        help="trusted factory as module:attribute; it receives runtime_dir",
    )
    source.add_argument("--config", type=Path, help="TOML service manifest")
    parser.add_argument(
        "--restart-timeout",
        type=float,
        help="maximum seconds to wait for graceful restart and replacement startup",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        help="owner-only directory for daemon metadata and local runtime state",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        help="working directory used to import the factory and resolve its files",
    )
    parser.add_argument(
        "--model",
        help="provider/model-id to set in a newly initialized config (init only)",
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
    if args.json and args.action != "status":
        raise SystemExit("--json is only accepted with the status action")
    if args.model is not None and args.action != "init":
        raise SystemExit("--model is only accepted with the init action")


def _main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    _validate_options(args)
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


def main(argv: Sequence[str] | None = None) -> None:
    """Run administration commands with concise operational errors."""
    try:
        _main(argv)
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted.\n")
        raise SystemExit(130) from None
    except Exception as error:
        sys.stderr.write(f"Error: {error}\n")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
