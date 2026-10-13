"""Command line entry point for the local AgentService daemon."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="msgflux service",
        description="Run or restart a local authenticated AgentService.",
    )
    parser.add_argument(
        "action", nargs="?", choices=("serve", "restart", "init"), default="serve"
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


def main(argv: Sequence[str] | None = None) -> None:
    """Run foreground serving or gracefully restart the local daemon."""
    args = _parser().parse_args(argv)
    if args.model is not None and args.action != "init":
        raise SystemExit("--model is only accepted with the init action")
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


if __name__ == "__main__":
    main()
