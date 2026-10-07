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
        prog="msgflux-service",
        description="Run or restart a local authenticated AgentService.",
    )
    parser.add_argument(
        "action", nargs="?", choices=("serve", "restart"), default="serve"
    )
    parser.add_argument(
        "--factory",
        help="trusted factory as module:attribute; it receives runtime_dir",
    )
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run foreground serving or gracefully restart the local daemon."""
    args = _parser().parse_args(argv)
    if args.action == "restart":
        if args.restart_timeout is not None and args.restart_timeout <= 0:
            raise SystemExit("--restart-timeout must be positive")
        from msgflux.runtime.service.local.discovery import (  # noqa: PLC0415
            restart_local_service,
        )

        async def restart() -> None:
            client = await restart_local_service(
                args.factory,
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
    if args.factory is None:
        raise SystemExit("serve requires --factory MODULE:CALLABLE")
    if args.cwd is not None:
        os.chdir(args.cwd.expanduser().resolve())
    current_directory = str(Path.cwd().resolve())
    if current_directory not in sys.path:
        sys.path.insert(0, current_directory)

    from msgflux.runtime.service.local.runner import (  # noqa: PLC0415
        serve_local_service,
    )

    asyncio.run(
        serve_local_service(
            args.factory,
            runtime_dir=args.runtime_dir,
            cwd=Path.cwd(),
        )
    )


if __name__ == "__main__":
    main()
