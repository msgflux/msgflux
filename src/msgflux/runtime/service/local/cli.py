"""Command line entry point for the foreground local AgentService daemon."""

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
        description="Run a local authenticated AgentService in the foreground.",
    )
    parser.add_argument(
        "--factory",
        required=True,
        help="trusted factory as module:attribute; it receives runtime_dir",
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
    """Parse options, set process cwd once, then run the async daemon."""
    args = _parser().parse_args(argv)
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
