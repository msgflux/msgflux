from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Sequence

from msgflux.vulcano.runtime import VulcanoRuntime

__all__ = ["main"]


def _non_negative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vulcano",
        description="Run the Vulcano code-agent terminal client.",
    )
    parser.add_argument(
        "--mock-delay",
        type=_non_negative_float,
        default=0.01,
        metavar="SECONDS",
        help="delay between mock streaming chunks (default: 0.01)",
    )
    parser.add_argument(
        "-e",
        "--extension",
        action="append",
        default=[],
        type=Path,
        metavar="PATH",
        help="load a trusted Python extension file or package (repeatable)",
    )
    parser.add_argument(
        "--no-extensions",
        action="store_true",
        help="disable installed, user, project, and explicit extensions",
    )
    parser.add_argument(
        "--trust-project-extensions",
        action="store_true",
        help="allow Python extensions from .vulcano/extensions in this project",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        app_module = import_module("msgflux.vulcano.app")
    except ModuleNotFoundError as error:
        if error.name not in {"rich", "textual"}:
            raise
        raise SystemExit(
            "Vulcano requires the optional TUI dependencies. "
            "Install them with: pip install 'msgflux[vulcano]'"
        ) from error

    app_type = app_module.VulcanoApp
    runtime = VulcanoRuntime(
        stream_delay=args.mock_delay,
        extension_paths=args.extension,
        extensions_enabled=not args.no_extensions,
        trust_project_extensions=args.trust_project_extensions,
    )
    app_type(runtime).run()
