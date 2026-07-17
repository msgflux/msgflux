from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
from typing import Sequence

from msgflux.runtime import ExecutionScope
from msgflux.vulcano.config import VulcanoSettings
from msgflux.vulcano.runtime import VulcanoRuntime
from msgflux.vulcano.sessions import SessionStore

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
    sessions = parser.add_mutually_exclusive_group()
    sessions.add_argument(
        "--resume",
        metavar="THREAD_ID",
        help="resume a durable session by thread id",
    )
    sessions.add_argument(
        "--fork",
        metavar="THREAD_ID",
        help="fork a durable session and continue on the new thread",
    )
    parser.add_argument(
        "--no-sessions",
        action="store_true",
        help="disable durable transcript persistence",
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
    settings = VulcanoSettings.load()
    session_store = (
        None if args.no_sessions else SessionStore(settings.home / "sessions")
    )
    scope = None
    if args.resume or args.fork:
        if session_store is None:
            raise SystemExit("--resume/--fork cannot be used with --no-sessions")
        try:
            if args.fork:
                thread_id = session_store.fork(args.fork).thread_id
            else:
                session_store.info(args.resume)
                thread_id = args.resume
        except (LookupError, ValueError) as error:
            raise SystemExit(str(error)) from error
        scope = ExecutionScope(thread_id=thread_id, namespace="vulcano")
    runtime = VulcanoRuntime(
        scope=scope,
        stream_delay=args.mock_delay,
        cwd=settings.cwd,
        session_store=session_store,
        export_directory=settings.cwd,
        extension_paths=args.extension,
        extensions_enabled=not args.no_extensions,
        trust_project_extensions=args.trust_project_extensions,
        extension_user_directory=settings.home / "extensions",
    )
    app_type(runtime, settings=settings).run()
