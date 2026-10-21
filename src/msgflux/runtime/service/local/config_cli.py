"""Offline service configuration validation through the manifest loader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import msgspec

from msgflux.runtime.service.local.manifest import (
    default_service_config_path,
    load_service_manifest,
)
from msgflux.utils.msgspec import msgspec_dumps


class _ValidationResult(msgspec.Struct, frozen=True):
    valid: bool
    config_path: str
    default_agent: str | None = None
    agents: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def _parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help="service TOML configuration",
    )
    common.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help="JSON result"
    )
    parser = argparse.ArgumentParser(
        prog="msgflux config",
        parents=[common],
        description="Validate service configuration or manage browser origins offline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "validate", parents=[common], help="validate service configuration"
    )
    cors = commands.add_parser(
        "cors", parents=[common], help="list or edit configured browser origins"
    )
    actions = cors.add_subparsers(dest="action", required=True)
    for action in ("ls", "add", "remove"):
        command = actions.add_parser(action, parents=[common])
        if action != "ls":
            command.add_argument(
                "origin", help="exact HTTP(S) origin or standalone '*'"
            )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.config = getattr(args, "config", None)
    args.json = getattr(args, "json", False)
    if args.command == "cors":
        _cors(args)
    else:
        _validate(args)


def _validate(args):
    selected = (args.config or default_service_config_path()).expanduser().absolute()
    try:
        manifest = load_service_manifest(selected)
        manifest.extension_catalog.validate(
            tuple(
                dict.fromkeys(
                    name
                    for agent in manifest.agents.values()
                    for name in agent.extensions
                )
            )
        )
    except (OSError, ValueError, TypeError) as error:
        result = _ValidationResult(False, str(selected), errors=(str(error),))
    else:
        result = _ValidationResult(
            True, str(manifest.path), manifest.defaults.agent, tuple(manifest.agents)
        )
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
    elif result.valid:
        sys.stdout.write(
            f"Configuration valid: {result.config_path}\n"
            f"Default agent: {result.default_agent}\n"
            f"Declared agents: {', '.join(result.agents)}\n"
        )
    else:
        sys.stderr.write(f"Invalid configuration: {result.config_path}\n")
        for error in result.errors:
            sys.stderr.write(f"- {error}\n")
    if not result.valid:
        raise SystemExit(1)


def _cors(args):
    from msgflux.runtime.service.local.cors_config import (  # noqa: PLC0415
        mutate_cors_config,
        read_cors_config,
    )

    selected = args.config or default_service_config_path()
    try:
        result = (
            read_cors_config(selected)
            if args.action == "ls"
            else mutate_cors_config(selected, args.origin, operation=args.action)
        )
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
        if args.json:
            sys.stdout.write(msgspec_dumps({"error": str(error)}) + "\n")
        else:
            sys.stderr.write(f"Error: {error}\n")
        raise SystemExit(1) from error
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
        return
    sys.stdout.write(f"Configuration: {result.config_path}\n")
    if result.cors_allow_origins:
        sys.stdout.write("CORS origins:\n")
        for origin in result.cors_allow_origins:
            sys.stdout.write(f"- {origin}\n")
    else:
        sys.stdout.write("CORS disabled\n")
    if args.action != "ls":
        if result.restart_required:
            sys.stdout.write(
                "Restart the service to apply the updated configuration.\n"
            )
        else:
            sys.stdout.write("No change\n")
