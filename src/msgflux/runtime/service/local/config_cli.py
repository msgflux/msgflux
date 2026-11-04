"""Offline service configuration validation and atomic value edits."""

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
        description="Read, edit, or validate service configuration offline.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "validate", parents=[common], help="validate service configuration"
    )
    for action in ("get", "set", "unset"):
        command = commands.add_parser(
            action, parents=[common], help=f"{action} stored configuration values"
        )
        if action == "get":
            command.add_argument(
                "key", nargs="?", help="TOML dotted key; omit for all values"
            )
        else:
            command.add_argument("key", help="TOML dotted key")
        if action == "set":
            command.add_argument("value", help="TOML literal or unquoted string")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    args.config = getattr(args, "config", None)
    args.json = getattr(args, "json", False)
    if args.command == "validate":
        _validate(args)
    else:
        _values(args)


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


def _fail(args, error):
    if args.json:
        sys.stdout.write(msgspec_dumps({"error": str(error)}) + "\n")
    else:
        sys.stderr.write(f"Error: {error}\n")
    raise SystemExit(1) from error


def _values(args):
    from msgflux.runtime.service.local.config_values import (  # noqa: PLC0415
        mutate_config_value,
        read_config_value,
    )

    selected = args.config or default_service_config_path()
    try:
        result = (
            read_config_value(selected, args.key)
            if args.command == "get"
            else mutate_config_value(
                selected,
                args.key,
                getattr(args, "value", None),
                operation=args.command,
            )
        )
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
        _fail(args, error)
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
    elif args.command == "get":
        value = result.value
        rendered = value if isinstance(value, str) else msgspec_dumps(value)
        if isinstance(value, dict):
            rendered = msgspec.json.format(rendered.encode(), indent=2).decode()
        sys.stdout.write(rendered + "\n")
    else:
        action = "Updated" if args.command == "set" else "Removed"
        status = action if result.changed else "No change"
        sys.stdout.write(f"{status} {args.key!r}: {result.config_path}\n")
        if result.restart_required:
            sys.stdout.write(
                "Restart the service to apply the updated configuration.\n"
            )
