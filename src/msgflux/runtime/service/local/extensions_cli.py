"""Discover extensions and manage their per-agent service configuration."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Sequence

import msgspec

from msgflux.nn.extensions.discovery import AgentExtensionCatalog
from msgflux.runtime.service.local.manifest import (
    default_service_config_path,
    load_extension_catalog,
)
from msgflux.utils.msgspec import msgspec_dumps


def _parser():
    parser = argparse.ArgumentParser(
        prog="msgflux extensions",
        description="Discover Agent extensions or add/remove their config selections.",
    )
    parser.add_argument("action", choices=("ls", "inspect", "add", "remove"))
    parser.add_argument("name", nargs="?", help="extension name")
    parser.add_argument("--config", type=Path, help="service TOML configuration")
    parser.add_argument("--json", action="store_true", help="JSON result")
    parser.add_argument("--agent", help="target agent ID for add/remove")
    parser.add_argument("--as", dest="alias", help="independent config alias for add")
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="factory option for add; TOML literal or unquoted text; repeatable",
    )
    parser.add_argument(
        "--options",
        action="store_true",
        help="inspect factory parameters/defaults; imports the selected module",
    )
    return parser


def _validate_arguments(parser, args):
    if args.action != "ls" and not args.name:
        parser.error(f"{args.action} requires an extension name")
    if args.action == "ls" and args.name:
        parser.error("ls does not accept an extension name")
    if args.action in {"add", "remove"} and not args.agent:
        parser.error(f"{args.action} requires --agent")
    if args.agent is not None and args.action not in {"add", "remove"}:
        parser.error("--agent requires add or remove")
    if (args.option or args.alias is not None) and args.action != "add":
        parser.error("--option and --as require add")
    if args.options and args.action != "inspect":
        parser.error("--options requires inspect")


def _parse_options(values):
    options = {}
    for item in values:
        key, separator, raw = item.partition("=")
        if not separator or not key.isidentifier():
            raise ValueError("--option must use a flat keyword KEY=VALUE")
        if key in options:
            raise ValueError(f"Duplicate --option {key!r}")
        try:
            document = tomllib.loads(f"value = {raw}")
            if set(document) != {"value"}:
                raise ValueError(f"--option {key!r} must contain a single value")
            value = document["value"]
        except tomllib.TOMLDecodeError as error:
            if raw.lstrip().startswith(("[", "{", '"', "'")):
                raise ValueError(
                    f"Invalid TOML literal for --option {key!r}: {error}"
                ) from error
            value = raw
        options[key] = value
    return options


def _execute(args):
    selected = args.config or default_service_config_path()
    if args.action in {"add", "remove"}:
        from msgflux.runtime.service.local.extension_config import (  # noqa: PLC0415
            mutate_extension_config,
        )

        return mutate_extension_config(
            selected,
            name=args.name,
            agent_id=args.agent,
            operation=args.action,
            alias=args.alias,
            options=_parse_options(args.option),
        )
    catalog = (
        load_extension_catalog(selected)
        if args.config is not None or selected.exists()
        else AgentExtensionCatalog()
    )
    if args.action == "ls":
        return catalog.list()
    info = catalog.resolve(args.name)
    if args.options:
        return {**msgspec.to_builtins(info), "options": catalog.options(args.name)}
    return info


def _print_result(args, result):
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
    elif args.action in {"add", "remove"}:
        action = "Added" if args.action == "add" else "Removed"
        status = f"{action} {result.extension!r}" if result.changed else "No change"
        sys.stdout.write(
            f"{status} for agent {result.agent_id!r}: {result.config_path}\n"
        )
        if result.restart_required:
            sys.stdout.write(
                "Restart the service to apply the updated configuration.\n"
            )
    elif args.action == "ls":
        sys.stdout.write("NAME\tPACKAGE\tVERSION\tFACTORY\n")
        for item in result:
            sys.stdout.write(
                f"{item.name}\t{item.distribution or '-'}\t"
                f"{item.version or '-'}\t{item.factory}\n"
            )
    else:
        info = result if isinstance(result, dict) else msgspec.to_builtins(result)
        for label, value in (
            ("Name", info["name"]),
            ("Factory", info["factory"]),
            ("Package", info["distribution"]),
            ("Version", info["version"]),
        ):
            sys.stdout.write(f"{label}: {value or '-'}\n")
        if args.options:
            _print_options(result["options"])


def _print_options(options):
    sys.stdout.write("OPTIONS\n")
    if not options:
        sys.stdout.write("  none\n")
    for item in options:
        if item.keyword_variadic:
            default = "additional keyword options"
        elif item.required:
            default = "required"
        elif item.has_default and item.default_available:
            default = f"default={msgspec_dumps(item.default)}"
        elif item.has_default:
            default = "default=<defined by factory>"
        else:
            default = "optional"
        sys.stdout.write(
            f"  {item.name}: {item.annotation or 'unspecified'} ({default})\n"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_arguments(parser, args)
    try:
        result = _execute(args)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
        if args.json:
            sys.stdout.write(msgspec_dumps({"error": str(error)}) + "\n")
        else:
            sys.stderr.write(f"Error: {error}\n")
        raise SystemExit(1) from error
    _print_result(args, result)
