"""Inspect installed Agent extension metadata without starting a service."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from msgflux.nn.extensions.discovery import AgentExtensionCatalog
from msgflux.runtime.service.local.manifest import (
    default_service_config_path,
    load_extension_catalog,
)
from msgflux.utils.msgspec import msgspec_dumps


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="msgflux extensions",
        description="Discover installed Agent extensions without importing their code.",
    )
    parser.add_argument("action", choices=("ls", "inspect"))
    parser.add_argument("name", nargs="?", help="extension name for inspect")
    parser.add_argument("--config", type=Path, help="service TOML configuration")
    parser.add_argument("--json", action="store_true", help="JSON metadata")
    args = parser.parse_args(argv)
    if args.action == "inspect" and not args.name:
        parser.error("inspect requires an extension name")
    if args.action == "ls" and args.name:
        parser.error("ls does not accept an extension name")
    try:
        selected = args.config or default_service_config_path()
        catalog = (
            load_extension_catalog(selected)
            if args.config is not None or selected.exists()
            else AgentExtensionCatalog()
        )
        result = (
            catalog.resolve(args.name) if args.action == "inspect" else catalog.list()
        )
    except (OSError, ValueError, TypeError) as error:
        if args.json:
            sys.stdout.write(msgspec_dumps({"error": str(error)}) + "\n")
        else:
            sys.stderr.write(f"Error: {error}\n")
        raise SystemExit(1) from error
    if args.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
    elif args.action == "inspect":
        for label, value in (
            ("Name", result.name),
            ("Factory", result.factory),
            ("Package", result.distribution),
            ("Version", result.version),
        ):
            sys.stdout.write(f"{label}: {value or '-'}\n")
    else:
        sys.stdout.write("NAME\tPACKAGE\tVERSION\tFACTORY\n")
        for item in result:
            sys.stdout.write(
                f"{item.name}\t{item.distribution or '-'}\t"
                f"{item.version or '-'}\t{item.factory}\n"
            )
