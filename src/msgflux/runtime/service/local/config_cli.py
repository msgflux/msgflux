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


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="msgflux config",
        description="Validate configuration without starting a service or models.",
    )
    parser.add_argument("action", choices=("validate",))
    parser.add_argument("--config", type=Path, help="service TOML configuration")
    parser.add_argument("--json", action="store_true", help="JSON validation result")
    args = parser.parse_args(argv)
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
