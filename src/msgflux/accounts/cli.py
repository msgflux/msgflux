"""Local management commands for named credential bundles."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path
from typing import Sequence

from msgflux.accounts.providers import get_provider_auth
from msgflux.accounts.store import Accounts
from msgflux.utils.msgspec import msgspec_dumps


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="msgflux accounts")
    commands = parser.add_subparsers(dest="command", required=True)
    login = commands.add_parser("login", help="authenticate a provider in an account")
    login.add_argument("provider")
    login.add_argument("--account-name", default="default")
    sources = login.add_mutually_exclusive_group()
    sources.add_argument(
        "--stdin", action="store_true", help="read an API key from stdin"
    )
    sources.add_argument(
        "--from-env",
        action="store_true",
        help="import the provider's conventional API-key variable",
    )
    listing = commands.add_parser("list", help="list stored accounts without secrets")
    inspect = commands.add_parser(
        "inspect", help="inspect provider authentication metadata"
    )
    inspect.add_argument("account_name", nargs="?", default="default")
    for command in (login, listing, inspect):
        command.add_argument(
            "--accounts-dir", type=Path, help="credential storage directory"
        )
        command.add_argument(
            "--config",
            type=Path,
            help=(
                "config file whose provider declarations provide dynamic "
                "providers (default: ~/.msgflux/config.toml when it exists)"
            ),
        )
        command.add_argument("--json", action="store_true", help="emit nonsecret JSON")
    return parser


def _load_provider_catalog(path: Path):
    """Read ONLY provider declarations from ``path`` (no defaults.model needed).

    Delegates to ``load_provider_catalog`` in the service manifest module;
    importing it here keeps static config handling free of external factory
    imports.
    """
    from msgflux.runtime.service.local.manifest import (  # noqa: PLC0415
        load_provider_catalog,
    )

    return load_provider_catalog(path)


def _default_config_path() -> Path:
    from msgflux.runtime.service.local.manifest import (  # noqa: PLC0415
        default_service_config_path,
    )

    return default_service_config_path()


def _catalog_for(options) -> object | None:
    config = getattr(options, "config", None)
    if config is None:
        default = _default_config_path()
        if default.exists():
            config = default
    if config is None:
        return None
    return _load_provider_catalog(config)


def _read_key(options, spec) -> str:
    if options.from_env:
        value = os.environ.get(spec.env_name or "")
        if not value:
            raise ValueError(f"Set {spec.env_name} before importing the API key")
        return value
    if options.stdin:
        return sys.stdin.readline().rstrip("\r\n")
    return getpass.getpass(f"{spec.env_name} for account {options.account_name}: ")


def _execute(options, accounts: Accounts) -> dict:
    if options.command == "list":
        return {"accounts": accounts.list()}
    if options.command == "inspect":
        return {
            "account_name": options.account_name,
            "providers": accounts.info(options.account_name),
        }
    if accounts.provider_catalog is not None:
        spec = get_provider_auth(
            options.provider, provider_catalog=accounts.provider_catalog
        )
    else:
        spec = get_provider_auth(options.provider)
    if spec.method == "oauth":
        if options.stdin or options.from_env:
            raise ValueError(
                "This provider uses OAuth; API-key input options do not apply"
            )
        asyncio.run(accounts.login(spec.provider, account_name=options.account_name))
    else:
        accounts.set_api_key(
            spec.provider, _read_key(options, spec), account_name=options.account_name
        )
    return {
        "status": "configured",
        "account_name": options.account_name,
        "provider": spec.provider,
        "method": spec.method,
    }


def _write_result(options, result, exit_code) -> None:
    if options.json:
        sys.stdout.write(msgspec_dumps(result) + "\n")
    elif exit_code:
        sys.stderr.write(result["error"] + "\n")
    elif options.command == "list":
        sys.stdout.write("\n".join(result["accounts"]) + "\n")
    elif options.command == "inspect":
        sys.stdout.write(f"Account: {options.account_name}\n")
        for provider, metadata in result["providers"].items():
            sys.stdout.write(f"{provider}: {metadata['kind']} ({metadata['status']})\n")
    else:
        sys.stdout.write(
            f"Configured {result['provider']} in account {result['account_name']}\n"
        )


def main(argv: Sequence[str] | None = None) -> None:
    options = _parser().parse_args(argv)
    try:
        catalog = _catalog_for(options)
    except Exception as error:
        result = {"status": "failed", "error": str(error)}
        _write_result(options, result, 1)
        raise SystemExit(1) from None
    accounts = Accounts(options.accounts_dir, provider_catalog=catalog)
    try:
        result = _execute(options, accounts)
    except KeyboardInterrupt:
        result = {"status": "interrupted", "error": "Account login interrupted"}
        exit_code = 130
    except Exception as error:
        result = {"status": "failed", "error": str(error)}
        exit_code = 1
    else:
        exit_code = 0
    _write_result(options, result, exit_code)
    if exit_code:
        raise SystemExit(exit_code)
