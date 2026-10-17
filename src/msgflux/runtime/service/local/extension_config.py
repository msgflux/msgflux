"""Formatting-preserving, atomic edits to an existing service configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import msgspec

from msgflux.runtime.service.local.config_editing import (
    edit_service_config,
    toml_parser,
)
from msgflux.runtime.service.local.manifest import (
    load_extension_catalog,
    load_service_manifest,
)


class ExtensionConfigMutation(msgspec.Struct, frozen=True):
    config_path: str
    agent_id: str
    extension: str
    changed: bool
    restart_required: bool


def mutate_extension_config(
    path: Path,
    *,
    name: str,
    agent_id: str,
    operation: Literal["add", "remove"],
    alias: str | None = None,
    options: dict[str, Any] | None = None,
) -> ExtensionConfigMutation:
    """Edit one agent's selection without constructing plugins or starting a daemon."""
    _validate_edit(name, operation, alias, options)
    selected_name = alias or name
    target = Path(path).expanduser().resolve(strict=True)

    def mutate(document):
        parser = toml_parser()
        agents = document.get("agents", {})
        if agent_id not in agents:
            raise ValueError(f"Unknown agent {agent_id!r} in {target}")
        agent = agents[agent_id]
        selected = agent.get("extensions", [])
        if operation == "add":
            _configure_extension(
                document, target, name, selected_name, agent_id, options
            )
            if selected_name not in selected:
                if "extensions" not in agent:
                    agent["extensions"] = parser.array()
                    selected = agent["extensions"]
                selected.append(selected_name)
        elif name in selected:
            selected.remove(name)

    def validate(candidate):
        if operation == "add":
            manifest = load_service_manifest(candidate)
            try:
                manifest.extension_catalog.validate(
                    manifest.agents[agent_id].extensions
                )
            except ValueError as error:
                raise ValueError(
                    f"{error}. Set constructor arguments with --option KEY=VALUE"
                ) from error
        # Removal only narrows selection. Do not resolve unavailable plugins,
        # so operators can repair selections one agent at a time.

    target, changed = edit_service_config(target, mutate, validate=validate)
    return ExtensionConfigMutation(
        str(target), agent_id, selected_name, changed, changed
    )


def _validate_edit(name, operation, alias, options):
    if operation not in {"add", "remove"}:
        raise ValueError("Extension operation must be add or remove")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Extension name must be a non-empty string")
    if alias is not None and (not isinstance(alias, str) or not alias.strip()):
        raise ValueError("Extension alias must be a non-empty string")
    if operation == "remove" and (alias is not None or options):
        raise ValueError("Removing an extension does not accept aliases or options")
    if options is not None and not isinstance(options, dict):
        raise ValueError("Extension options must be a mapping of constructor arguments")


def _configure_extension(document, path, name, selected_name, agent_id, overrides):
    catalog = load_extension_catalog(path)
    factory = catalog.resolve(name).factory
    extensions = document.get("extensions", {})
    source = extensions.get(name, {})
    existing = extensions.get(selected_name)
    old_options = _options(existing) if existing is not None else {}
    if selected_name != name and existing is not None:
        if catalog.resolve(selected_name).factory != factory:
            raise ValueError(
                f"Extension alias {selected_name!r} references another factory"
            )
    selected_options = old_options if existing is not None else _options(source)
    merged = {**deepcopy(selected_options), **(overrides or {})}
    if not _same_value(merged, old_options):
        others = [
            other
            for other, config in document["agents"].items()
            if other != agent_id and selected_name in config.get("extensions", [])
        ]
        if others:
            raise ValueError(
                f"Extension {selected_name!r} is also selected by {', '.join(others)}; "
                "use --as to configure independent options"
            )
    if selected_name != name or overrides:
        _write_options(
            document, selected_name, merged, factory if selected_name != name else None
        )


def _write_options(document, name, options, factory):
    parser = toml_parser()
    if "extensions" not in document:
        document["extensions"] = parser.table()
    if name not in document["extensions"]:
        document["extensions"][name] = parser.table()
    target = document["extensions"][name]
    if factory is not None and target.get("factory") != factory:
        target["factory"] = factory
    if options:
        if "options" not in target:
            target["options"] = parser.table()
        for key, value in options.items():
            if key not in target["options"] or not _same_value(
                target["options"][key], value
            ):
                target["options"][key] = value


def _options(config):
    values = config.get("options", {})
    return values.unwrap() if hasattr(values, "unwrap") else dict(values)


def _same_value(left, right):
    left = left.unwrap() if hasattr(left, "unwrap") else left
    right = right.unwrap() if hasattr(right, "unwrap") else right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_value(value, right[key]) for key, value in left.items()
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_value(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right
