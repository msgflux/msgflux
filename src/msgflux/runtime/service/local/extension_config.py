"""Formatting-preserving, atomic edits to an existing service configuration."""

from __future__ import annotations

import os
import stat
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import msgspec

from msgflux.runtime.service.local.files import _fsync_directory, process_lock
from msgflux.runtime.service.local.manifest import (
    _ServiceConfig,
    load_extension_catalog,
    load_service_manifest,
)


class ExtensionConfigMutation(msgspec.Struct, frozen=True):
    config_path: str
    agent_id: str
    extension: str
    changed: bool
    restart_required: bool


def _tomlkit():
    try:
        import tomlkit  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "Editing extension configuration requires msgflux[service]"
        ) from error
    return tomlkit


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
    target = Path(path).expanduser().resolve(strict=True)
    parser = _tomlkit()
    with process_lock(target.with_name(f".{target.name}.lock"), blocking=True):
        original, metadata = _snapshot(target)
        try:
            document = parser.parse(original.decode("utf-8"))
        except (UnicodeDecodeError, parser.exceptions.ParseError) as error:
            raise ValueError(f"Invalid service config {target}: {error}") from error
        try:
            msgspec.convert(document.unwrap(), type=_ServiceConfig, strict=True)
        except msgspec.ValidationError as error:
            raise ValueError(f"Invalid service config {target}: {error}") from error
        agents = document.get("agents", {})
        if agent_id not in agents:
            raise ValueError(f"Unknown agent {agent_id!r} in {target}")
        agent = agents[agent_id]
        selected = agent.get("extensions", [])
        if operation == "add":
            selected_name = alias or name
            _configure_extension(
                document, target, name, selected_name, agent_id, options
            )
            if selected_name not in selected:
                if "extensions" not in agent:
                    agent["extensions"] = parser.array()
                    selected = agent["extensions"]
                selected.append(selected_name)
        else:
            selected_name = name
            if name in selected:
                selected.remove(name)
        candidate = parser.dumps(document).encode("utf-8")
        changed = candidate != original
        _validate_and_publish(
            target,
            original,
            metadata,
            candidate,
            agent_id=agent_id,
            validate_factories=operation == "add",
            changed=changed,
        )
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
    parser = _tomlkit()
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


def _snapshot(path):
    with path.open("rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Service config must be a regular file")
        content = stream.read()
    return content, metadata


def _require_unchanged(path, original, metadata):
    current, current_metadata = _snapshot(path)
    published = path.lstat()
    expected = metadata.st_dev, metadata.st_ino, stat.S_IMODE(metadata.st_mode)
    if (
        current != original
        or (
            current_metadata.st_dev,
            current_metadata.st_ino,
            stat.S_IMODE(current_metadata.st_mode),
        )
        != expected
        or (published.st_dev, published.st_ino, stat.S_IMODE(published.st_mode))
        != expected
    ):
        raise RuntimeError("Service config changed while editing; retry the command")


def _validate_and_publish(
    path, original, metadata, candidate, *, agent_id, validate_factories, changed
):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".toml", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), stat.S_IMODE(metadata.st_mode))
            output.write(candidate)
            output.flush()
            os.fsync(output.fileno())
        if validate_factories:
            manifest = load_service_manifest(temporary)
            try:
                manifest.extension_catalog.validate(
                    manifest.agents[agent_id].extensions
                )
            except ValueError as error:
                raise ValueError(
                    f"{error}. Set constructor arguments with --option KEY=VALUE"
                ) from error
        # Removal only narrows selection. The typed document was checked above;
        # resolving remaining plugins would prevent repairing an unavailable
        # plugin shared by multiple agents, one selection at a time.
        _require_unchanged(path, original, metadata)
        if changed:
            os.replace(temporary, path)
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
