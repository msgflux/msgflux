"""Generic stored-value access and edits to service TOML configuration."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

import msgspec

from msgflux.runtime.service.local.config_editing import (
    edit_service_config,
    same_config_value,
    toml_parser,
)
from msgflux.runtime.service.local.manifest import load_service_manifest


class ConfigValueView(msgspec.Struct, frozen=True):
    config_path: str
    key: str | None
    value: Any


class ConfigValueMutation(msgspec.Struct, frozen=True):
    config_path: str
    key: str
    changed: bool
    restart_required: bool


def parse_config_path(key: str) -> tuple[str, ...]:
    """Parse a TOML dotted key, including quoted segments, into its segments."""
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Config key must be a non-empty string")
    if "\n" in key or "\r" in key:
        raise ValueError(f"Invalid config key {key!r}")
    if _has_unquoted_assignment_or_comment(key):
        raise ValueError(f"Invalid config key {key!r}")
    try:
        document = tomllib.loads(f"{key} = 0")
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Invalid config key {key!r}: {error}") from error
    segments: list[str] = []
    current: Any = document
    while isinstance(current, dict):
        if len(current) != 1:
            raise ValueError(f"Invalid config key {key!r}")
        segment, current = next(iter(current.items()))
        if not isinstance(segment, str) or not segment:
            raise ValueError(f"Invalid config key {key!r}")
        segments.append(segment)
    if type(current) is not int or current != 0 or not segments:
        raise ValueError(f"Invalid config key {key!r}")
    return tuple(segments)


def parse_config_value(raw: str, *, label: str = "value") -> Any:
    """Parse one TOML literal, falling back to an unquoted string.

    A successfully parsed document must contain exactly the intended value, so
    injected additional assignments are rejected. Strings/containers that look
    like TOML literals but are malformed are errors, not strings.
    """
    if not isinstance(raw, str):
        raise ValueError(f"Invalid TOML literal for {label}: value must be a string")
    try:
        document = tomllib.loads(f"value = {raw}")
    except tomllib.TOMLDecodeError as error:
        if raw.lstrip().startswith(("[", "{", '"', "'")):
            raise ValueError(f"Invalid TOML literal for {label}: {error}") from error
        return raw
    if set(document) != {"value"}:
        raise ValueError(f"{label} must contain a single value")
    return document["value"]


def _has_unquoted_assignment_or_comment(key: str) -> bool:
    """Whether ``key`` has ``=`` or ``#`` outside a quoted key segment.

    The TOML probe in :func:`parse_config_path` appends its own sentinel
    assignment, so a key that itself carries an assignment or comment could
    otherwise be smuggled past it (for example ``defaults.model = 0 #``).
    ``=``/``#`` inside quoted segments (``agents."a=b#c"``) are legitimate
    key characters and must be allowed, hence the quote-aware scan.
    """
    index = 0
    while index < len(key):
        char = key[index]
        if char in {'"', "'"}:
            quote = char
            index += 1
            while index < len(key) and key[index] != quote:
                # Only basic (double-quoted) strings have escapes; in literal
                # (single-quoted) strings a backslash is an ordinary character.
                index += 2 if quote == '"' and key[index] == "\\" else 1
            index += 1  # step past the closing quote (or end of string)
        elif char in {"=", "#"}:
            return True
        else:
            index += 1
    return False


def read_config_value(path: Path, key: str | None = None) -> ConfigValueView:
    """Return the stored value for ``key`` (or the whole document), unedited."""
    target = Path(path).expanduser().resolve(strict=True)
    try:
        document = tomllib.loads(target.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"Invalid service config {target}: {error}") from error
    if key is None:
        return ConfigValueView(str(target), None, document)
    segments = parse_config_path(key)
    current: Any = document
    for segment in segments:
        if not isinstance(current, dict) or segment not in current:
            raise ValueError(f"No config value for {key!r} in {target}")
        current = current[segment]
    return ConfigValueView(str(target), key, current)


def mutate_config_value(
    path: Path,
    key: str,
    raw_value: str | None = None,
    *,
    operation: Literal["set", "unset"],
) -> ConfigValueMutation:
    """Set or unset one stored value, fully validating the candidate document."""
    if operation not in {"set", "unset"}:
        raise ValueError("Config operation must be set or unset")
    segments = parse_config_path(key)
    parsed_value: Any = None
    if operation == "set":
        if raw_value is None:
            raise ValueError(f"Setting {key!r} requires a value")
        parsed_value = parse_config_value(raw_value)
    elif raw_value is not None:
        raise ValueError(f"Unsetting {key!r} does not accept a value")
    target = Path(path).expanduser().resolve(strict=True)

    def mutate(document):
        if operation == "set":
            _apply_set(document, segments, parsed_value, key, target)
        else:
            _apply_unset(document, segments, key, target)

    def validate(candidate):
        # Full manifest validation without importing/validating plugins.
        load_service_manifest(candidate)

    target, changed = edit_service_config(
        target, mutate, validate=validate, validate_original=False
    )
    return ConfigValueMutation(str(target), key, changed, changed)


def _apply_set(document, segments, parsed_value, key, target):
    parser = toml_parser()
    current = document
    for segment in segments[:-1]:
        if segment not in current:
            current[segment] = (
                parser.inline_table() if _is_inline_table(current) else parser.table()
            )
        elif not isinstance(current[segment], dict):
            raise ValueError(
                f"Config key {key!r} traverses a non-table value "
                f"at {segment!r} in {target}"
            )
        current = current[segment]
    leaf = segments[-1]
    if leaf in current and same_config_value(current[leaf], parsed_value):
        return
    current[leaf] = parsed_value


def _apply_unset(document, segments, key, target):
    current = document
    for segment in segments[:-1]:
        if segment not in current:
            return
        if not isinstance(current[segment], dict):
            raise ValueError(
                f"Config key {key!r} traverses a non-table value "
                f"at {segment!r} in {target}"
            )
        current = current[segment]
    current.pop(segments[-1], None)


def _is_inline_table(container) -> bool:
    parser = toml_parser()
    return isinstance(container, parser.items.InlineTable)
