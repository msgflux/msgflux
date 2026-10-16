"""Metadata discovery and explicit construction of Agent extension factories."""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from importlib import metadata
from inspect import (
    Parameter,
    isasyncgenfunction,
    iscoroutine,
    iscoroutinefunction,
    signature,
)
from math import isfinite
from types import UnionType
from typing import Any, Union, get_args, get_origin

import msgspec

from msgflux.nn.extensions.base import AgentExtension
from msgflux.utils.imports import load_factory, validate_factory_path

GROUP = "msgflux.agent_extensions"
_UNAVAILABLE = object()


class ExtensionConfig(msgspec.Struct, forbid_unknown_fields=True):
    """Optional factory override and constructor options for a named extension."""

    factory: str | None = None
    options: dict[str, Any] = msgspec.field(default_factory=dict)


class ExtensionInfo(msgspec.Struct, frozen=True):
    """Available factory metadata; resolving it does not import plugin code."""

    name: str
    factory: str
    distribution: str | None = None
    version: str | None = None


class ExtensionOption(msgspec.Struct, frozen=True):
    """One keyword parameter, with safely representable constructor defaults."""

    name: str
    annotation: str | None
    required: bool
    has_default: bool
    default: Any = None
    default_available: bool = False
    keyword_variadic: bool = False


class AgentExtensionCatalog:
    """Discover installed entry points and trusted, explicitly configured factories.

    Listing is metadata-only. Validation imports only selected factories and
    checks arguments without invoking them; each creation invokes its factory.
    """

    def __init__(self, configured: Mapping[str, ExtensionConfig] | None = None) -> None:
        self._configured = deepcopy(dict(configured or {}))
        for name, config in self._configured.items():
            _validate_name(name)
            if not isinstance(config, ExtensionConfig):
                raise TypeError(f"Extension {name!r} requires ExtensionConfig")
            if config.factory is not None:
                try:
                    validate_factory_path(config.factory)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Extension {name!r}: {error}") from error
        self._installed: dict[str, list[ExtensionInfo]] = {}
        for entry in metadata.entry_points(group=GROUP):
            distribution = entry.dist
            info = ExtensionInfo(
                name=entry.name,
                factory=f"{entry.module}:{entry.attr}",
                distribution=(
                    distribution.metadata.get("Name") if distribution else None
                ),
                version=distribution.version if distribution else None,
            )
            self._installed.setdefault(entry.name, []).append(info)

    def list(self) -> tuple[ExtensionInfo, ...]:
        """Return all available names in order; ambiguous names require an override."""
        return tuple(
            self.resolve(name)
            for name in sorted(self._installed.keys() | self._configured.keys())
        )

    def resolve(self, name: str) -> ExtensionInfo:
        """Resolve one factory using metadata, without importing or constructing it."""
        _validate_name(name)
        config = self._configured.get(name)
        if config is not None and config.factory is not None:
            return ExtensionInfo(name=name, factory=config.factory)
        candidates = self._installed.get(name, ())
        if not candidates:
            raise ValueError(f"Unknown Agent extension {name!r}")
        if len(candidates) > 1:
            choices = sorted(
                f"{info.factory} ({info.distribution or 'unknown distribution'}"
                f"{f' {info.version}' if info.version else ''})"
                for info in candidates
            )
            raise ValueError(
                f"Agent extension {name!r} has multiple installed factories: "
                f"{', '.join(choices)}; configure an explicit factory"
            )
        return candidates[0]

    def validate(self, names: Iterable[str]) -> None:
        """Import selected factories and bind options without constructing instances."""
        for name in names:
            self._factory(name)

    def options(self, name: str) -> tuple[ExtensionOption, ...]:
        """Inspect factory keyword parameters without binding or invoking it."""
        _info, _factory, parameters = self._load_factory(name)
        return tuple(
            _describe_option(parameter)
            for parameter in parameters.parameters.values()
            if parameter.kind
            not in {Parameter.POSITIONAL_ONLY, Parameter.VAR_POSITIONAL}
        )

    def create(self, name: str) -> AgentExtension:
        """Construct one extension; its selected alias owns execution-local state."""
        info, factory, options = self._factory(name)
        try:
            extension = factory(**options)
            if iscoroutine(extension):
                extension.close()
                raise TypeError("extension factories must be synchronous")
            if not isinstance(extension, AgentExtension):
                raise TypeError("factory must return an AgentExtension")
            if extension._agent_ref is not None and extension._agent_ref() is not None:
                raise ValueError(
                    "factory returned an extension already bound to an Agent"
                )
            extension.name = name
            return extension
        except Exception as error:
            raise ValueError(
                f"Cannot create Agent extension {name!r} ({info.factory}): {error}"
            ) from error

    def _factory(self, name):
        info, factory, parameters = self._load_factory(name)
        config = self._configured.get(name)
        try:
            options = deepcopy(config.options) if config is not None else {}
            positional = [
                parameter.name
                for parameter in parameters.parameters.values()
                if parameter.kind == Parameter.POSITIONAL_ONLY
                and parameter.default is Parameter.empty
            ]
            if positional:
                raise TypeError(
                    "Required positional-only factory parameters cannot be configured: "
                    + ", ".join(positional)
                )
            parameters.bind(**options)
            return info, factory, options
        except Exception as error:
            raise ValueError(
                f"Invalid Agent extension {name!r} ({info.factory}): {error}"
            ) from error

    def _load_factory(self, name):
        info = self.resolve(name)
        try:
            factory = load_factory(info.factory)
            if any(
                iscoroutinefunction(candidate) or isasyncgenfunction(candidate)
                for candidate in (factory, factory.__call__)
            ):
                raise TypeError("extension factories must be synchronous")
            return info, factory, signature(factory)
        except Exception as error:
            raise ValueError(
                f"Invalid Agent extension {name!r} ({info.factory}): {error}"
            ) from error


def _describe_option(parameter):
    has_default = parameter.default is not Parameter.empty
    default = _json_default(parameter.default) if has_default else _UNAVAILABLE
    variadic = parameter.kind == Parameter.VAR_KEYWORD
    return ExtensionOption(
        name=parameter.name,
        annotation=_annotation_name(parameter.annotation),
        required=not has_default and not variadic,
        has_default=has_default,
        default=None if default is _UNAVAILABLE else default,
        default_available=default is not _UNAVAILABLE,
        keyword_variadic=variadic,
    )


def _annotation_name(annotation):
    if annotation is Parameter.empty:
        return None
    if isinstance(annotation, str):
        return annotation
    if annotation is None:
        return "None"
    if isinstance(annotation, type):
        return annotation.__qualname__
    origin = get_origin(annotation)
    if origin is None:
        return None
    arguments = tuple(_annotation_name(argument) for argument in get_args(annotation))
    if any(argument is None for argument in arguments):
        return None
    if origin in {Union, UnionType}:
        return " | ".join(arguments)
    if not isinstance(origin, type):
        return None
    name = origin.__qualname__
    return f"{name}[{', '.join(arguments)}]" if arguments else name


def _json_default(value, seen=frozenset()):
    """Copy only JSON primitive trees; never invoke custom repr or encoders."""
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        return value if isfinite(value) else _UNAVAILABLE
    if type(value) not in {list, tuple, dict} or id(value) in seen or len(seen) >= 64:
        return _UNAVAILABLE
    seen = seen | {id(value)}
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            return _UNAVAILABLE
        result = {key: _json_default(item, seen) for key, item in value.items()}
        return (
            _UNAVAILABLE
            if any(item is _UNAVAILABLE for item in result.values())
            else result
        )
    result = [_json_default(item, seen) for item in value]
    return _UNAVAILABLE if any(item is _UNAVAILABLE for item in result) else result


def _validate_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Extension name must be a non-empty string")
