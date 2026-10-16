"""Metadata discovery and explicit construction of Agent extension factories."""

from collections.abc import Iterable, Mapping
from copy import deepcopy
from importlib import metadata
from inspect import isasyncgenfunction, iscoroutine, iscoroutinefunction, signature
from typing import Any

import msgspec

from msgflux.nn.extensions.base import AgentExtension
from msgflux.utils.imports import load_factory, validate_factory_path

GROUP = "msgflux.agent_extensions"


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
        info = self.resolve(name)
        config = self._configured.get(name)
        try:
            options = deepcopy(config.options) if config is not None else {}
            factory = load_factory(info.factory)
            if any(
                iscoroutinefunction(candidate) or isasyncgenfunction(candidate)
                for candidate in (factory, factory.__call__)
            ):
                raise TypeError("extension factories must be synchronous")
            signature(factory).bind(**options)
            return info, factory, options
        except Exception as error:
            raise ValueError(
                f"Invalid Agent extension {name!r} ({info.factory}): {error}"
            ) from error


def _validate_name(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Extension name must be a non-empty string")
