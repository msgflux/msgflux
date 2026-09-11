"""Provider-neutral declaration policies for executable tools."""

from __future__ import annotations

from typing import Any, Collection, Mapping

import msgspec

from msgflux.tools.runtime import _copy_mapping, _require_name


class DispatchSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Open dispatch selection compiled from one tool declaration."""

    name: str = "foreground"
    options: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(
            self, "name", _require_name(self.name, "dispatch.name")
        )
        msgspec.structs.force_setattr(
            self,
            "options",
            _copy_mapping(self.options, "dispatch.options"),
        )

    @classmethod
    def coerce(cls, value: DispatchSpec | str | None) -> DispatchSpec:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        raise TypeError("`dispatch` must be a DispatchSpec, string, or None")


class ContextBinding(msgspec.Struct, frozen=True, kw_only=True):
    """Bind one named runtime source to one hidden tool parameter."""

    source: str
    parameter: str | None = None
    required: bool = True
    options: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        source = _require_name(self.source, "context.source")
        parameter = self.parameter if self.parameter is not None else source
        msgspec.structs.force_setattr(self, "source", source)
        msgspec.structs.force_setattr(
            self,
            "parameter",
            _require_name(parameter, "context.parameter"),
        )
        if not isinstance(self.required, bool):
            raise TypeError("`context.required` must be a bool")
        msgspec.structs.force_setattr(
            self,
            "options",
            _copy_mapping(self.options, "context.options"),
        )

    @classmethod
    def coerce(cls, value: ContextBinding | str) -> ContextBinding:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(source=value)
        raise TypeError("Context bindings must be ContextBinding instances or strings")


class ContextSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Explicit runtime values made available to one tool."""

    bindings: tuple[ContextBinding | str, ...] = ()

    def __post_init__(self) -> None:
        bindings = tuple(ContextBinding.coerce(item) for item in self.bindings)
        parameters = [binding.parameter for binding in bindings]
        if len(parameters) != len(set(parameters)):
            raise ValueError("Context bindings must target unique parameters")
        msgspec.structs.force_setattr(self, "bindings", bindings)

    @classmethod
    def coerce(
        cls,
        value: ContextSpec | Collection[ContextBinding | str] | None,
    ) -> ContextSpec:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str) or not isinstance(value, Collection):
            raise TypeError("`context` must be a ContextSpec or collection of bindings")
        return cls(bindings=tuple(value))


class LoadingSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Catalog visibility policy independent from mutable thread state."""

    deferred: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.deferred, bool):
            raise TypeError("`loading.deferred` must be a bool")


__all__ = ["ContextBinding", "ContextSpec", "DispatchSpec", "LoadingSpec"]
