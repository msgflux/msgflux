"""Provider-neutral tool catalog contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Collection, Mapping

import msgspec

from msgflux.tools.runtime import _copy_mapping, _require_name


class NativeToolBinding(msgspec.Struct, frozen=True, kw_only=True):
    """Provider-native representation supported by a logical tool."""

    provider: str
    api_mode: str
    kind: str
    execution: str
    options: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("provider", "api_mode", "kind"):
            msgspec.structs.force_setattr(
                self,
                field_name,
                _require_name(getattr(self, field_name), field_name),
            )
        if self.execution not in {"client", "provider"}:
            raise ValueError("`execution` must be `client` or `provider`")
        msgspec.structs.force_setattr(
            self,
            "options",
            _copy_mapping(self.options, "native_binding.options"),
        )


class ToolRef(msgspec.Struct, frozen=True, kw_only=True):
    """Stable reference used by buckets and handles without exposing internals."""

    library_id: str
    tool_id: str

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(
            self,
            "library_id",
            _require_name(self.library_id, "library_id"),
        )
        msgspec.structs.force_setattr(
            self,
            "tool_id",
            _require_name(self.tool_id, "tool_id"),
        )


class ToolChoice(msgspec.Struct, frozen=True, kw_only=True):
    """Provider-neutral catalog selection policy."""

    mode: str = "auto"
    name: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "none", "required", "tool"}:
            raise ValueError("`choice.mode` must be auto, none, required, or tool")
        if self.mode == "tool":
            msgspec.structs.force_setattr(
                self,
                "name",
                _require_name(self.name, "choice.name"),
            )
        elif self.name is not None:
            raise ValueError("`choice.name` is only valid when mode is `tool`")

    @classmethod
    def coerce(
        cls,
        value: ToolChoice | str | Mapping[str, Any] | None,
    ) -> ToolChoice:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            function = value.get("function")
            if value.get("type") == "function" and isinstance(function, Mapping):
                name = function.get("name")
            elif value.get("type") == "function":
                name = value.get("name")
            else:
                raise ValueError("Tool choice mappings must select a function tool")
            return cls(mode="tool", name=_require_name(name, "choice.name"))
        if isinstance(value, str) and value.strip():
            if value in {"auto", "none", "required"}:
                return cls(mode=value)
            return cls(mode="tool", name=value)
        raise TypeError("`choice` must be a ToolChoice, string, mapping, or None")


class ToolCatalogEntry(msgspec.Struct, frozen=True, kw_only=True):
    """Execution-free projection of a tool in one thread catalog snapshot."""

    ref: ToolRef
    description: str | None
    input_schema: Mapping[str, Any]
    annotations: Mapping[str, Any] = msgspec.field(default_factory=dict)
    native_bindings: tuple[NativeToolBinding, ...] = ()
    strict: bool | None = None
    namespace: str | None = None
    catalog_role: str | None = None
    kind: str = "tool"
    deferred: bool = False
    loaded: bool = False
    display_name: str | None = None
    usage_guidance: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ToolRef):
            raise TypeError("`ref` must be a ToolRef")
        if not isinstance(self.deferred, bool) or not isinstance(self.loaded, bool):
            raise TypeError("`deferred` and `loaded` must be bool values")
        if self.loaded and not self.deferred:
            raise ValueError("Only deferred tools can be marked as loaded")
        if self.strict is not None and not isinstance(self.strict, bool):
            raise TypeError("`strict` must be bool or None")
        if self.namespace is not None:
            msgspec.structs.force_setattr(
                self,
                "namespace",
                _require_name(self.namespace, "namespace"),
            )
        if self.catalog_role is not None:
            msgspec.structs.force_setattr(
                self,
                "catalog_role",
                _require_name(self.catalog_role, "catalog_role"),
            )
        msgspec.structs.force_setattr(self, "kind", _require_name(self.kind, "kind"))
        msgspec.structs.force_setattr(
            self,
            "input_schema",
            _copy_mapping(self.input_schema, "input_schema"),
        )
        msgspec.structs.force_setattr(
            self,
            "annotations",
            _copy_mapping(self.annotations, "annotations"),
        )

    @property
    def name(self) -> str:
        return self.ref.tool_id

    def to_function_schema(self) -> dict[str, Any]:
        """Return the provider-neutral nested function-tool representation."""
        function: dict[str, Any] = {"name": self.name}
        if self.description is not None:
            function["description"] = self.description
        function["parameters"] = deepcopy(dict(self.input_schema))
        if self.strict is not None:
            function["strict"] = self.strict
        return {"type": "function", "function": function}

    @classmethod
    def from_definition(
        cls,
        definition: Any,
        *,
        library_id: str,
        loaded: bool = False,
    ) -> ToolCatalogEntry:
        """Project an execution definition without retaining its executor."""
        return cls(
            ref=ToolRef(library_id=library_id, tool_id=definition.name),
            description=definition.description,
            input_schema=definition.input_schema,
            annotations=definition.annotations,
            native_bindings=definition.native_bindings,
            strict=definition.metadata.get("strict"),
            namespace=definition.metadata.get("execution_namespace"),
            catalog_role=definition.metadata.get("catalog_role"),
            kind=definition.kind,
            deferred=definition.loading.deferred,
            loaded=loaded,
            display_name=definition.display_name,
            usage_guidance=definition.usage_guidance,
        )


class ToolCatalogView(msgspec.Struct, frozen=True, kw_only=True):
    """Immutable tool catalog snapshot scoped to one conversation thread."""

    library_id: str
    thread_id: str
    entries: tuple[ToolCatalogEntry, ...]
    choice: ToolChoice | str | Mapping[str, Any] = msgspec.field(
        default_factory=ToolChoice
    )

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(
            self,
            "library_id",
            _require_name(self.library_id, "library_id"),
        )
        msgspec.structs.force_setattr(
            self,
            "thread_id",
            _require_name(self.thread_id, "thread_id"),
        )
        entries = tuple(self.entries)
        if not all(isinstance(entry, ToolCatalogEntry) for entry in entries):
            raise TypeError("`entries` must contain ToolCatalogEntry values")
        names = [entry.name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError("Tool catalog entries must have unique names")
        foreign = [
            entry.name for entry in entries if entry.ref.library_id != self.library_id
        ]
        if foreign:
            formatted = ", ".join(f"`{name}`" for name in foreign)
            raise ValueError(
                f"Tool catalog entries belong to another library: {formatted}"
            )
        choice = ToolChoice.coerce(self.choice)
        if choice.mode == "tool" and choice.name not in set(names):
            raise ValueError(f"Selected tool `{choice.name}` is not in the catalog")
        msgspec.structs.force_setattr(self, "entries", entries)
        msgspec.structs.force_setattr(self, "choice", choice)
        search_entries = [entry for entry in entries if entry.catalog_role == "search"]
        if len(search_entries) > 1:
            raise ValueError("Tool catalog views support at most one search entry")
        if search_entries and search_entries[0].deferred:
            raise ValueError("The tool catalog search entry cannot be deferred")

    @property
    def search_entry(self) -> ToolCatalogEntry | None:
        return next(
            (entry for entry in self.entries if entry.catalog_role == "search"),
            None,
        )

    def tool_entries(self) -> tuple[ToolCatalogEntry, ...]:
        search = self.search_entry
        return tuple(entry for entry in self.entries if entry is not search)

    @property
    def has_deferred(self) -> bool:
        selected = self.choice.name if self.choice.mode == "tool" else None
        return any(
            entry.deferred and not entry.loaded and entry.name != selected
            for entry in self.tool_entries()
        )

    def visible_entries(self) -> tuple[ToolCatalogEntry, ...]:
        selected = self.choice.name if self.choice.mode == "tool" else None
        visible = tuple(
            entry
            for entry in self.tool_entries()
            if not entry.deferred or entry.loaded or entry.name == selected
        )
        search = self.search_entry
        if self.has_deferred and search is not None:
            return (*visible, search)
        return visible

    @property
    def annotations(self) -> dict[str, dict[str, Any]]:
        """Return annotations for entries visible to portable tool protocols."""
        return {
            entry.name: deepcopy(dict(entry.annotations))
            for entry in self.visible_entries()
            if entry.annotations
        }

    def portable_schemas(self) -> list[dict[str, Any]]:
        """Return nested function schemas for function-only consumers."""
        return [entry.to_function_schema() for entry in self.visible_entries()]

    def cache_key_data(self) -> dict[str, Any]:
        """Return request-relevant catalog state without thread identity."""
        return {
            "library_id": self.library_id,
            "entries": [
                {
                    "schema": entry.to_function_schema(),
                    "native_bindings": msgspec.to_builtins(entry.native_bindings),
                    "catalog_role": entry.catalog_role,
                    "kind": entry.kind,
                    "deferred": entry.deferred,
                    "loaded": entry.loaded,
                }
                for entry in self.entries
            ],
            "choice": msgspec.to_builtins(self.choice),
        }

    def with_tools(self, names: Collection[str]) -> ToolCatalogView:
        """Return a view containing selected regular tools and the search entry."""
        included = set(names)
        available = {entry.name for entry in self.tool_entries()}
        unknown = included - available
        if unknown:
            formatted = ", ".join(f"`{name}`" for name in sorted(unknown))
            raise ValueError(f"Catalog tools are not available: {formatted}")
        entries = tuple(
            entry
            for entry in self.entries
            if entry.catalog_role == "search" or entry.name in included
        )
        choice = self.choice
        if choice.mode == "tool" and choice.name not in included:
            choice = ToolChoice()
        return msgspec.structs.replace(self, entries=entries, choice=choice)

    def with_choice(
        self,
        choice: ToolChoice | str | Mapping[str, Any] | None,
    ) -> ToolCatalogView:
        """Return a view with a normalized provider-neutral selection policy."""
        normalized = ToolChoice.coerce(choice)
        if not self.tool_entries():
            normalized = ToolChoice(mode="none")
        return msgspec.structs.replace(self, choice=normalized)


__all__ = [
    "NativeToolBinding",
    "ToolCatalogEntry",
    "ToolCatalogView",
    "ToolChoice",
    "ToolRef",
]
