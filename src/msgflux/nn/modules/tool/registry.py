"""Stable executable tool registry and catalog projection."""

from __future__ import annotations

from typing import Any, Collection, Mapping

import msgspec

from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool.definitions import ToolDefinition
from msgflux.nn.modules.tool.implementations import MCPTool
from msgflux.nn.parameter import Parameter
from msgflux.tools.catalog import ToolCatalogEntry, ToolCatalogView, ToolChoice, ToolRef
from msgflux.tools.runtime import _require_name
from msgflux.tools.types import ToolBucket


class ToolRegistry(Module):
    """Own stable logical definitions without owning executor modules."""

    def __init__(
        self,
        library_id: str,
        definitions: Collection[ToolDefinition] = (),
    ) -> None:
        super().__init__()
        self.library_id = _require_name(library_id, "library_id")
        self._captured: set[str] = set()
        self._definitions: dict[str, ToolDefinition] = {}
        for definition in definitions:
            self.add(definition)

    _text_fields = ("description", "usage_guidance")

    @staticmethod
    def _text_key(name: str, field: str) -> str:
        return field + "__" + name.replace("%", "%25").replace(".", "%2E")

    def _store(self, definition: ToolDefinition) -> None:
        name = definition.name
        managed_externally = (
            name in self._captured
            or definition.loading.deferred
            or definition.native_bindings
            or isinstance(definition.executor, MCPTool)
            or isinstance(getattr(definition.executor, "impl", None), ToolBucket)
        )
        for field in self._text_fields:
            key = self._text_key(name, field)
            text = getattr(definition, field) or ""
            if managed_externally:
                self._parameters.pop(key, None)
            elif key in self._parameters:
                self._parameters[key].copy_to_data(text)
            else:
                self.register_parameter(
                    key, Parameter(text, f"tool_{field}", requires_grad=False)
                )
        self._definitions[name] = definition

    def text_parameter(self, name: str, field: str) -> Parameter:
        try:
            return self._parameters[self._text_key(name, field)]
        except KeyError as exc:
            raise ValueError(
                f"Tool {name!r} requires a portable local declaration"
            ) from exc

    def named_text_parameters(self):
        for name in self._definitions:
            for field in self._text_fields:
                parameter = self._parameters.get(self._text_key(name, field))
                if parameter is not None:
                    yield name, field, parameter

    def capture(self, name: str) -> None:
        """Keep the effective text when this tool moves behind a bucket."""
        definition = self.get(name)
        self._captured.add(name)
        self._store(definition)

    def add(self, definition: ToolDefinition, *, captured: bool = False) -> ToolRef:
        if not isinstance(definition, ToolDefinition):
            raise TypeError("`definition` must be a ToolDefinition")
        if definition.name in self._definitions:
            raise ValueError(f"Tool `{definition.name}` is already registered")
        if not isinstance(definition.executor, Module):
            raise TypeError("Tool executors must inherit msgflux.nn.Module")
        if captured:
            self._captured.add(definition.name)
        self._store(definition)
        return self.ref(definition.name)

    def replace(self, definition: ToolDefinition) -> ToolDefinition:
        """Replace one definition while preserving its stable registry name."""
        if not isinstance(definition, ToolDefinition):
            raise TypeError("`definition` must be a ToolDefinition")
        if definition.name not in self._definitions:
            raise ValueError(f"Tool `{definition.name}` is not registered")
        if not isinstance(definition.executor, Module):
            raise TypeError("Tool executors must inherit msgflux.nn.Module")
        previous = self.get(definition.name)
        if definition.executor is not previous.executor:
            raise ValueError(
                "Replacing a definition cannot change its executor; remove and "
                "add the tool through ToolLibrary instead"
            )
        self._store(definition)
        return previous

    def remove(self, tool: ToolRef | str) -> ToolDefinition:
        name = self._resolve_name(tool)
        definition = self.get(name)
        del self._definitions[name]
        for field in self._text_fields:
            self._parameters.pop(self._text_key(name, field), None)
        self._captured.discard(name)
        return definition

    def clear(self) -> None:
        for name, field, _ in self.named_text_parameters():
            del self._parameters[self._text_key(name, field)]
        self._definitions.clear()
        self._captured.clear()

    def get(self, tool: ToolRef | str) -> ToolDefinition:
        name = self._resolve_name(tool)
        try:
            definition = self._definitions[name]
        except KeyError as exc:
            raise ValueError(f"Tool `{name}` is not registered") from exc
        changes = {}
        for field in self._text_fields:
            parameter = self._parameters.get(self._text_key(name, field))
            if parameter is not None and parameter.data != (
                getattr(definition, field) or ""
            ):
                changes[field] = parameter.data
        return msgspec.structs.replace(definition, **changes) if changes else definition

    def has(self, tool: ToolRef | str) -> bool:
        try:
            name = self._resolve_name(tool)
        except ValueError:
            return False
        return name in self._definitions

    def ref(self, name: str) -> ToolRef:
        return ToolRef(library_id=self.library_id, tool_id=name)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self.get(name) for name in self._definitions)

    def catalog_view(
        self,
        thread_id: str,
        *,
        loaded_tools: Collection[str] = (),
        choice: ToolChoice | str | Mapping[str, Any] | None = None,
        include_tools: Collection[str] | None = None,
    ) -> ToolCatalogView:
        included = (
            set(self._definitions) if include_tools is None else set(include_tools)
        )
        unknown_included = included - self._definitions.keys()
        if unknown_included:
            formatted = ", ".join(f"`{name}`" for name in sorted(unknown_included))
            raise ValueError(f"Catalog tools are not registered: {formatted}")
        loaded = set(loaded_tools)
        unknown = loaded - included
        if unknown:
            formatted = ", ".join(f"`{name}`" for name in sorted(unknown))
            raise ValueError(f"Loaded tools are not registered: {formatted}")
        non_deferred = {
            name for name in loaded if not self._definitions[name].loading.deferred
        }
        if non_deferred:
            formatted = ", ".join(f"`{name}`" for name in sorted(non_deferred))
            raise ValueError(f"Only deferred tools can be loaded: {formatted}")
        entries = tuple(
            ToolCatalogEntry.from_definition(
                self.get(name),
                library_id=self.library_id,
                loaded=name in loaded,
            )
            for name in self._definitions
            if name in included
        )
        return ToolCatalogView(
            library_id=self.library_id,
            thread_id=thread_id,
            entries=entries,
            choice=choice if choice is not None else ToolChoice(),
        )

    def _resolve_name(self, tool: ToolRef | str) -> str:
        if isinstance(tool, ToolRef):
            if tool.library_id != self.library_id:
                raise ValueError(
                    f"Tool ref belongs to `{tool.library_id}`, not `{self.library_id}`"
                )
            return tool.tool_id
        return _require_name(tool, "tool")


__all__ = ["ToolRegistry"]
