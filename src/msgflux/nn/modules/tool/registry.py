"""Stable executable tool registry and catalog projection."""

from __future__ import annotations

from typing import Any, Collection, Mapping

from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool.definitions import ToolDefinition
from msgflux.tools.catalog import ToolCatalogEntry, ToolCatalogView, ToolChoice, ToolRef
from msgflux.tools.runtime import _require_name


class ToolRegistry(Module):
    """Own stable logical definitions without owning executor modules."""

    def __init__(
        self,
        library_id: str,
        definitions: Collection[ToolDefinition] = (),
    ) -> None:
        super().__init__()
        self.library_id = _require_name(library_id, "library_id")
        self._definitions: dict[str, ToolDefinition] = {}
        for definition in definitions:
            self.add(definition)

    def add(self, definition: ToolDefinition) -> ToolRef:
        if not isinstance(definition, ToolDefinition):
            raise TypeError("`definition` must be a ToolDefinition")
        if definition.name in self._definitions:
            raise ValueError(f"Tool `{definition.name}` is already registered")
        if not isinstance(definition.executor, Module):
            raise TypeError("Tool executors must inherit msgflux.nn.Module")
        self._definitions[definition.name] = definition
        return self.ref(definition.name)

    def replace(self, definition: ToolDefinition) -> ToolDefinition:
        """Replace one definition while preserving its stable registry name."""
        if not isinstance(definition, ToolDefinition):
            raise TypeError("`definition` must be a ToolDefinition")
        if definition.name not in self._definitions:
            raise ValueError(f"Tool `{definition.name}` is not registered")
        if not isinstance(definition.executor, Module):
            raise TypeError("Tool executors must inherit msgflux.nn.Module")
        previous = self._definitions[definition.name]
        if definition.executor is not previous.executor:
            raise ValueError(
                "Replacing a definition cannot change its executor; remove and "
                "add the tool through ToolLibrary instead"
            )
        self._definitions[definition.name] = definition
        return previous

    def remove(self, tool: ToolRef | str) -> ToolDefinition:
        name = self._resolve_name(tool)
        try:
            definition = self._definitions.pop(name)
        except KeyError as exc:
            raise ValueError(f"Tool `{name}` is not registered") from exc
        return definition

    def clear(self) -> None:
        self._definitions.clear()

    def get(self, tool: ToolRef | str) -> ToolDefinition:
        name = self._resolve_name(tool)
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise ValueError(f"Tool `{name}` is not registered") from exc

    def has(self, tool: ToolRef | str) -> bool:
        try:
            name = self._resolve_name(tool)
        except ValueError:
            return False
        return name in self._definitions

    def ref(self, name: str) -> ToolRef:
        return ToolRef(library_id=self.library_id, tool_id=name)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._definitions.values())

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
                definition,
                library_id=self.library_id,
                loaded=definition.name in loaded,
            )
            for definition in self._definitions.values()
            if definition.name in included
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
