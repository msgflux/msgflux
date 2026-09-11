"""Provider-neutral tool catalog passed from agents to models."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Mapping

from msgspec import Struct

if TYPE_CHECKING:
    from msgflux.tools.catalog import ToolCatalogEntry, ToolCatalogView


class ToolSpec(Struct, kw_only=True):
    """Logical tool definition independent from a provider wire protocol."""

    name: str
    description: str | None = None
    parameters: Dict[str, Any] | None = None
    strict: bool | None = None
    annotations: Dict[str, Any] | None = None
    defer_loading: bool = False
    loaded: bool = False
    namespace: str | None = None
    ref: Any | None = None
    native_bindings: tuple[Any, ...] = ()
    kind: str = "tool"
    display_name: str | None = None
    usage_guidance: str | None = None
    catalog_role: str | None = None

    @classmethod
    def from_definition(
        cls,
        definition: Any,
        *,
        loaded: bool = False,
        namespace: str | None = None,
    ) -> ToolSpec:
        """Project one canonical runtime definition into a model catalog spec."""
        loading = getattr(definition, "loading", None)
        metadata = getattr(definition, "metadata", {})
        strict = metadata.get("strict") if isinstance(metadata, Mapping) else None
        return cls(
            name=definition.name,
            description=definition.description,
            parameters=deepcopy(dict(definition.input_schema)),
            strict=strict,
            annotations=deepcopy(dict(definition.annotations)) or None,
            defer_loading=bool(getattr(loading, "deferred", False)),
            loaded=loaded,
            namespace=namespace,
            native_bindings=tuple(getattr(definition, "native_bindings", ())),
            kind=getattr(definition, "kind", "tool"),
            display_name=getattr(definition, "display_name", None),
            usage_guidance=getattr(definition, "usage_guidance", None),
            catalog_role=metadata.get("catalog_role")
            if isinstance(metadata, Mapping)
            else None,
        )

    @classmethod
    def from_catalog_entry(cls, entry: ToolCatalogEntry) -> ToolSpec:
        """Adapt one canonical catalog entry for legacy provider consumers."""
        return cls(
            name=entry.name,
            description=entry.description,
            parameters=deepcopy(dict(entry.input_schema)),
            strict=entry.strict,
            annotations=deepcopy(dict(entry.annotations)) or None,
            defer_loading=entry.deferred,
            loaded=entry.loaded,
            namespace=entry.namespace,
            ref=entry.ref,
            native_bindings=tuple(entry.native_bindings),
            kind=entry.kind,
            display_name=entry.display_name,
            usage_guidance=entry.usage_guidance,
            catalog_role=entry.catalog_role,
        )

    @classmethod
    def from_function_schema(
        cls,
        schema: Mapping[str, Any],
        *,
        annotations: Mapping[str, Any] | None = None,
        defer_loading: bool = False,
        loaded: bool = False,
        namespace: str | None = None,
    ) -> ToolSpec:
        """Build a logical spec from an OpenAI-style function schema."""
        function = schema.get("function")
        if schema.get("type") != "function" or not isinstance(function, Mapping):
            raise ValueError("Tool schemas must use the function-tool shape")
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Function tool schemas require a non-empty name")
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise TypeError("Function tool parameters must be a mapping or None")
        strict = function.get("strict")
        if strict is not None and not isinstance(strict, bool):
            raise TypeError("Function tool strict must be bool or None")
        return cls(
            name=name,
            description=function.get("description"),
            parameters=deepcopy(dict(parameters)) if parameters is not None else None,
            strict=strict,
            annotations=deepcopy(dict(annotations)) if annotations else None,
            defer_loading=defer_loading,
            loaded=loaded,
            namespace=namespace,
        )

    def to_chat_completion_tool(self) -> Dict[str, Any]:
        """Compile this logical spec to the portable function-tool shape."""
        function: Dict[str, Any] = {"name": self.name}
        if self.description is not None:
            function["description"] = self.description
        if self.parameters is not None:
            function["parameters"] = deepcopy(self.parameters)
        if self.strict is not None:
            function["strict"] = self.strict
        return {"type": "function", "function": function}

    def to_responses_tool(self, *, native_deferred: bool = False) -> Dict[str, Any]:
        """Compile this logical spec to a Responses function-tool shape."""
        tool = self.to_chat_completion_tool()["function"]
        tool = {"type": "function", **tool}
        tool.setdefault("parameters", None)
        tool.setdefault("strict", False)
        if native_deferred and self.defer_loading and not self.loaded:
            tool["defer_loading"] = True
        return tool


class ToolCatalog(Struct, kw_only=True):
    """Thread-scoped logical tool surface consumed by provider compilers."""

    tools: List[ToolSpec]
    choice: str | Dict[str, Any] | None = None
    catalog_id: str | None = None
    search_tool: ToolSpec | None = None

    @classmethod
    def from_view(cls, view: ToolCatalogView) -> ToolCatalog:
        """Adapt one canonical thread view for legacy Agent and Model paths."""
        choice = None
        if view.choice.mode in {"none", "required"}:
            choice = view.choice.mode
        elif view.choice.mode == "tool":
            choice = view.choice.name
        return cls(
            tools=[ToolSpec.from_catalog_entry(entry) for entry in view.tool_entries()],
            choice=choice,
            catalog_id=view.library_id,
            search_tool=(
                ToolSpec.from_catalog_entry(view.search_entry)
                if view.search_entry is not None
                else None
            ),
        )

    @classmethod
    def from_function_schemas(
        cls,
        schemas: List[Mapping[str, Any]],
        *,
        choice: str | Dict[str, Any] | None = None,
        annotations: Mapping[str, Mapping[str, Any]] | None = None,
        catalog_id: str | None = None,
    ) -> ToolCatalog:
        """Convert legacy function schemas at the provider-neutral boundary."""
        annotation_map = annotations or {}
        tools = []
        for schema in schemas:
            function = schema.get("function")
            name = function.get("name") if isinstance(function, Mapping) else None
            tools.append(
                ToolSpec.from_function_schema(
                    schema,
                    annotations=annotation_map.get(name)
                    if isinstance(name, str)
                    else None,
                )
            )
        return cls(tools=tools, choice=choice, catalog_id=catalog_id)

    @property
    def annotations(self) -> Dict[str, Dict[str, Any]]:
        return {
            tool.name: deepcopy(tool.annotations)
            for tool in self.portable_tools()
            if tool.annotations
        }

    @property
    def has_deferred_tools(self) -> bool:
        return any(
            tool.defer_loading and not tool.loaded and not self.is_selected(tool)
            for tool in self.tools
        )

    def is_selected(self, tool: ToolSpec) -> bool:
        """Return whether tool choice explicitly selects this function."""
        if isinstance(self.choice, str):
            return self.choice not in {"auto", "required", "none"} and (
                self.choice == tool.name
            )
        if isinstance(self.choice, Mapping):
            function = self.choice.get("function")
            return isinstance(function, Mapping) and function.get("name") == tool.name
        return False

    def portable_tools(self) -> List[ToolSpec]:
        """Return tools visible to function-only providers for this thread."""
        visible = [
            tool
            for tool in self.tools
            if not tool.defer_loading or tool.loaded or self.is_selected(tool)
        ]
        if self.has_deferred_tools and self.search_tool is not None:
            visible.append(self.search_tool)
        return visible

    def portable_schemas(self) -> List[Dict[str, Any]]:
        return [tool.to_chat_completion_tool() for tool in self.portable_tools()]
