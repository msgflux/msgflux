"""Canonical executable tool definitions and declaration compiler."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Collection, Mapping, Protocol, runtime_checkable

import msgspec

from msgflux.runtime.permissions import (
    ResourcePermission,
    normalize_permissions,
    normalize_resources,
)
from msgflux.tools.catalog import NativeToolBinding
from msgflux.tools.helpers import RUNTIME_BACKGROUND_PARAM
from msgflux.tools.runtime import FeedbackSpec, _copy_mapping, _require_name
from msgflux.tools.specs import ContextBinding, ContextSpec, DispatchSpec, LoadingSpec


@runtime_checkable
class ToolExecutor(Protocol):
    """Execution adapter owned by one logical tool definition.

    Local Python and remote MCP tools implement the same boundary. Dispatch
    extensions decide *when* the adapter runs; the adapter decides *how* the
    action reaches its implementation.
    """

    def __call__(self, **arguments: Any) -> Any: ...

    async def acall(self, **arguments: Any) -> Any: ...


class ToolDeclaration(msgspec.Struct, frozen=True, kw_only=True):
    """Normalized frontend declaration compiled into a ToolDefinition."""

    name: str
    description: str
    annotations: Mapping[str, Any]
    config: Mapping[str, Any]
    implementation: Callable[..., Any]
    display_name: str | None = None
    usage_guidance: str | None = None
    execution_namespace: str | None = None

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(self, "name", _require_name(self.name, "name"))
        if not isinstance(self.description, str):
            raise TypeError("`description` must be a string")
        msgspec.structs.force_setattr(
            self,
            "annotations",
            _copy_mapping(self.annotations, "annotations"),
        )
        msgspec.structs.force_setattr(
            self,
            "config",
            _copy_mapping(self.config, "config"),
        )
        if not callable(self.implementation):
            raise TypeError("`implementation` must be callable")


class ToolDefinition(msgspec.Struct, frozen=True, kw_only=True):
    """Stable logical declaration compiled before a tool can be executed."""

    name: str
    executor: ToolExecutor
    input_schema: Mapping[str, Any]
    description: str | None = None
    annotations: Mapping[str, Any] = msgspec.field(default_factory=dict)
    dispatch: DispatchSpec | str = msgspec.field(default_factory=DispatchSpec)
    feedback: FeedbackSpec | str = msgspec.field(default_factory=FeedbackSpec)
    context: ContextSpec | tuple[ContextBinding | str, ...] = msgspec.field(
        default_factory=ContextSpec
    )
    loading: LoadingSpec = msgspec.field(default_factory=LoadingSpec)
    retry: Any = None
    required_permissions: tuple[str, ...] = ()
    required_resources: tuple[ResourcePermission, ...] = ()
    native_bindings: tuple[NativeToolBinding, ...] = ()
    kind: str = "tool"
    display_name: str | None = None
    usage_guidance: str | None = None
    declaration: Mapping[str, Any] = msgspec.field(default_factory=dict)
    metadata: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(self, "name", _require_name(self.name, "name"))
        if not isinstance(self.executor, ToolExecutor):
            raise TypeError(
                "`executor` must provide synchronous `__call__` and async `acall`"
            )
        msgspec.structs.force_setattr(
            self,
            "required_permissions",
            normalize_permissions(self.required_permissions),
        )
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("`description` must be a string or None")
        msgspec.structs.force_setattr(
            self, "required_resources", normalize_resources(self.required_resources)
        )
        msgspec.structs.force_setattr(self, "kind", _require_name(self.kind, "kind"))
        msgspec.structs.force_setattr(
            self, "dispatch", DispatchSpec.coerce(self.dispatch)
        )
        msgspec.structs.force_setattr(
            self, "feedback", FeedbackSpec.coerce(self.feedback)
        )
        msgspec.structs.force_setattr(self, "context", ContextSpec.coerce(self.context))
        if not isinstance(self.loading, LoadingSpec):
            raise TypeError("`loading` must be a LoadingSpec")
        native_bindings = tuple(self.native_bindings)
        if not all(
            isinstance(binding, NativeToolBinding) for binding in native_bindings
        ):
            raise TypeError("`native_bindings` must contain NativeToolBinding values")
        native_keys = [
            (binding.provider, binding.api_mode, binding.kind)
            for binding in native_bindings
        ]
        if len(native_keys) != len(set(native_keys)):
            raise ValueError(
                "Native tool bindings must be unique per provider/API/kind"
            )
        msgspec.structs.force_setattr(self, "native_bindings", native_bindings)
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
        msgspec.structs.force_setattr(
            self,
            "declaration",
            _copy_mapping(self.declaration, "declaration"),
        )
        msgspec.structs.force_setattr(
            self,
            "metadata",
            _copy_mapping(self.metadata, "metadata"),
        )


class ToolDefinitionCompiler:
    """Compile legacy decorator metadata once into the canonical contract."""

    @classmethod
    def compile(
        cls,
        declaration: ToolDeclaration,
        *,
        executor: ToolExecutor,
    ) -> ToolDefinition:
        if not isinstance(declaration, ToolDeclaration):
            raise TypeError("`declaration` must be a ToolDeclaration")
        if not isinstance(executor, ToolExecutor):
            raise TypeError("`executor` must implement ToolExecutor")
        config = dict(declaration.config)
        config.setdefault("defer_loading", False)
        config.setdefault(
            "tool_kind",
            getattr(declaration.implementation, "tool_kind", "tool"),
        )
        input_schema, schema_metadata = cls._extract_executor_schema(executor)
        runtime_metadata = {
            **schema_metadata,
            "catalog_role": getattr(declaration.implementation, "catalog_role", None),
            "execution_namespace": declaration.execution_namespace,
            "declared_usage_guidance": declaration.usage_guidance,
            "bucket": cls._compile_bucket_presentation(declaration.implementation),
            "background_capabilities": config.get("background_capabilities"),
            "disable_input": bool(config.get("disable_input", False)),
            "hidden_params": config.get("_hidden_params"),
        }
        runtime_metadata = {
            key: value
            for key, value in runtime_metadata.items()
            if value is not None and value != {}
        }
        native_bindings = tuple(getattr(executor, "native_bindings", ()))
        return ToolDefinition(
            name=declaration.name,
            executor=executor,
            input_schema=input_schema,
            description=declaration.description,
            annotations=declaration.annotations,
            dispatch=cls._compile_dispatch(config),
            feedback=cls._compile_feedback(config),
            context=cls._compile_context(config),
            loading=LoadingSpec(deferred=bool(config.get("defer_loading", False))),
            retry=config.get("retry"),
            required_permissions=config.get("required_permissions", ()),
            required_resources=config.get("required_resources", ()),
            native_bindings=native_bindings,
            kind=config.get("tool_kind", "tool"),
            display_name=declaration.display_name,
            usage_guidance=declaration.usage_guidance,
            declaration=config,
            metadata=runtime_metadata,
        )

    @staticmethod
    def _compile_bucket_presentation(impl: Any) -> dict[str, Any]:
        """Project executor-owned details required by execution-free buckets."""
        model = getattr(impl, "model", None)
        if getattr(model, "msgflux_type", None) != "model_gateway":
            return {}
        return {
            "models": tuple(
                {
                    "name": model_name,
                    "description": model.get_model_description(model_name),
                }
                for model_name in model.model_names
            )
        }

    @classmethod
    def refresh_presentation(
        cls,
        definition: ToolDefinition,
        declaration: ToolDeclaration,
    ) -> ToolDefinition:
        """Refresh mutable bucket presentation without recompiling policies."""
        if not isinstance(definition, ToolDefinition):
            raise TypeError("`definition` must be a ToolDefinition")
        if not isinstance(declaration, ToolDeclaration):
            raise TypeError("`declaration` must be a ToolDeclaration")
        executor_impl = getattr(definition.executor, "impl", definition.executor)
        if declaration.implementation is not executor_impl:
            raise ValueError("Presentation refresh cannot change the implementation")

        input_schema, schema_metadata = cls._extract_executor_schema(
            definition.executor
        )
        runtime_metadata = dict(definition.metadata)
        runtime_metadata.pop("strict", None)
        runtime_metadata.update(schema_metadata)
        return msgspec.structs.replace(
            definition,
            input_schema=input_schema,
            description=declaration.description,
            annotations=declaration.annotations,
            display_name=declaration.display_name,
            usage_guidance=declaration.usage_guidance,
            metadata=runtime_metadata,
        )

    @classmethod
    def _extract_executor_schema(
        cls,
        executor: ToolExecutor,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        mcp_tool_info = getattr(executor, "_mcp_tool_info", None)
        if mcp_tool_info is not None:
            parameters = getattr(mcp_tool_info, "inputSchema", None)
            return (
                deepcopy(dict(parameters)) if isinstance(parameters, Mapping) else {},
                {},
            )
        get_json_schema = getattr(executor, "get_json_schema", None)
        if not callable(get_json_schema):
            raise TypeError("Compiled tool executors must provide `get_json_schema()`")
        return cls._extract_input_schema(get_json_schema())

    @staticmethod
    def _compile_dispatch(config: Mapping[str, Any]) -> DispatchSpec:
        if config.get("dispatch") is not None:
            return DispatchSpec.coerce(config["dispatch"])
        if config.get("background", False):
            return DispatchSpec(
                name="background",
                options={
                    "capabilities": config.get("background_capabilities"),
                },
            )
        if config.get("detached", False):
            return DispatchSpec(name="detached")
        if config.get("allow_background", False):
            return DispatchSpec(
                name="optional_background",
                options={
                    "argument": RUNTIME_BACKGROUND_PARAM,
                    "capabilities": config.get("background_capabilities"),
                },
            )
        return DispatchSpec(name="foreground")

    @staticmethod
    def _compile_feedback(config: Mapping[str, Any]) -> FeedbackSpec:
        if config.get("feedback") is not None:
            return FeedbackSpec.coerce(config["feedback"])
        if config.get("call_as_response", False):
            return FeedbackSpec(name="call_as_response")
        if config.get("handoff", False):
            return FeedbackSpec(name="handoff")
        if config.get("return_direct", False):
            return FeedbackSpec(name="direct")
        return FeedbackSpec(name="model")

    @staticmethod
    def _compile_context(config: Mapping[str, Any]) -> ContextSpec:
        configured = config.get("runtime_inputs")
        if configured is not None:
            context = ContextSpec.coerce(configured)
            bindings = []
            for binding in context.bindings:
                selected_binding = binding
                if (
                    binding.source == "messages"
                    and config.get("tool_kind") == "agent"
                    and "copy" not in binding.options
                ):
                    selected_binding = msgspec.structs.replace(
                        binding,
                        options={**binding.options, "copy": True},
                    )
                bindings.append(selected_binding)
            return ContextSpec(bindings=tuple(bindings))
        bindings = []
        for source in ("message", "messages", "handle"):
            if config.get(f"inject_{source}", False):
                options = (
                    {"copy": True}
                    if source == "messages" and config.get("tool_kind") == "agent"
                    else {}
                )
                bindings.append(ContextBinding(source=source, options=options))
        inject_vars = config.get("inject_vars", False)
        if inject_vars is not False:
            is_selection = isinstance(inject_vars, Collection) and not isinstance(
                inject_vars,
                (str, bytes, Mapping),
            )
            if is_selection and config.get("tool_kind") != "agent":
                bindings.extend(
                    ContextBinding(
                        source="vars",
                        parameter=name,
                        options={"key": name},
                    )
                    for name in inject_vars
                )
            else:
                options = {"select": tuple(inject_vars)} if is_selection else {}
                bindings.append(ContextBinding(source="vars", options=options))
        return ContextSpec(bindings=tuple(bindings))

    @staticmethod
    def _extract_input_schema(
        schema: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(schema, Mapping):
            raise TypeError("Tool schemas must be mappings")
        function = schema.get("function")
        if schema.get("type") == "function" and isinstance(function, Mapping):
            parameters = function.get("parameters", {})
            strict = function.get("strict")
        elif schema.get("type") == "function":
            parameters = schema.get("parameters", {})
            strict = schema.get("strict")
        else:
            raise ValueError("Tool schemas must use a function-tool shape")
        if not isinstance(parameters, Mapping):
            raise TypeError("Tool schema parameters must be a mapping")
        metadata = {"strict": strict} if strict is not None else {}
        return deepcopy(dict(parameters)), metadata


__all__ = [
    "ContextBinding",
    "ContextSpec",
    "DispatchSpec",
    "LoadingSpec",
    "ToolDefinition",
    "ToolDefinitionCompiler",
    "ToolExecutor",
]
