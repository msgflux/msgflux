"""Local and MCP tool implementation adapters."""

import asyncio
import inspect
from contextvars import copy_context
from copy import deepcopy
from typing import Any, Callable, Dict, Mapping, Optional, get_type_hints

import msgflux.nn.functional as F
from msgflux.core.dotdict import dotdict
from msgflux.nn.modules.module import Module
from msgflux.nn.modules.tool.definitions import ContextSpec, ToolDeclaration
from msgflux.protocols.mcp import (
    convert_mcp_schema_to_tool_schema,
    extract_tool_result_text,
)
from msgflux.runtime.permissions import require_permissions
from msgflux.telemetry.span import aset_tool_attributes, set_tool_attributes
from msgflux.tools.helpers import (
    RUNTIME_BACKGROUND_PARAM,
    is_background_capable,
    normalize_background_capabilities,
)
from msgflux.tools.types import (
    ToolBucket,
    ToolLibraryOperator,
    unwrap_hidden_annotation,
)
from msgflux.utils.chat import generate_tool_json_schema
from msgflux.utils.inspect import fn_has_parameters, get_fn_param_defaults
from msgflux.utils.msgspec import restore_transport_value
from msgflux.utils.tenacity import apply_retry, default_tool_retry


class Tool(Module):
    """Tool is Module type that provide a json schema to tools."""

    _event_source_type = "tool"

    def get_json_schema(self):
        return generate_tool_json_schema(self)


class MCPTool(Tool):
    """MCP Tool Proxy - wraps remote MCP tool as a Tool object.

    This allows MCP tools to be treated exactly like local tools,
    enabling polymorphism and unified telemetry.

    Args:
        name: Tool name (without namespace prefix)
        mcp_client: Connected MCP client
        mcp_tool_info: MCP tool metadata
        namespace: MCP server namespace
        config: Optional tool configuration

    Example:
        >>> mcp_tool = MCPTool(
        ...     name="read_file",
        ...     mcp_client=client,
        ...     mcp_tool_info=tool_info,
        ...     namespace="filesystem"
        ... )
        >>> result = mcp_tool(path="/file.txt")
    """

    def __init__(
        self,
        name: str,
        mcp_client: Any,  # MCPClient type
        mcp_tool_info: Any,  # MCPToolInfo type
        namespace: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Set full tool name with namespace
        full_name = f"{namespace}__{name}"
        self.set_name(full_name)
        self.register_buffer(
            "display_name",
            (config.get("display_name") or full_name) if config else full_name,
        )
        self.register_buffer(
            "usage_guidance",
            config.get("usage_guidance") if config else None,
        )

        # Store MCP-specific data
        self._mcp_client = mcp_client
        self._mcp_tool_info = mcp_tool_info
        self._namespace = namespace
        self._mcp_tool_name = name

        # Set description from MCP tool info
        if hasattr(mcp_tool_info, "description"):
            self.set_description(mcp_tool_info.description)

        # Store config
        tc = {**(config or {})}
        tc.setdefault("tool_kind", "tool")
        self.register_buffer("tool_config", tc)

        # Apply retry
        retry_config = tc.get("retry")
        self.forward = apply_retry(
            self.forward, retry_config, default=default_tool_retry
        )
        self.aforward = apply_retry(
            self.aforward, retry_config, default=default_tool_retry
        )

    def get_json_schema(self) -> Dict[str, Any]:
        """Convert MCP tool schema to standard tool JSON schema."""
        return convert_mcp_schema_to_tool_schema(self._mcp_tool_info, self._namespace)

    @set_tool_attributes(execution_type="remote", protocol="mcp")
    def forward(self, **kwargs) -> Any:
        """Execute MCP tool call."""
        require_permissions(
            self.tool_config.get("required_permissions", ()),
            self.tool_config.get("required_resources", ()),
        )
        # Call MCP tool (wrap async in sync)
        result = F.wait_for(self._mcp_client.call_tool, self._mcp_tool_name, kwargs)

        # Handle errors
        if result.isError:
            error_text = extract_tool_result_text(result)
            raise RuntimeError(f"MCP tool error: {error_text}")

        # Extract and return result
        return extract_tool_result_text(result)

    @aset_tool_attributes(execution_type="remote", protocol="mcp")
    async def aforward(self, **kwargs) -> Any:
        """Execute MCP tool call asynchronously."""
        require_permissions(
            self.tool_config.get("required_permissions", ()),
            self.tool_config.get("required_resources", ()),
        )
        # Call MCP tool
        result = await self._mcp_client.call_tool(self._mcp_tool_name, kwargs)

        # Handle errors
        if result.isError:
            error_text = extract_tool_result_text(result)
            raise RuntimeError(f"MCP tool error: {error_text}")

        # Extract and return result
        return extract_tool_result_text(result)


class LocalTool(Tool):
    """Local tool implementation."""

    def __init__(
        self,
        name: str,
        description: str,
        annotations: Dict[str, Any],
        tool_config: Dict[str, Any],
        impl: Callable,
        display_name: Optional[str] = None,
        transport_params: Optional[Dict[str, Any]] = None,
        usage_guidance: Optional[str] = None,
        execution_namespace: Optional[str] = None,
    ):
        super().__init__()
        self.set_name(name)
        self.set_description(description)
        self.register_buffer("display_name", display_name or name)
        self.register_buffer("usage_guidance", usage_guidance)
        self.register_buffer("execution_namespace", execution_namespace)
        self.set_annotations(annotations)
        self.register_buffer("tool_config", tool_config)
        self.register_buffer("transport_params", transport_params or {})
        self.impl = impl  # Not a buffer for now
        self.register_buffer(
            "native_bindings", tuple(getattr(impl, "native_bindings", ()))
        )
        self._param_defaults = get_fn_param_defaults(impl)

        # Apply retry
        retry_config = tool_config.get("retry")
        self.forward = apply_retry(
            self.forward, retry_config, default=default_tool_retry
        )
        self.aforward = apply_retry(
            self.aforward, retry_config, default=default_tool_retry
        )

    def _restore_transport_params(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Restore transport-lowered tool params using the original annotations."""
        annotations = {
            name: hint
            for name, hint in self.get_module_annotations().items()
            if name != "return"
        }
        if not annotations:
            return kwargs
        restored = dict(kwargs)
        for param_name, type_hint in annotations.items():
            if param_name not in restored:
                continue
            restored[param_name] = restore_transport_value(
                restored[param_name],
                type_hint,
                restore_structs=True,
            )
        return restored

    def _strip_none_default_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Treat `null` tool arguments as omission when Python defaults exist.

        This mirrors the tool schema contract used by LocalTool/function tools:
        strict providers require every field in `required`, so optional/defaulted
        params are represented as nullable in the schema and mapped back to
        Python defaults here when the model emits `null`.
        """
        if not self._param_defaults:
            return kwargs
        return {
            key: value
            for key, value in kwargs.items()
            if not (key in self._param_defaults and value is None)
        }

    def _prepare_call_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = self._restore_transport_params(kwargs)
        return self._strip_none_default_kwargs(kwargs)

    @set_tool_attributes(execution_type="local")
    def forward(self, **kwargs):
        require_permissions(
            self.tool_config.get("required_permissions", ()),
            self.tool_config.get("required_resources", ()),
        )
        kwargs = self._prepare_call_kwargs(kwargs)
        if inspect.iscoroutinefunction(self.impl):
            return F.wait_for(self.impl, **kwargs)
        return self.impl(**kwargs)

    @aset_tool_attributes(execution_type="local")
    async def aforward(self, *args, **kwargs):
        require_permissions(
            self.tool_config.get("required_permissions", ()),
            self.tool_config.get("required_resources", ()),
        )
        kwargs = self._prepare_call_kwargs(kwargs)
        if hasattr(self.impl, "acall"):
            return await self.impl.acall(*args, **kwargs)
        elif inspect.iscoroutinefunction(self.impl):
            return await self.impl(*args, **kwargs)
        # Fall back to sync call in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        context = copy_context()
        return await loop.run_in_executor(
            None, lambda: context.run(self.impl, *args, **kwargs)
        )


def _inspect_tool_declaration(impl: Callable) -> ToolDeclaration:  # noqa: C901
    """Normalize a callable into the frontend declaration contract."""
    original_class = impl if inspect.isclass(impl) else None
    if original_class is not None:
        # Constructors may specialize tool_config for this instance.
        impl = impl()
        if not callable(impl):
            raise AttributeError("Tool class instances must define __call__")
    tool_config = dotdict(deepcopy(getattr(impl, "tool_config", dotdict())))
    if "spawn" in tool_config:
        raise ValueError("The `spawn` tool option was removed; use `detached`.")
    if ToolLibraryOperator.is_operator_tool(impl):
        inherited_config = getattr(type(impl), "tool_config", {})
        if inherited_config.get("runtime_inputs") is not None:
            tool_config["runtime_inputs"] = inherited_config["runtime_inputs"]
        for key in ("inject_handle", "inject_message", "inject_messages"):
            if inherited_config.get(key):
                tool_config[key] = True
    tool_config.setdefault("defer_loading", False)

    name_overridden = tool_config.pop("name_overridden", None)
    configured_description = tool_config.get("description")
    configured_display_name = tool_config.get("display_name")
    configured_usage_guidance = tool_config.get("usage_guidance")

    # Case 1: Uninitialized or initialized class
    if inspect.isclass(impl) or callable(impl):
        if not callable(impl):
            raise NotImplementedError(
                "To transform a class in `nn.Tool`"
                " is necessary implement a `def __call__`"
            )

        doc = (
            configured_description
            or getattr(impl, "description", None)
            or getattr(impl, "__doc__", None)
            or getattr(impl.__call__, "__doc__", None)
        )
        if doc is None:
            raise NotImplementedError(
                "To transform a class into a `nn.Tool` "
                "it is necessary to implement a docstring. "
                "Can be: a cls attr `self.docstring`, or"
                "a docstring in the class or in `def __call__`"
            )

        name = (
            name_overridden
            or getattr(impl, "name", None)
            or getattr(impl, "__name__", None)
            or getattr(original_class, "__name__", None)
        )
        display_name = configured_display_name or getattr(impl, "display_name", None)
        usage_guidance = configured_usage_guidance or getattr(
            impl, "usage_guidance", None
        )

        # Instantiate class first if needed, so we can get instance attributes
        class_annotation_source = original_class

        # Now extract annotations (after instantiation for classes)
        annotation_source = None
        has_bucket_annotations = isinstance(impl, ToolBucket) and (
            "annotations" in vars(type(impl)) or "annotations" in vars(impl)
        )
        annotations = getattr(impl, "annotations", None)
        if isinstance(impl, ToolBucket) and not has_bucket_annotations:
            annotations = getattr(impl.__call__, "__annotations__", None)
            annotation_source = impl.__call__ if annotations is not None else None
        if annotations is None:
            annotations = getattr(impl, "__annotations__", None)
            if annotations is not None:
                annotation_source = class_annotation_source or impl
        if annotations is None:
            annotations = getattr(impl.__call__, "__annotations__", None)
            if annotations is not None:
                annotation_source = impl.__call__
        if annotations is None:
            if fn_has_parameters(impl.__call__):
                raise NotImplementedError(
                    "To transform a class in `nn.Tool` is necessary "
                    "to implement annotations of types hint in "
                    "`self.annotations`, `self.__annotations__` or in `def __call__`"
                )
            annotations = {}
        annotations = _resolve_tool_annotations(annotation_source, annotations)

    # Case 2: Function
    elif inspect.isfunction(impl) or inspect.iscoroutinefunction(impl):
        if configured_description or (
            hasattr(impl, "__doc__") and impl.__doc__ is not None
        ):
            doc = configured_description or impl.__doc__
        else:
            raise NotImplementedError(
                "To transform a function into a `nn.Tool` "
                "is necessary to implement a docstring"
            )

        annotations = impl.__annotations__
        annotation_source = impl

        if annotations is None:
            if fn_has_parameters(impl):
                raise NotImplementedError(
                    "To transform a function into a `nn.Tool` "
                    "is necessary to implement parameters "
                    "annotations of types hint "
                )
            annotations = {}
        annotations = _resolve_tool_annotations(annotation_source, annotations)

        name = name_overridden or impl.__name__
        display_name = configured_display_name or getattr(impl, "display_name", None)
        usage_guidance = configured_usage_guidance or getattr(
            impl, "usage_guidance", None
        )

    else:
        raise ValueError(
            "The given object is not a callable function, class, or instance"
        )

    if tool_config.get("handoff", False):
        name = "transfer_to_" + name

    if isinstance(impl, ToolBucket):
        tool_kind = ToolBucket.tool_kind
    else:
        tool_kind = tool_config.get("tool_kind") or getattr(impl, "tool_kind", None)
        tool_kind = tool_kind or "tool"
    if not isinstance(tool_kind, str) or not tool_kind.strip():
        raise ValueError(f"The tool `{name}` must define a non-empty tool_kind.")
    tool_config["tool_kind"] = tool_kind

    declared_capabilities = tool_config.get("background_capabilities")
    if declared_capabilities is not None:
        if not is_background_capable(tool_config):
            raise ValueError(
                "`background_capabilities` requires `background=True` or "
                "`allow_background=True`."
            )
        tool_config["background_capabilities"] = normalize_background_capabilities(
            declared_capabilities
        )

    annotations, hidden_params = _split_hidden_annotations(annotations)
    if hidden_params:
        tool_config["_hidden_params"] = hidden_params

    if tool_config.get("handoff", False) or tool_config.get("disable_input", False):
        annotations = {}  # pass only the model state
    else:
        configured_context = tool_config.get("runtime_inputs")
        if configured_context is not None:
            for binding in ContextSpec.coerce(configured_context).bindings:
                annotations.pop(binding.parameter, None)
        if tool_config.get("inject_message", False):
            annotations.pop("message", None)
        if tool_config.get("inject_messages", False):
            annotations.pop("messages", None)
        if tool_config.get("inject_handle", False):
            annotations.pop("handle", None)
        if tool_config.get("inject_vars", False):
            annotations.pop("vars", None)
        if tool_config.get("allow_background", False) and not tool_config.get(
            "background", False
        ):
            annotations[RUNTIME_BACKGROUND_PARAM] = Optional[bool]

    if tool_config.get("detached"):
        doc = "This tool will not generate a return. \n" + doc
    if tool_config.get("background"):
        doc = "This tool runs in the background and returns a task id. \n" + doc
    elif tool_config.get("allow_background", False):
        doc = (
            "This tool can run in the background when "
            f"`{RUNTIME_BACKGROUND_PARAM}=true`; otherwise it runs normally. \n" + doc
        )

    return ToolDeclaration(
        name=name,
        description=doc,
        annotations=annotations,
        config=tool_config,
        implementation=impl,
        display_name=display_name or name,
        usage_guidance=usage_guidance,
        execution_namespace=(
            impl.get_module_name()
            if tool_kind == "agent" and hasattr(impl, "get_module_name")
            else None
        ),
    )


def _convert_declaration_to_local_tool(declaration: ToolDeclaration) -> LocalTool:
    return LocalTool(
        name=declaration.name,
        description=declaration.description,
        annotations=dict(declaration.annotations),
        tool_config=dotdict(declaration.config),
        impl=declaration.implementation,
        display_name=declaration.display_name,
        usage_guidance=declaration.usage_guidance,
        execution_namespace=declaration.execution_namespace,
    )


def _resolve_tool_annotations(
    annotation_source: Callable | None,
    annotations: Mapping[str, Any],
) -> Dict[str, Any]:
    resolved = dict(annotations)
    if annotation_source is None:
        return resolved
    try:
        type_hints = get_type_hints(annotation_source)
    except Exception:
        return resolved
    if not type_hints:
        return resolved
    resolved.update(type_hints)
    return resolved


def _split_hidden_annotations(
    annotations: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    public_annotations: Dict[str, Any] = {}
    hidden_params: Dict[str, Any] = {}
    for name, annotation in annotations.items():
        hidden_type = unwrap_hidden_annotation(annotation)
        if hidden_type is not None:
            if name == "return":
                raise ValueError("`Hidden[...]` cannot be used as a return type.")
            hidden_params[name] = hidden_type
            continue
        public_annotations[name] = annotation
    return public_annotations, hidden_params


def _convert_module_to_nn_tool(impl: Callable) -> Tool:
    """Convert a callable in nn.Tool."""
    return _convert_declaration_to_local_tool(_inspect_tool_declaration(impl))


def _declaration_from_tool(tool: Tool) -> ToolDeclaration:
    return ToolDeclaration(
        name=tool.name,
        description=tool.get_module_description() or "",
        annotations=tool.get_module_annotations(),
        config=getattr(tool, "tool_config", {}),
        implementation=getattr(tool, "impl", tool),
        display_name=getattr(tool, "display_name", None) or tool.name,
        usage_guidance=getattr(tool, "usage_guidance", None),
        execution_namespace=getattr(tool, "execution_namespace", None),
    )


__all__ = ["LocalTool", "MCPTool", "Tool"]
