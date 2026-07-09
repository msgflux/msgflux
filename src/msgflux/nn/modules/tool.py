import asyncio
import inspect
from copy import deepcopy
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    get_type_hints,
)

import msgflux.nn.functional as F
from msgflux.auto import AutoParams
from msgflux.core.dotdict import dotdict
from msgflux.exceptions import TaskError, TaskInterruptRequestedError
from msgflux.logger import logger
from msgflux.nn.modules.container import ModuleDict
from msgflux.nn.modules.module import Module
from msgflux.protocols.mcp import (
    MCPClient,
    convert_mcp_schema_to_tool_schema,
    extract_tool_result_text,
    filter_tools,
)
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    InMemoryAgentInboxStore,
)
from msgflux.runtime.background import BackgroundTaskDispatcher
from msgflux.runtime.context import get_execution_context
from msgflux.runtime.tools.task import (
    AGENT_TASK_TOOLS,
    BASE_TASK_TOOLS,
)
from msgflux.tasks import InMemoryTaskStore
from msgflux.telemetry.span import (
    aset_tool_attributes,
    set_tool_attributes,
)
from msgflux.tools.builtin.tool_search import ToolSearchTool
from msgflux.tools.dataclasses import InternalToolState
from msgflux.tools.handles import ToolLibraryHandle
from msgflux.tools.helpers import (
    RESERVED_TOOL_KINDS as _RESERVED_TOOL_KINDS,
)
from msgflux.tools.helpers import (
    RUNTIME_BACKGROUND_PARAM as _RUNTIME_BACKGROUND_PARAM,
)
from msgflux.tools.helpers import (
    is_background_capable as _is_background_capable,
)
from msgflux.tools.helpers import (
    is_reserved_tool_kind as _is_reserved_tool_kind,
)
from msgflux.tools.helpers import (
    should_copy_injected_messages as _should_copy_injected_messages,
)
from msgflux.tools.responses import ToolCall, ToolResponses
from msgflux.tools.types import (
    ToolBackground,
    ToolBucket,
    ToolLibraryOperator,
    ToolMetadata,
    unwrap_hidden_annotation,
)
from msgflux.utils.chat import generate_tool_json_schema
from msgflux.utils.inspect import fn_has_parameters, get_fn_param_defaults
from msgflux.utils.msgspec import restore_transport_value
from msgflux.utils.tenacity import apply_retry, default_tool_retry


class Tool(Module):
    """Tool is Module type that provide a json schema to tools."""

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
        tc = config or {}
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
    ):
        super().__init__()
        self.set_name(name)
        self.set_description(description)
        self.register_buffer("display_name", display_name or name)
        self.register_buffer("usage_guidance", usage_guidance)
        self.set_annotations(annotations)
        self.register_buffer("tool_config", tool_config)
        self.register_buffer("transport_params", transport_params or {})
        self.impl = impl  # Not a buffer for now
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

    @set_tool_attributes(execution_type="local")
    def forward(self, **kwargs):
        kwargs = self._restore_transport_params(kwargs)
        kwargs = self._strip_none_default_kwargs(kwargs)
        if inspect.iscoroutinefunction(self.impl):
            return F.wait_for(self.impl, **kwargs)
        return self.impl(**kwargs)

    @aset_tool_attributes(execution_type="local")
    async def aforward(self, *args, **kwargs):
        kwargs = self._restore_transport_params(kwargs)
        kwargs = self._strip_none_default_kwargs(kwargs)
        if hasattr(self.impl, "acall"):
            return await self.impl.acall(*args, **kwargs)
        elif inspect.iscoroutinefunction(self.impl):
            return await self.impl(*args, **kwargs)
        # Fall back to sync call in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.impl(*args, **kwargs))


def _inspect_tool_metadata(impl: Callable) -> ToolMetadata:  # noqa: C901
    """Extract normalized metadata from a callable tool."""
    tool_config = dotdict(deepcopy(getattr(impl, "tool_config", dotdict())))

    name_overridden = tool_config.pop("name_overridden", None)
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
            getattr(impl, "description", None)
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
        )
        display_name = configured_display_name or getattr(impl, "display_name", None)
        usage_guidance = configured_usage_guidance or getattr(
            impl, "usage_guidance", None
        )

        # Instantiate class first if needed, so we can get instance attributes
        class_annotation_source = impl if inspect.isclass(impl) else None
        if inspect.isclass(impl):
            impl = impl()  # Initialized
            display_name = display_name or getattr(impl, "display_name", None)
            usage_guidance = usage_guidance or getattr(impl, "usage_guidance", None)

        # Now extract annotations (after instantiation for classes)
        annotation_source = None
        annotations = getattr(impl, "annotations", None)
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
        if hasattr(impl, "__doc__") and impl.__doc__ is not None:
            doc = impl.__doc__
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

    tool_config["tool_kind"] = getattr(impl, "tool_kind", None) or "tool"

    annotations, hidden_params = _split_hidden_annotations(annotations)
    if hidden_params:
        tool_config["_hidden_params"] = hidden_params

    if tool_config.get("handoff", False) or tool_config.get("disable_input", False):
        annotations = {}  # pass only the model state
    else:
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
            annotations[_RUNTIME_BACKGROUND_PARAM] = Optional[bool]

    if tool_config.get("spawn"):
        doc = "This tool will not generate a return. \n" + doc
    if tool_config.get("background"):
        doc = "This tool runs in the background and returns a task id. \n" + doc
    elif tool_config.get("allow_background", False):
        doc = (
            "This tool can run in the background when "
            f"`{_RUNTIME_BACKGROUND_PARAM}=true`; otherwise it runs normally. \n" + doc
        )

    return ToolMetadata(
        name=name,
        description=doc,
        annotations=annotations,
        tool_config=tool_config,
        impl=impl,
        display_name=display_name or name,
        usage_guidance=usage_guidance,
    )


def _convert_metadata_to_local_tool(metadata: ToolMetadata) -> LocalTool:
    return LocalTool(
        name=metadata.name,
        description=metadata.description,
        annotations=metadata.annotations,
        tool_config=metadata.tool_config,
        impl=metadata.impl,
        display_name=metadata.display_name,
        usage_guidance=metadata.usage_guidance,
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
    return _convert_metadata_to_local_tool(_inspect_tool_metadata(impl))


def _metadata_from_tool(tool: Tool) -> ToolMetadata:
    return ToolMetadata(
        name=tool.name,
        description=tool.get_module_description() or "",
        annotations=tool.get_module_annotations(),
        tool_config=getattr(tool, "tool_config", {}),
        impl=getattr(tool, "impl", tool),
        display_name=getattr(tool, "display_name", None) or tool.name,
        usage_guidance=getattr(tool, "usage_guidance", None),
        source_tool=tool,
    )


class ToolLibrary(Module, metaclass=AutoParams):
    """ToolLibrary is a Module type that manage tool calls over the tool library."""

    def __init__(
        self,
        name: str,
        tools: List[Callable],
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        task_store: Any | None = None,
    ):
        """Initialize the ToolLibrary.

        Args:
        name:
            Library name.
        tools:
            A list of callables.
        mcp_servers:
            List of MCP server configurations. Each config should contain:
            - name: Namespace for tools from this server
            - transport: "stdio" or "http"
            - For stdio: command, args, cwd, env
            - For http: base_url, headers
            - Optional: include_tools, exclude_tools, tool_config
        """
        super().__init__()
        self.set_name(f"{name}_tool_library")
        self.library = ModuleDict()
        self.register_buffer("tool_configs", {})
        self.register_buffer("on_demand_tools", {})
        self.register_buffer("mcp_clients", {})
        self._task_store = task_store
        self._agent_inbox: Optional[AgentInbox] = None
        self._bucket_tool_names_by_capture_kind: Dict[str, str] = {}
        self._internal_tool_state = InternalToolState()
        self._handle: Optional[ToolLibraryHandle] = None
        self._background_dispatcher: Optional[BackgroundTaskDispatcher] = None
        for tool in tools:
            self.add(tool)
        if mcp_servers:
            self._initialize_mcp_clients(mcp_servers)

    def get_handle(self) -> ToolLibraryHandle:
        if self._handle is None:
            self._handle = ToolLibraryHandle(self)
        return self._handle

    def get_background_dispatcher(self) -> BackgroundTaskDispatcher:
        if self._background_dispatcher is None:
            self._background_dispatcher = BackgroundTaskDispatcher(self.get_handle())
        return self._background_dispatcher

    def _get_default_task_store(self) -> Any:
        if self._task_store is None:
            self._task_store = InMemoryTaskStore()
        return self._task_store

    def get_agent_inbox(self) -> AgentInbox:
        if self._agent_inbox is None:
            self._agent_inbox = AgentInbox(
                owner=self.name,
                store=InMemoryAgentInboxStore(),
            )
        return self._agent_inbox

    def add(self, tool: Callable) -> str:
        """Add a local tool in library."""
        if isinstance(tool, ToolMetadata):
            metadata = tool
        elif isinstance(tool, Tool):
            metadata = _metadata_from_tool(tool)
        else:
            metadata = _inspect_tool_metadata(tool)

        if metadata.name in self.library.keys():
            raise ValueError(
                f"The tool name `{metadata.name}` is already in tool library"
            )
        if metadata.name in self.on_demand_tools:
            raise ValueError(
                f"The tool name `{metadata.name}` is already in on-demand tools"
            )

        # On-demand tools are searchable but not callable until tool_search promotes
        # them back through this same registration path.
        if metadata.tool_config.get("on_demand", False):
            self.on_demand_tools[metadata.name] = metadata
            self.tool_configs[metadata.name] = metadata.tool_config
            self._sync_on_demand_runtime_tools()
            return metadata.name

        # Buckets expose one public tool while absorbing tools of a matching kind.
        bucket_name = None
        if not ToolLibraryOperator.is_runtime_metadata(metadata):
            bucket_name = ToolBucket.find_bucket_for_metadata(
                metadata,
                self._bucket_tool_names_by_capture_kind,
                reserved_tool_kinds=_RESERVED_TOOL_KINDS,
            )
        if bucket_name is not None:
            ToolBucket.add_to_bucket(self.library[bucket_name], bucket_name, metadata)
            return metadata.name

        # Normal tools become directly callable and visible according to their config.
        self._register_tool_metadata(metadata)
        return metadata.name

    def remove(self, tool_name: str):
        if tool_name in self.library.keys():
            config = self.tool_configs.get(tool_name, {})
            is_task_tool = ToolBackground.is_installed_tool(
                tool_name,
                self._internal_tool_state,
            )
            is_tool_search = self._is_installed_tool_search(tool_name)
            was_background = _is_background_capable(config)
            was_background_agent = ToolBackground.is_agent_capable(
                self.library.get(tool_name),
                config,
            )

            self._remove_registered_tool(tool_name)

            if is_tool_search:
                self._internal_tool_state.tool_search_disabled = True
                return

            if is_task_tool:
                self._internal_tool_state.disabled_background_task_tool_names.add(
                    tool_name
                )
                self._internal_tool_state.background_task_tool_names.discard(tool_name)
                self._internal_tool_state.agent_task_tool_names.discard(tool_name)
                return

            if was_background:
                self._internal_tool_state.background_tool_names.discard(tool_name)
            if was_background_agent:
                self._internal_tool_state.background_agent_tool_names.discard(tool_name)
            if was_background or was_background_agent:
                ToolBackground.sync_task_tools(
                    library=self,
                    state=self._internal_tool_state,
                    base_tools=BASE_TASK_TOOLS,
                    agent_tools=AGENT_TASK_TOOLS,
                    metadata_factory=_inspect_tool_metadata,
                )
            self._sync_on_demand_runtime_tools()
        elif tool_name in self.on_demand_tools:
            self.on_demand_tools.pop(tool_name, None)
            self.tool_configs.pop(tool_name, None)
            self._sync_on_demand_runtime_tools()
        else:
            raise ValueError(f"The tool name `{tool_name}` is not in tool library")

    def _remove_registered_tool(self, tool_name: str) -> None:
        if tool_name in self.library:
            self.library.pop(tool_name)
        self.tool_configs.pop(tool_name, None)
        for capture_kind, bucket_name in list(
            self._bucket_tool_names_by_capture_kind.items()
        ):
            if bucket_name == tool_name:
                self._bucket_tool_names_by_capture_kind.pop(capture_kind, None)

    def clear(self):
        self.library.clear()
        self.tool_configs.clear()
        self.on_demand_tools.clear()
        self._bucket_tool_names_by_capture_kind.clear()
        for mcp_data in self.mcp_clients.values():
            F.wait_for(mcp_data["client"].disconnect)
        self.mcp_clients.clear()
        self._internal_tool_state.clear()
        if self._background_dispatcher is not None:
            self._background_dispatcher.clear()

    def _register_tool_metadata(self, metadata: ToolMetadata) -> Tool:
        capture_kind = ToolBucket.validate_registration(
            metadata,
            self._bucket_tool_names_by_capture_kind,
        )
        if capture_kind is not None:
            ToolBucket.validate_existing_captures(
                bucket_name=metadata.name,
                capture_kind=capture_kind,
                tools=self.library,
                tool_configs=self.tool_configs,
                metadata_factory=_metadata_from_tool,
                is_reserved_tool=self._is_reserved_or_runtime_tool,
            )

        tool = (
            metadata.source_tool
            if isinstance(metadata.source_tool, Tool)
            else _convert_metadata_to_local_tool(metadata)
        )
        self.tool_configs[tool.name] = getattr(tool, "tool_config", {})
        self.library.update({tool.name: tool})
        ToolBackground.record_registered_tool(
            tool.name,
            self.tool_configs[tool.name],
            self._internal_tool_state,
        )
        if tool.name == ToolSearchTool.name and ToolLibraryOperator.is_runtime_tool(
            tool
        ):
            self._internal_tool_state.tool_search_disabled = False
        self._apply_tool_registration_effects(tool.name)
        self._register_bucket_if_needed(tool.name, tool)
        return tool

    def _register_bucket_if_needed(self, tool_name: str, tool: Tool) -> None:
        config = self.tool_configs.get(tool_name, {})
        if config.get("tool_kind") != "bucket":
            return
        impl = getattr(tool, "impl", None)
        capture_kind = ToolBucket.require_capture_kind(tool_name, impl)
        self._bucket_tool_names_by_capture_kind[capture_kind] = tool_name
        for metadata in ToolBucket.pop_existing_captures(
            bucket_name=tool_name,
            capture_kind=capture_kind,
            tools=self.library,
            tool_configs=self.tool_configs,
            metadata_factory=_metadata_from_tool,
            is_reserved_tool=self._is_reserved_or_runtime_tool,
        ):
            ToolBucket.add_to_bucket(
                self.library[tool_name],
                tool_name,
                metadata,
            )

    @staticmethod
    def _is_reserved_or_runtime_tool(
        _tool_name: str,
        tool: Tool,
        config: Mapping[str, Any],
    ) -> bool:
        return _is_reserved_tool_kind(config) or ToolLibraryOperator.is_runtime_tool(
            tool
        )

    def _initialize_mcp_clients(self, mcp_servers: List[Dict[str, Any]]):
        """Initialize MCP clients from server configurations."""
        for server_config in mcp_servers:
            namespace = server_config.get("name")
            if not namespace:
                raise ValueError("MCP server config must include 'name' field")

            transport_type = server_config.get("transport", "stdio")

            # Create client based on transport type
            if transport_type == "stdio":
                command = server_config.get("command")
                if not command:
                    raise ValueError(
                        f"MCP server '{namespace}' stdio transport requires 'command'"
                    )
                client = MCPClient.from_stdio(
                    command=command,
                    args=server_config.get("args"),
                    cwd=server_config.get("cwd"),
                    env=server_config.get("env"),
                    timeout=server_config.get("timeout", 30.0),
                )
            elif transport_type == "http":
                base_url = server_config.get("base_url")
                if not base_url:
                    raise ValueError(
                        f"MCP server '{namespace}' http transport requires 'base_url'"
                    )
                client = MCPClient.from_http(
                    base_url=base_url,
                    timeout=server_config.get("timeout", 30.0),
                    headers=server_config.get("headers"),
                    auth=server_config.get("auth"),
                )
            else:
                raise ValueError(
                    f"Unknown transport type: {transport_type}. "
                    "Supported types: 'stdio', 'http'"
                )

            # Connect and list tools with error handling
            try:
                F.wait_for(client.connect)
                all_tools = F.wait_for(client.list_tools, use_cache=False)

                # Apply filters
                include_tools = server_config.get("include_tools")
                exclude_tools = server_config.get("exclude_tools")
                filtered_tools = filter_tools(all_tools, include_tools, exclude_tools)

                # Create MCPTool for each remote tool
                tool_configs = server_config.get("tool_config", {})
                for mcp_tool_info in filtered_tools:
                    tool_config = tool_configs.get(mcp_tool_info.name, {})

                    # Create MCPTool instance
                    mcp_tool = MCPTool(
                        name=mcp_tool_info.name,
                        mcp_client=client,
                        mcp_tool_info=mcp_tool_info,
                        namespace=namespace,
                        config=tool_config,
                    )

                    if mcp_tool.tool_config.get("on_demand", False):
                        metadata = _metadata_from_tool(mcp_tool)
                        self.on_demand_tools[mcp_tool.name] = metadata
                        self.tool_configs[mcp_tool.name] = mcp_tool.tool_config
                        self._sync_on_demand_runtime_tools()
                    else:
                        # Add to library (will have name like "namespace__tool_name")
                        self.library.update({mcp_tool.name: mcp_tool})
                        self.tool_configs[mcp_tool.name] = mcp_tool.tool_config
                        self._apply_tool_registration_effects(mcp_tool.name)

                self.mcp_clients[namespace] = {
                    "client": client,
                    "tools": filtered_tools,
                    "tool_config": tool_configs,
                }

                logger.debug(
                    f"Successfully connected to MCP server `{namespace}` "
                    f"with {len(filtered_tools)} tools"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize MCP server '{namespace}': {e!s}",
                    exc_info=True,
                )
                # Continue with other servers instead of failing completely

    def get_tools(self) -> Iterator[Dict[str, Tool]]:
        return self.library.items()

    def get_tool_names(self) -> List[str]:
        """Get names of all tools."""
        return list(self.library.keys()) + [
            name for name in self.on_demand_tools if name not in self.library
        ]

    def get_tool_display_names(self) -> Dict[str, str]:
        """Return human-readable display names keyed by registered tool name."""
        display_names = {}
        for tool_name, tool in self.library.items():
            display_names[tool_name] = getattr(tool, "display_name", None) or tool_name

        return display_names

    def get_tool_usage_guidance(
        self, tool_names: Optional[set[str]] = None
    ) -> List[Dict[str, str]]:
        """Return usage guidance metadata for tools that define it."""
        guidance = []
        display_names = self.get_tool_display_names()

        for tool_name, tool in self.library.items():
            if tool_names is not None and tool_name not in tool_names:
                continue
            usage_guidance = getattr(tool, "usage_guidance", None)
            if usage_guidance:
                guidance.append(
                    {
                        "name": tool_name,
                        "display_name": display_names.get(tool_name, tool_name),
                        "guidance": usage_guidance,
                    }
                )

        return guidance

    def get_mcp_tool_names(self) -> List[str]:
        """Get names of all MCP tools (with namespace)."""
        tool_names = []
        for namespace, mcp_data in self.mcp_clients.items():
            for tool in mcp_data["tools"]:
                tool_names.append(f"{namespace}__{tool.name}")
        return tool_names

    def get_tool_json_schemas(self) -> List[Dict[str, Any]]:
        """Returns a list of JSON schemas from local and MCP tools."""
        schemas = []
        for tool_name in self.library:
            if not self._is_tool_exposed(tool_name):
                continue
            schemas.append(self.library[tool_name].get_json_schema())

        return schemas

    def get_tool_annotations(self) -> Dict[str, Dict[str, Any]]:
        """Return local tool annotations keyed by tool name."""
        annotations = {}
        for tool_name, tool in self.library.items():
            if not self._is_tool_exposed(tool_name):
                continue
            annotations[tool_name] = {
                name: hint
                for name, hint in tool.get_module_annotations().items()
                if name != "return"
            }
        return annotations

    def set_agent_inbox(self, agent_inbox: AgentInbox) -> None:
        self._agent_inbox = agent_inbox

    def set_task_store(self, task_store: Any) -> None:
        if task_store is not None:
            self._task_store = task_store

    def get_task_store(self, task_store: Any = None) -> Any:
        if task_store is not None:
            return task_store
        context_task_store = get_execution_context().get("task_store")
        if context_task_store is not None:
            return context_task_store
        return self._get_default_task_store()

    # --- Tool Visibility Helpers ---

    def _apply_tool_registration_effects(self, tool_name: str) -> None:
        config = self.tool_configs.get(tool_name, {})
        if _is_reserved_tool_kind(config):
            return
        if _is_background_capable(config):
            self._internal_tool_state.background_tool_names.add(tool_name)
            if ToolBackground.is_agent_capable(self.library.get(tool_name), config):
                self._internal_tool_state.background_agent_tool_names.add(tool_name)
            ToolBackground.sync_task_tools(
                library=self,
                state=self._internal_tool_state,
                base_tools=BASE_TASK_TOOLS,
                agent_tools=AGENT_TASK_TOOLS,
                metadata_factory=_inspect_tool_metadata,
            )

    def _is_tool_exposed(self, tool_name: str) -> bool:
        return tool_name not in self.on_demand_tools

    def _sync_on_demand_runtime_tools(self) -> None:
        if self.on_demand_tools:
            self._ensure_on_demand_runtime_tools()
            return
        self._internal_tool_state.tool_search_disabled = False
        if not self._is_installed_tool_search(ToolSearchTool.name):
            return
        self._remove_registered_tool(ToolSearchTool.name)

    # --- Task Runtime Registration ---

    def _is_installed_tool_search(self, tool_name: str) -> bool:
        return (
            tool_name == ToolSearchTool.name
            and tool_name in self.library
            and ToolLibraryOperator.is_runtime_tool(self.library.get(tool_name))
        )

    def _ensure_on_demand_runtime_tools(self) -> None:
        if (
            self._is_installed_tool_search(ToolSearchTool.name)
            or self._internal_tool_state.tool_search_disabled
        ):
            return
        self.add(ToolSearchTool())

    # --- Tool Call Preparation ---

    def _build_call_params(  # noqa: C901
        self,
        *,
        tool: Tool,
        tool_name: str,
        tool_params: Any,
        config: Mapping[str, Any],
        message: Optional[Any],
        messages: List[Dict[str, Any]],
        vars: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if config.get("handoff", False) or config.get("disable_input", False):
            call_params: Dict[str, Any] = {}
        else:
            call_params = self._coerce_tool_params(tool_name, tool_params)

        for param_name in config.get("_hidden_params") or {}:
            call_params.pop(param_name, None)

        inject_vars = config.get("inject_vars", False)
        if inject_vars:
            if isinstance(inject_vars, list):
                for key in inject_vars:
                    if key not in vars:
                        raise ValueError(
                            f"The tool `{tool_name}` requires the injected "
                            f"parameter `{key}`, but it was not found."
                        )
                    call_params[key] = vars[key]
            elif inject_vars is True:
                call_params["vars"] = vars

        if config.get("inject_messages", False):
            if _should_copy_injected_messages(tool, config):
                call_params["messages"] = deepcopy(messages)
            else:
                call_params["messages"] = messages

        if config.get("inject_message", False):
            call_params["message"] = message

        if config.get("inject_handle", False):
            context = get_execution_context()
            call_params["handle"] = self.get_handle().for_tool(
                tool_name=tool_name,
                agent_inbox=context.get("agent_inbox"),
                task_store=context.get("task_store"),
            )

        return call_params

    def _should_dispatch_background(
        self,
        *,
        config: Mapping[str, Any],
        call_params: Dict[str, Any],
    ) -> bool:
        if config.get("background", False):
            call_params.pop(_RUNTIME_BACKGROUND_PARAM, None)
            return True
        if not config.get("allow_background", False):
            return False
        return call_params.pop(_RUNTIME_BACKGROUND_PARAM, False) is True

    @staticmethod
    def _coerce_tool_params(tool_name: str, tool_params: Any) -> Dict[str, Any]:
        if tool_params is None:
            return {}
        if isinstance(tool_params, Mapping):
            return dict(tool_params)
        raise TypeError(
            f"Tool `{tool_name}` parameters must be a mapping or None, "
            f"given `{type(tool_params)}`."
        )

    @staticmethod
    def build_call_parameters_for_response(
        params: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if params is None:
            return None
        if hasattr(params, "to_dict"):
            parameters = params.to_dict()
        else:
            parameters = dict(params)
        for key in (
            "vars",
            "messages",
            "message",
            "task",
            "notification",
            "scope",
            "handle",
            "tool_call_id",
            _RUNTIME_BACKGROUND_PARAM,
        ):
            parameters.pop(key, None)
        return parameters

    def _record_tool_activity(
        self,
        *,
        activity_recorder: Any,
        tool_name: str,
        parameters: Mapping[str, Any] | None,
    ) -> None:
        config = self.tool_configs.get(tool_name, {})
        if (
            activity_recorder is None
            or _is_reserved_tool_kind(config)
            or ToolLibraryOperator.is_runtime_tool(self.library.get(tool_name))
        ):
            return
        activity_recorder.tool_call(tool_name, parameters)

    def _prepare_tool_kwargs(
        self,
        *,
        tool: Tool,
        tool_name: str,
        tool_params: Any,
        config: Mapping[str, Any],
        message: Optional[Any],
        messages: List[Dict[str, Any]],
        vars: Mapping[str, Any],
        activity_recorder: Any,
    ) -> tuple[Dict[str, Any], dict[str, Any] | None]:
        call_params = self._build_call_params(
            tool=tool,
            tool_name=tool_name,
            tool_params=tool_params,
            config=config,
            message=message,
            messages=messages,
            vars=vars,
        )
        response_params = self.build_call_parameters_for_response(call_params)
        self._record_tool_activity(
            activity_recorder=activity_recorder,
            tool_name=tool_name,
            parameters=response_params,
        )
        return call_params, response_params

    def forward(  # noqa: C901
        self,
        tool_callings: List[Tuple[str, str, Any]],
        message: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> ToolResponses:
        """Executes tool calls with tool config logic.

        Args:
            tool_callings:
                A list of tuples containing the tool id, name and parameters.
                !!! example
                    [('123121', 'tool_name1', {'parameter1': 'value1'}),
                    ('322', 'tool_name2', '')]
            messages:
                The current messages (chat history) for the `handoff` functionality.
            message:
                The original message/envelope passed to the parent Agent.
            vars:
                Extra kwargs to be used in tools.

        Returns:
            ToolResponses:
                Structured object containing all tool call results.
        """
        if messages is None:
            messages = []

        if vars is None:
            vars = {}

        activity_recorder = get_execution_context().get("task_activity_recorder")
        prepared_calls = []
        call_metadata = []
        tool_calls: List[ToolCall] = []
        return_directly = True if tool_callings else False

        for tool_id, tool_name, tool_params in tool_callings:
            if tool_name not in self.library:
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
                        error=f"Error: Tool `{tool_name}` not found.",
                    )
                )
                return_directly = False
                continue

            # Get tool
            tool = self.library[tool_name]
            config = self.tool_configs.get(tool_name, {})
            call_params, response_params = self._prepare_tool_kwargs(
                tool=tool,
                tool_name=tool_name,
                tool_params=tool_params,
                config=config,
                message=message,
                messages=messages,
                vars=vars,
                activity_recorder=activity_recorder,
            )

            if config.get("spawn", False):
                return_directly = False
                F.spawn(tool, **call_params)
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=response_params,
                        result=f"The `{tool_name}` tool was dispatched. "
                        "This tool will not generate a return.",
                    )
                )
                continue

            if self._should_dispatch_background(
                config=config,
                call_params=call_params,
            ):
                return_directly = False
                tool_calls.append(
                    self.get_background_dispatcher().dispatch(
                        tool=tool,
                        tool_id=tool_id,
                        tool_name=tool_name,
                        call_params=call_params,
                        config=config,
                    )
                )
                continue

            if config.get(
                "call_as_response", False
            ):  # return function call as response
                tool_calls.append(
                    ToolCall(id=tool_id, name=tool_name, parameters=response_params)
                )
                return_directly = True
                continue

            if not config.get("return_direct", False):
                return_directly = False

            # Add tool_call_id for telemetry
            call_params["tool_call_id"] = tool_id
            prepared_calls.append(partial(tool, **call_params))

            call_metadata.append(
                dotdict(
                    id=tool_id,
                    name=tool_name,
                    config=config,
                    params=call_params,
                )
            )

        if prepared_calls:
            results = F.scatter_gather(prepared_calls)
            for meta, result in zip(call_metadata, results):
                if isinstance(result, TaskError) and isinstance(
                    result.exception, TaskInterruptRequestedError
                ):
                    raise result.exception
                parameters = self.build_call_parameters_for_response(meta.params)
                tool_calls.append(
                    ToolCall(
                        id=meta.id,
                        name=meta.name,
                        parameters=parameters,
                        result=None if isinstance(result, TaskError) else result,
                        error=str(result) if isinstance(result, TaskError) else None,
                    )
                )

        return ToolResponses(return_directly=return_directly, tool_calls=tool_calls)

    async def aforward(  # noqa: C901
        self,
        tool_callings: List[Tuple[str, str, Any]],
        message: Optional[Any] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> ToolResponses:
        """Async version of forward. Executes tool calls with logic for
        `handoff`, `return_direct`.

        Args:
            tool_callings:
                A list of tuples containing the tool id, name and parameters.
                !!! example
                    [('123121', 'tool_name1', {'parameter1': 'value1'}),
                    ('322', 'tool_name2', '')]
            messages:
                The current messages (chat history) for the `handoff` functionality.
            message:
                The original message/envelope passed to the parent Agent.
            vars:
                Extra kwargs to be used in tools.

        Returns:
            ToolResponses:
                Structured object containing all tool call results.
        """
        if messages is None:
            messages = []

        if vars is None:
            vars = {}

        activity_recorder = get_execution_context().get("task_activity_recorder")
        prepared_calls = []
        call_metadata = []
        tool_calls: List[ToolCall] = []
        return_directly = True if tool_callings else False

        for tool_id, tool_name, tool_params in tool_callings:
            if tool_name not in self.library:
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
                        error=f"Error: Tool `{tool_name}` not found.",
                    )
                )
                return_directly = False
                continue

            # Get tool
            tool = self.library[tool_name]
            config = self.tool_configs.get(tool_name, {})
            call_params, response_params = self._prepare_tool_kwargs(
                tool=tool,
                tool_name=tool_name,
                tool_params=tool_params,
                config=config,
                message=message,
                messages=messages,
                vars=vars,
                activity_recorder=activity_recorder,
            )

            if config.get("spawn", False):
                return_directly = False
                await F.aspawn(tool, **call_params)
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=response_params,
                        result=f"The `{tool_name}` tool was dispatched. "
                        "This tool will not generate a return.",
                    )
                )
                continue

            if self._should_dispatch_background(
                config=config,
                call_params=call_params,
            ):
                return_directly = False
                tool_calls.append(
                    self.get_background_dispatcher().dispatch(
                        tool=tool,
                        tool_id=tool_id,
                        tool_name=tool_name,
                        call_params=call_params,
                        config=config,
                    )
                )
                continue

            if config.get(
                "call_as_response", False
            ):  # return function call as response
                tool_calls.append(
                    ToolCall(id=tool_id, name=tool_name, parameters=response_params)
                )
                return_directly = True
                continue

            if not config.get("return_direct", False):
                return_directly = False

            # Add tool_call_id for telemetry
            call_params["tool_call_id"] = tool_id
            prepared_calls.append(partial(tool.acall, **call_params))

            call_metadata.append(
                dotdict(
                    id=tool_id,
                    name=tool_name,
                    config=config,
                    params=call_params,
                )
            )

        if prepared_calls:
            results = await F.ascatter_gather(prepared_calls)
            for meta, result in zip(call_metadata, results):
                if isinstance(result, TaskError) and isinstance(
                    result.exception, TaskInterruptRequestedError
                ):
                    raise result.exception
                parameters = self.build_call_parameters_for_response(meta.params)
                tool_calls.append(
                    ToolCall(
                        id=meta.id,
                        name=meta.name,
                        parameters=parameters,
                        result=None if isinstance(result, TaskError) else result,
                        error=str(result) if isinstance(result, TaskError) else None,
                    )
                )
        return ToolResponses(return_directly=return_directly, tool_calls=tool_calls)
