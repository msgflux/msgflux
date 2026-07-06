import asyncio
import inspect
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

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
)
from msgflux.runtime.background import BackgroundTaskDispatcher
from msgflux.runtime.context import get_execution_context
from msgflux.runtime.task_tools import TaskRuntimeTools
from msgflux.tasks import InMemoryTaskStore
from msgflux.telemetry.span import (
    aset_tool_attributes,
    set_tool_attributes,
)
from msgflux.tools.builtin.tool_search import ToolSearchTool
from msgflux.tools.handles import ToolLibraryHandle
from msgflux.tools.helpers import (
    RUNTIME_BACKGROUND_PARAM as _RUNTIME_BACKGROUND_PARAM,
)
from msgflux.tools.helpers import (
    should_copy_injected_messages as _should_copy_injected_messages,
)
from msgflux.tools.helpers import (
    uses_handle_injection as _uses_handle_injection,
)
from msgflux.tools.responses import ToolCall, ToolResponses
from msgflux.tools.types import ToolLibraryOperator, ToolMetadata
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
        if inspect.isclass(impl):
            impl = impl()  # Initialized
            display_name = display_name or getattr(impl, "display_name", None)
            usage_guidance = usage_guidance or getattr(impl, "usage_guidance", None)

        # Now extract annotations (after instantiation for classes)
        annotations = (
            getattr(impl, "annotations", None)
            or getattr(impl, "__annotations__", None)
            or getattr(impl.__call__, "__annotations__", None)
        )
        if annotations is None:
            if fn_has_parameters(impl.__call__):
                raise NotImplementedError(
                    "To transform a class in `nn.Tool` is necessary "
                    "to implement annotations of types hint in "
                    "`self.annotations`, `self.__annotations__` or in `def __call__`"
                )
            annotations = {}

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

        if annotations is None:
            if fn_has_parameters(impl):
                raise NotImplementedError(
                    "To transform a function into a `nn.Tool` "
                    "is necessary to implement parameters "
                    "annotations of types hint "
                )
            annotations = {}

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
    if (
        isinstance(impl, ToolLibraryOperator)
        or getattr(impl, "inject_handle", False)
    ):
        tool_config["inject_handle"] = True

    annotations = dict(annotations)

    if tool_config.get("handoff", False) or tool_config.get("disable_input", False):
        annotations = {}  # pass only the model state
    else:
        if tool_config.get("inject_message", False):
            annotations.pop("message", None)
        if tool_config.get("inject_messages", False):
            annotations.pop("messages", None)
        if tool_config.get("inject_vars", False):
            annotations.pop("vars", None)
        if tool_config.get("inject_task", False):
            annotations.pop("task", None)
        if tool_config.get("inject_notification", False):
            annotations.pop("notification", None)
        if _uses_handle_injection(tool_config):
            annotations.pop("handle", None)
        if (
            tool_config.get("allow_background", False)
            and not tool_config.get("background", False)
        ):
            annotations[_RUNTIME_BACKGROUND_PARAM] = Optional[bool]

    if tool_config.get("spawn"):
        doc = "This tool will not generate a return. \n" + doc
    if tool_config.get("background"):
        doc = "This tool runs in the background and returns a task id. \n" + doc
    elif tool_config.get("allow_background", False):
        doc = (
            "This tool can run in the background when "
            f"`{_RUNTIME_BACKGROUND_PARAM}=true`; otherwise it runs normally. \n"
            + doc
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
        self._runtime_tool_names: set[str] = set()
        self._bucket_tool_names_by_capture_kind: Dict[str, str] = {}
        self._task_runtime_enabled = False
        self._agent_task_runtime_enabled = False
        self._handle: Optional[ToolLibraryHandle] = None
        self._background_dispatcher: Optional[BackgroundTaskDispatcher] = None
        self._task_runtime_tools: Optional[TaskRuntimeTools] = None
        self._tool_search_enabled = False
        for tool in tools:
            self.add(tool)
        if mcp_servers:
            self._initialize_mcp_clients(mcp_servers)

    @property
    def handle(self) -> ToolLibraryHandle:
        if self._handle is None:
            self._handle = ToolLibraryHandle(self)
        return self._handle

    @property
    def background_dispatcher(self) -> BackgroundTaskDispatcher:
        if self._background_dispatcher is None:
            self._background_dispatcher = BackgroundTaskDispatcher(self.handle)
        return self._background_dispatcher

    @property
    def task_runtime_tools(self) -> TaskRuntimeTools:
        if self._task_runtime_tools is None:
            self._task_runtime_tools = TaskRuntimeTools(
                self.handle,
            )
        return self._task_runtime_tools

    @property
    def task_store(self) -> Any:
        if self._task_store is None:
            self._task_store = InMemoryTaskStore()
        return self._task_store

    @property
    def agent_inbox(self) -> AgentInbox:
        if self._agent_inbox is None:
            self._agent_inbox = AgentInbox(owner=self.name)
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
        bucket_name = self._find_bucket_for_metadata(metadata)
        if bucket_name is not None:
            self._add_tool_to_bucket(bucket_name, metadata)
            return metadata.name

        # Normal tools become directly callable and visible according to their config.
        self._register_tool_metadata(metadata)
        return metadata.name

    def remove(self, tool_name: str):
        if tool_name in self.library.keys():
            self.library.pop(tool_name)
            self.tool_configs.pop(tool_name, None)
            for capture_kind, bucket_name in list(
                self._bucket_tool_names_by_capture_kind.items()
            ):
                if bucket_name == tool_name:
                    self._bucket_tool_names_by_capture_kind.pop(capture_kind, None)
            self._sync_on_demand_runtime_tools()
        elif tool_name in self.on_demand_tools:
            self.on_demand_tools.pop(tool_name, None)
            self.tool_configs.pop(tool_name, None)
            self._sync_on_demand_runtime_tools()
        else:
            raise ValueError(f"The tool name `{tool_name}` is not in tool library")

    def clear(self):
        self.library.clear()
        self.tool_configs.clear()
        self.on_demand_tools.clear()
        self._bucket_tool_names_by_capture_kind.clear()
        for mcp_data in self.mcp_clients.values():
            F.wait_for(mcp_data["client"].disconnect)
        self.mcp_clients.clear()
        self._task_runtime_enabled = False
        self._agent_task_runtime_enabled = False
        self._runtime_tool_names.clear()
        if self._background_dispatcher is not None:
            self._background_dispatcher.clear()
        if self._task_runtime_tools is not None:
            self._task_runtime_tools.reset()
        self._tool_search_enabled = False

    def _register_tool_metadata(self, metadata: ToolMetadata) -> Tool:
        if metadata.tool_config.get("tool_kind") == "bucket":
            capture_kind = getattr(metadata.impl, "capture_kind", None)
            if not isinstance(capture_kind, str) or not capture_kind:
                raise ValueError(
                    f"The bucket tool `{metadata.name}` must define capture_kind."
                )
            existing = self._bucket_tool_names_by_capture_kind.get(capture_kind)
            if existing is not None and existing != metadata.name:
                raise ValueError(
                    f"The bucket capture kind `{capture_kind}` is already handled by "
                    f"`{existing}`."
                )
            self._validate_existing_tools_for_bucket(metadata.name, capture_kind)

        tool = (
            metadata.source_tool
            if isinstance(metadata.source_tool, Tool)
            else _convert_metadata_to_local_tool(metadata)
        )
        self.tool_configs[tool.name] = getattr(tool, "tool_config", {})
        self.library.update({tool.name: tool})
        self._apply_tool_registration_effects(tool.name)
        self._register_bucket_if_needed(tool.name, tool)
        return tool

    def _register_bucket_if_needed(self, tool_name: str, tool: Tool) -> None:
        config = self.tool_configs.get(tool_name, {})
        if config.get("tool_kind") != "bucket":
            return
        impl = getattr(tool, "impl", None)
        capture_kind = getattr(impl, "capture_kind", None)
        if not isinstance(capture_kind, str) or not capture_kind:
            raise ValueError(f"The bucket tool `{tool_name}` must define capture_kind.")
        self._bucket_tool_names_by_capture_kind[capture_kind] = tool_name
        self._capture_existing_tools_for_bucket(tool_name, capture_kind)

    def _find_bucket_for_metadata(self, metadata: ToolMetadata) -> Optional[str]:
        tool_kind = metadata.tool_config.get("tool_kind", "tool")
        if tool_kind == "bucket":
            return None
        return self._bucket_tool_names_by_capture_kind.get(tool_kind)

    def _add_tool_to_bucket(self, bucket_name: str, metadata: ToolMetadata) -> None:
        self._validate_bucket_capture(bucket_name, metadata)
        bucket_tool = self.library[bucket_name]
        bucket_impl = getattr(bucket_tool, "impl", None)
        if bucket_impl is None or not hasattr(bucket_impl, "add"):
            raise ValueError(f"The bucket tool `{bucket_name}` cannot capture tools.")
        bucket_impl.add(metadata)
        self._refresh_bucket_tool(bucket_name)

    @staticmethod
    def _validate_bucket_capture(bucket_name: str, metadata: ToolMetadata) -> None:
        if metadata.tool_config.get("background", False) or metadata.tool_config.get(
            "allow_background", False
        ):
            raise ValueError(
                "Bucket-captured tools cannot use `background=True` or "
                f"`allow_background=True`. Tool `{metadata.name}` would be captured "
                f"by bucket `{bucket_name}`."
            )

    def _refresh_bucket_tool(self, bucket_name: str) -> None:
        bucket_tool = self.library[bucket_name]
        bucket_impl = getattr(bucket_tool, "impl", None)
        if bucket_impl is None:
            return
        description = getattr(bucket_impl, "description", None)
        if isinstance(description, str):
            bucket_tool.set_description(description)
        bucket_tool.register_buffer(
            "usage_guidance",
            getattr(bucket_impl, "usage_guidance", None),
        )

    def _capture_existing_tools_for_bucket(
        self,
        bucket_name: str,
        capture_kind: str,
    ) -> None:
        for tool_name, tool in list(self.library.items()):
            if tool_name == bucket_name or tool_name in self._runtime_tool_names:
                continue
            config = self.tool_configs.get(tool_name, {})
            if config.get("tool_kind") != capture_kind:
                continue
            metadata = _metadata_from_tool(tool)
            self.library.pop(tool_name)
            self.tool_configs.pop(tool_name, None)
            self._add_tool_to_bucket(bucket_name, metadata)

    def _validate_existing_tools_for_bucket(
        self,
        bucket_name: str,
        capture_kind: str,
    ) -> None:
        for tool_name, tool in self.library.items():
            if tool_name == bucket_name or tool_name in self._runtime_tool_names:
                continue
            config = self.tool_configs.get(tool_name, {})
            if config.get("tool_kind") != capture_kind:
                continue
            self._validate_bucket_capture(bucket_name, _metadata_from_tool(tool))

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

    def _sync_task_store_from_context(self) -> None:
        task_store = get_execution_context().get("task_store")
        if task_store is not None:
            self.set_task_store(task_store)
        elif self._task_store is None and self._task_runtime_enabled:
            self._task_store = InMemoryTaskStore()

    # --- Tool Visibility Helpers ---

    def _apply_tool_registration_effects(self, tool_name: str) -> None:
        config = self.tool_configs.get(tool_name, {})
        can_run_background = config.get("background", False) or config.get(
            "allow_background", False
        )
        if can_run_background:
            # Background-capable tools need the shared task control surface.
            self.task_runtime_tools.ensure_base_tools()
            self._task_runtime_enabled = True
        tool = self.library.get(tool_name)
        impl = getattr(tool, "impl", None) if tool is not None else None
        supports_task_message = bool(getattr(impl, "supports_task_message", False))
        if can_run_background and (
            config.get("tool_kind") == "agent" or supports_task_message
        ):
            # Background agents also expose activity and message-resume controls.
            self.task_runtime_tools.ensure_agent_tools()
            self._agent_task_runtime_enabled = True

    def _is_tool_exposed(self, tool_name: str) -> bool:
        if tool_name in self._runtime_tool_names:
            return True
        return tool_name not in self.on_demand_tools

    def _sync_on_demand_runtime_tools(self) -> None:
        if self.on_demand_tools:
            self._ensure_on_demand_runtime_tools()
            return
        if not self._tool_search_enabled:
            return
        self._tool_search_enabled = False
        self.handle.runtime_tool_names.discard(ToolSearchTool.name)
        if ToolSearchTool.name in self.library:
            self.library.pop(ToolSearchTool.name)
        self.tool_configs.pop(ToolSearchTool.name, None)

    # --- Task Runtime Registration ---

    def _ensure_task_runtime_tools(self) -> None:
        self.task_runtime_tools.ensure_base_tools()
        self._task_runtime_enabled = True

    def _ensure_agent_task_runtime_tools(self) -> None:
        self.task_runtime_tools.ensure_agent_tools()
        self._agent_task_runtime_enabled = True

    def _ensure_on_demand_runtime_tools(self) -> None:
        if self._tool_search_enabled:
            return
        self._tool_search_enabled = True
        self.handle.runtime_tool_names.add(ToolSearchTool.name)
        tool = _convert_module_to_nn_tool(ToolSearchTool())
        if tool.name in self.library and self.tool_configs.get(tool.name):
            raise ValueError(
                f"The runtime tool `{tool.name}` conflicts with an existing tool."
            )
        if (
            tool.name in self.library
            and tool.name not in self.handle.runtime_tool_names
        ):
            raise ValueError(
                f"The runtime tool `{tool.name}` conflicts with an existing tool."
            )
        self.library.update({tool.name: tool})
        self.tool_configs[tool.name] = getattr(tool, "tool_config", {})

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

        if (
            config.get("inject_notification", False)
            and config.get("tool_kind") != "agent"
        ):
            call_params["notification"] = self.handle.build_notification_handle(
                tool_name=tool_name
            )

        if _uses_handle_injection(config):
            call_params["handle"] = self.handle

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

        self._sync_task_store_from_context()
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
            if (
                activity_recorder is not None
                and tool_name not in self._runtime_tool_names
            ):
                activity_recorder.tool_call(tool_name, tool_params)
            call_params = self._build_call_params(
                tool=tool,
                tool_name=tool_name,
                tool_params=tool_params,
                config=config,
                message=message,
                messages=messages,
                vars=vars,
            )

            if config.get("spawn", False):
                return_directly = False
                F.spawn(tool, **call_params)
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
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
                    self.background_dispatcher.dispatch(
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
                    ToolCall(id=tool_id, name=tool_name, parameters=tool_params)
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
                if (
                    isinstance(result, TaskError)
                    and isinstance(result.exception, TaskInterruptRequestedError)
                ):
                    raise result.exception
                parameters = self.handle.build_call_parameters_for_response(
                    meta.params
                )
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

        self._sync_task_store_from_context()
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
            if (
                activity_recorder is not None
                and tool_name not in self._runtime_tool_names
            ):
                activity_recorder.tool_call(tool_name, tool_params)
            call_params = self._build_call_params(
                tool=tool,
                tool_name=tool_name,
                tool_params=tool_params,
                config=config,
                message=message,
                messages=messages,
                vars=vars,
            )

            if config.get("spawn", False):
                return_directly = False
                await F.aspawn(tool, **call_params)
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
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
                    self.background_dispatcher.dispatch(
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
                    ToolCall(id=tool_id, name=tool_name, parameters=tool_params)
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
                if (
                    isinstance(result, TaskError)
                    and isinstance(result.exception, TaskInterruptRequestedError)
                ):
                    raise result.exception
                parameters = self.handle.build_call_parameters_for_response(
                    meta.params
                )
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
