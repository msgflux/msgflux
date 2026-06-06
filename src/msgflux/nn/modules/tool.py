import asyncio
import inspect
import time
from concurrent.futures import CancelledError as FutureCancelledError
from concurrent.futures import TimeoutError as FutureTimeoutError
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import partial
from importlib import import_module
from threading import Lock
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple
from uuid import uuid4

import msgspec

import msgflux.nn.functional as F
from msgflux._private.executor import Executor
from msgflux.auto import AutoParams
from msgflux.chat_messages import ChatMessages
from msgflux.core.dotdict import dotdict
from msgflux.exceptions import (
    TaskError,
    TaskPauseRequestedError,
    TaskStopRequestedError,
)
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
    AgentNotification,
    ToolNotificationHandle,
)
from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_context,
)
from msgflux.tasks import TaskActivityRecorder, TaskHandle, TaskStore
from msgflux.telemetry.span import (
    aset_tool_attributes,
    set_tool_attributes,
)
from msgflux.utils.chat import generate_tool_json_schema
from msgflux.utils.inspect import fn_has_parameters, get_fn_param_defaults
from msgflux.utils.msgspec import restore_transport_value
from msgflux.utils.tenacity import apply_retry, default_tool_retry


def _should_copy_injected_messages(tool: Callable, config: Mapping[str, Any]) -> bool:
    if not config.get("inject_messages", False):
        return False

    agent_type = import_module("msgflux.nn.modules.agent").Agent
    return isinstance(getattr(tool, "impl", tool), agent_type)


@dataclass
class ToolCall:
    """Represents the execution of a single tool call."""

    id: str
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class ToolResponses:
    """Represents the execution of tool calls."""

    return_directly: bool
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> bytes:
        """Returns a encoded-JSON."""
        return msgspec.json.encode(self.to_dict())

    def get_by_id(self, tool_id: str) -> Optional[ToolCall]:
        """Retrieve a tool_call by tool id."""
        return next((r for r in self.tool_calls if r.id == tool_id), None)

    def get_by_name(self, tool_name: str) -> Optional[ToolCall]:
        """Retrieve a tool_call by tool name."""
        return next((r for r in self.tool_calls if r.name == tool_name), None)


class ToolLibraryHandle:
    """Controlled handle exposed to runtime-aware tools."""

    def __init__(self, library: "ToolLibrary"):
        self._library = library

    def add(self, tool: Callable) -> str:
        self._library.add(tool)
        return getattr(tool, "name", None) or getattr(tool, "__name__", None)

    def remove(self, tool_name: str) -> str:
        if tool_name in self._library._runtime_tool_names:
            raise ValueError(f"The runtime tool `{tool_name}` cannot be removed.")
        self._library.remove(tool_name)
        return tool_name

    def list_tools(self) -> List[str]:
        return self._library.get_tool_names()


def _uses_library_injection(config: Mapping[str, Any]) -> bool:
    return config.get("inject_library", False)


def _is_agent_tool_impl(impl: Any) -> bool:
    agent_type = import_module("msgflux.nn.modules.agent").Agent
    return isinstance(impl, agent_type)


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


def _convert_module_to_nn_tool(impl: Callable) -> Tool:  # noqa: C901
    """Convert a callable in nn.Tool."""
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

    tool_config["tool_kind"] = "agent" if _is_agent_tool_impl(impl) else "tool"

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
        if _uses_library_injection(tool_config):
            annotations.pop("tool_library", None)

    if tool_config.get("spawn"):
        doc = "This tool will not generate a return. \n" + doc
    if tool_config.get("background"):
        doc = "This tool runs in the background and returns a task id. \n" + doc

    return LocalTool(
        name=name,
        description=doc,
        annotations=annotations,
        tool_config=tool_config,
        impl=impl,
        display_name=display_name or name,
        usage_guidance=usage_guidance,
    )


class ToolLibrary(Module, metaclass=AutoParams):
    """ToolLibrary is a Module type that manage tool calls over the tool library."""

    def __init__(
        self,
        name: str,
        tools: List[Callable],
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
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
        self.register_buffer("mcp_clients", {})
        self.task_store = TaskStore()
        self.agent_inbox = AgentInbox(owner=f"{name}_tool_library")
        self._runtime_tool_names = {
            "task_status",
            "task_list",
            "task_output",
            "task_wait",
            "task_stop",
        }
        self._task_runtime_enabled = False
        self._agent_task_runtime_enabled = False
        self._on_demand_runtime_enabled = False
        self._loaded_on_demand_tool_names: set[str] = set()
        self._task_futures: Dict[str, Any] = {}
        self._task_futures_lock = Lock()
        self._task_inboxes: Dict[str, AgentInbox] = {}
        self._task_inboxes_lock = Lock()
        for tool in tools:
            self.add(tool)
        if mcp_servers:
            self._initialize_mcp_clients(mcp_servers)

    def add(self, tool: Callable):
        """Add a local tool in library."""
        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if name in self.library.keys():
            raise ValueError(f"The tool name `{name}` is already in tool library")
        if not isinstance(tool, Tool):
            tool = _convert_module_to_nn_tool(tool)

        # Store tool config (may be empty dict for local tools)
        self.tool_configs[tool.name] = getattr(tool, "tool_config", {})
        self.library.update({tool.name: tool})
        self._apply_tool_registration_effects(tool.name)

    def remove(self, tool_name: str):
        if tool_name in self.library.keys():
            self.library.pop(tool_name)
            self.tool_configs.pop(tool_name, None)
            self._loaded_on_demand_tool_names.discard(tool_name)
            self._sync_on_demand_runtime_tools()
        else:
            raise ValueError(f"The tool name `{tool_name}` is not in tool library")

    def clear(self):
        self.library.clear()
        self.tool_configs.clear()
        for mcp_data in self.mcp_clients.values():
            F.wait_for(mcp_data["client"].disconnect)
        self.mcp_clients.clear()
        self._task_runtime_enabled = False
        self._agent_task_runtime_enabled = False
        self._on_demand_runtime_enabled = False
        self._loaded_on_demand_tool_names.clear()
        self._runtime_tool_names = {
            "task_status",
            "task_list",
            "task_output",
            "task_wait",
            "task_stop",
        }
        with self._task_futures_lock:
            self._task_futures.clear()

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
        return list(self.library.keys())

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
        self.agent_inbox = agent_inbox

    # --- Tool Visibility Helpers ---

    def _apply_tool_registration_effects(self, tool_name: str) -> None:
        config = self.tool_configs.get(tool_name, {})
        if config.get("background", False):
            self._ensure_task_runtime_tools()
        if config.get("background", False) and config.get("tool_kind") == "agent":
            self._ensure_agent_task_runtime_tools()
        if config.get("on_demand", False):
            self._ensure_on_demand_runtime_tools()

    def _get_on_demand_tool_names(self) -> List[str]:
        return [
            tool_name
            for tool_name, config in self.tool_configs.items()
            if config.get("on_demand", False)
        ]

    def _is_tool_exposed(self, tool_name: str) -> bool:
        if tool_name in self._runtime_tool_names:
            return True
        config = self.tool_configs.get(tool_name, {})
        if not config.get("on_demand", False):
            return True
        return tool_name in self._loaded_on_demand_tool_names

    def _load_on_demand_tools(self, tool_names: List[str]) -> List[str]:
        newly_loaded = []
        for tool_name in tool_names:
            if tool_name not in self._loaded_on_demand_tool_names:
                self._loaded_on_demand_tool_names.add(tool_name)
                newly_loaded.append(tool_name)
        return newly_loaded

    def _sync_on_demand_runtime_tools(self) -> None:
        if self._get_on_demand_tool_names():
            self._ensure_on_demand_runtime_tools()
            return
        if not self._on_demand_runtime_enabled:
            return
        self._on_demand_runtime_enabled = False
        self._runtime_tool_names.discard("tool_search")
        if "tool_search" in self.library:
            self.library.pop("tool_search")
        self.tool_configs.pop("tool_search", None)

    # --- Task Runtime Registration ---

    def _ensure_task_runtime_tools(self) -> None:
        if self._task_runtime_enabled:
            return
        self._task_runtime_enabled = True
        self._register_runtime_tool(
            name="task_status",
            description="Get the current status of a background task by task_id.",
            annotations={"task_id": str},
            impl=self._task_status,
        )
        self._register_runtime_tool(
            name="task_list",
            description="List background tasks registered in the current tool library.",
            annotations={"status": Optional[str]},
            impl=self._task_list,
        )
        self._register_runtime_tool(
            name="task_output",
            description="Get the final output of a background task by task_id.",
            annotations={"task_id": str},
            impl=self._task_output,
        )
        self._register_runtime_tool(
            name="task_wait",
            description=(
                "Wait for a background task to finish. "
                "Returns the final output, failed payload, or a timeout status."
            ),
            annotations={"task_id": str, "timeout": Optional[float]},
            impl=self._task_wait,
        )
        self._register_runtime_tool(
            name="task_stop",
            description=(
                "Request a cooperative stop for a background task. "
                "Stops immediately only if the task has not started yet."
            ),
            annotations={"task_id": str},
            impl=self._task_stop,
        )

    def _ensure_agent_task_runtime_tools(self) -> None:
        if self._agent_task_runtime_enabled:
            return
        self._agent_task_runtime_enabled = True
        self._runtime_tool_names.add("task_activity")
        self._register_runtime_tool(
            name="task_activity",
            description="List compact activity entries for a background agent task.",
            annotations={"task_id": str, "limit": Optional[int]},
            impl=self._task_activity,
        )
        self._runtime_tool_names.add("task_message")
        self._register_runtime_tool(
            name="task_message",
            description=(
                "Send a message to a background agent task. "
                "If it is still running, deliver the message to its inbox. "
                "If it already stopped, resume the task from its checkpoint."
            ),
            annotations={"task_id": str, "message": str},
            impl=self._task_message,
        )

    def _ensure_on_demand_runtime_tools(self) -> None:
        if self._on_demand_runtime_enabled:
            return
        self._on_demand_runtime_enabled = True
        self._runtime_tool_names.add("tool_search")
        self._register_runtime_tool(
            name="tool_search",
            description=(
                "Search registered on-demand tools by keyword or exact "
                "selection. Matching tools become available in the next model "
                "call. Use `select:tool_a,tool_b` for direct selection."
            ),
            annotations={"query": str, "max_results": Optional[int]},
            impl=self._tool_search,
        )

    def _register_runtime_tool(
        self,
        *,
        name: str,
        description: str,
        annotations: Dict[str, Any],
        impl: Callable,
    ) -> None:
        tool = LocalTool(
            name=name,
            description=description,
            annotations=annotations,
            tool_config={},
            impl=impl,
        )
        if name in self.library and self.tool_configs.get(name):
            raise ValueError(
                f"The runtime tool `{name}` conflicts with an existing tool."
            )
        if name in self.library and name not in self._runtime_tool_names:
            raise ValueError(
                f"The runtime tool `{name}` conflicts with an existing tool."
            )
        self.library.update({name: tool})
        self.tool_configs[name] = {}

    # --- Task Runtime Tools ---

    def _task_status(self, task_id: str) -> Dict[str, Any]:
        task = self.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        payload = task.to_dict()
        payload.update(self._build_task_timing_fields(task))
        last_activity = self.task_store.get_last_activity(task_id)
        if last_activity is not None:
            payload["last_activity_summary"] = self._format_task_activity_entry(
                last_activity
            )
        return payload

    def _task_list(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        tasks = []
        for task in self.task_store.list(status=status):
            payload = task.to_dict()
            payload.update(self._build_task_timing_fields(task))
            last_activity = self.task_store.get_last_activity(task.task_id)
            if last_activity is not None:
                payload["last_activity_summary"] = self._format_task_activity_entry(
                    last_activity
                )
            tasks.append(payload)
        return tasks

    def _task_output(self, task_id: str) -> Any:
        task = self.task_store.get(task_id)
        return self._build_task_result(task_id=task_id, task=task)

    def _tool_search(
        self,
        query: str,
        max_results: Optional[int] = 5,
    ) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("`query` must be a non-empty string.")
        if max_results is not None:
            if isinstance(max_results, bool) or not isinstance(max_results, int):
                raise TypeError(
                    f"`max_results` must be int or None, given `{type(max_results)}`"
                )
            if max_results <= 0:
                raise ValueError("`max_results` must be greater than 0.")

        on_demand_tool_names = self._get_on_demand_tool_names()
        total = len(on_demand_tool_names)
        if total == 0:
            return {
                "query": query,
                "matches": [],
                "loaded": [],
                "already_loaded": [],
                "total_on_demand_tools": 0,
            }

        if query.lower().startswith("select:"):
            requested = [
                item.strip()
                for item in query.split(":", 1)[1].split(",")
                if item.strip()
            ]
            matches = self._select_on_demand_tools(requested)
        else:
            matches = self._search_on_demand_tools(
                query=query,
                max_results=max_results or 5,
            )

        newly_loaded = self._load_on_demand_tools(matches)
        already_loaded = [
            tool_name for tool_name in matches if tool_name not in newly_loaded
        ]
        return {
            "query": query,
            "matches": matches,
            "loaded": newly_loaded,
            "already_loaded": already_loaded,
            "total_on_demand_tools": total,
        }

    def _task_activity(
        self,
        task_id: str,
        limit: Optional[int] = 10,
    ) -> Any:
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError(f"`limit` must be int or None, given `{type(limit)}`")
            if limit <= 0:
                raise ValueError("`limit` must be greater than 0.")
        task = self.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.metadata.get("task_kind") != "agent":
            return {
                "task_id": task_id,
                "status": "unsupported",
                "error": "task_activity is only available for background agent tasks.",
            }
        activity = self.task_store.list_activity(task_id, limit=limit)
        return [self._format_task_activity_entry(item) for item in activity]

    def _task_wait(self, task_id: str, timeout: Optional[float] = None) -> Any:  # noqa: C901
        if timeout is not None:
            if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
                raise TypeError(
                    f"`timeout` must be float, int or None, given `{type(timeout)}`"
                )
            if timeout < 0:
                raise ValueError("`timeout` must be greater than or equal to 0.")

        task = self.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.status in {"completed", "failed", "stopped"}:
            return self._build_task_result(task_id=task_id, task=task)

        future = self._get_task_future(task_id)
        if future is not None:
            try:
                future.result(timeout=timeout)
            except FutureTimeoutError:
                task = self.task_store.get(task_id)
                return self._build_task_timeout_result(task_id=task_id, task=task)
            except Exception:
                task = self.task_store.get(task_id)
                return self._build_task_result(task_id=task_id, task=task)
            task = self.task_store.get(task_id)
            return self._build_task_result(task_id=task_id, task=task)

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            task = self.task_store.get(task_id)
            if task is None or task.status in {"completed", "failed", "stopped"}:
                return self._build_task_result(task_id=task_id, task=task)
            if deadline is not None and time.monotonic() >= deadline:
                return self._build_task_timeout_result(task_id=task_id, task=task)
            time.sleep(0.05)

    def _task_stop(self, task_id: str) -> Dict[str, Any]:
        task = self.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}

        if task.status in {"completed", "failed", "stopped"}:
            return {
                "task_id": task_id,
                "status": task.status,
                "message": "Task already reached a terminal state.",
            }

        self.task_store.request_stop(task_id)
        future = self._get_task_future(task_id)
        if future is not None and future.cancel():
            stopped = self.task_store.stop(
                task_id,
                reason="Task was cancelled before it started running.",
            )
            return {
                "task_id": task_id,
                "status": "stopped",
                "message": "Task stopped before execution started.",
                "task_status": stopped.status if stopped is not None else "stopped",
            }

        return {
            "task_id": task_id,
            "status": "stop_requested",
            "message": (
                "Stop requested. The task will stop at the next cooperative checkpoint."
            ),
        }

    def _task_message(self, task_id: str, message: str) -> Dict[str, Any]:
        task = self.task_store.get(task_id)
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.metadata.get("task_kind") != "agent":
            return {
                "task_id": task_id,
                "status": "unsupported",
                "error": "task_message is only available for background agent tasks.",
            }
        if not isinstance(message, str) or not message.strip():
            raise ValueError("`message` must be a non-empty string.")

        task_inbox = self._get_task_inbox(task_id)
        if task.status == "running" and task_inbox is not None:
            task_inbox.publish(
                {
                    "source": "task_message",
                    "ref": task_id,
                    "status": "message",
                    "hint": message.strip(),
                    "metadata": {"direction": "root_to_task"},
                }
            )
            self.task_store.add_activity(
                task_id,
                kind="message",
                summary=f"Root message: {self._truncate_activity_text(message)}",
                metadata={"direction": "root_to_task"},
            )
            return {
                "task_id": task_id,
                "status": "delivered",
                "message": "Message delivered to the running background agent.",
            }

        resumed = self._resume_background_agent_task(task=task, message=message.strip())
        return {
            "task_id": task_id,
            "status": "resumed",
            "message": resumed,
        }

    # --- Task Runtime Helpers ---

    def _build_task_result(self, *, task_id: str, task: Optional[Any]) -> Any:
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        if task.status == "completed":
            return task.result
        if task.status == "failed":
            return {"task_id": task_id, "status": task.status, "error": task.error}
        if task.status == "stopped":
            return {
                "task_id": task_id,
                "status": task.status,
                "reason": task.metadata.get("stop_reason"),
            }
        return {
            "task_id": task_id,
            "status": task.status,
            "progress": task.progress.to_dict(),
        }

    def _build_task_timeout_result(
        self, *, task_id: str, task: Optional[Any]
    ) -> Dict[str, Any]:
        if task is None:
            return {"task_id": task_id, "status": "not_found"}
        payload = {
            "task_id": task_id,
            "status": "timeout",
            "task_status": task.status,
        }
        if task.status not in {"completed", "failed"}:
            if task.status == "stopped":
                payload["reason"] = task.metadata.get("stop_reason")
                return payload
            payload["progress"] = task.progress.to_dict()
        elif task.status == "failed":
            payload["error"] = task.error
        return payload

    def _build_task_timing_fields(self, task: Any) -> Dict[str, Any]:
        started_at = task.created_at
        now = time.time()
        created_ts = self._parse_utc_timestamp(task.created_at)
        completed_ts = self._parse_utc_timestamp(task.completed_at)
        payload: Dict[str, Any] = {"started_at": started_at}
        if created_ts is None:
            return payload
        if completed_ts is not None:
            payload["elapsed_seconds"] = round(completed_ts - created_ts, 3)
        else:
            payload["running_for_seconds"] = round(now - created_ts, 3)
        return payload

    def _format_task_activity_entry(self, activity: Any) -> str:
        label_map = {
            "status": "Status",
            "progress": "Progress",
            "tool_call": "ToolCall",
            "error": "Error",
            "message": "Message",
        }
        label = label_map.get(activity.kind, activity.kind.title())
        return f"{label}: {activity.summary}"

    def _select_on_demand_tools(self, requested: List[str]) -> List[str]:
        resolved = []
        normalized = {
            tool_name.lower(): tool_name
            for tool_name in self._get_on_demand_tool_names()
        }
        for tool_name in requested:
            match = normalized.get(tool_name.lower())
            if match is not None and match not in resolved:
                resolved.append(match)
        return resolved

    def _search_on_demand_tools(self, *, query: str, max_results: int) -> List[str]:
        query_lower = query.strip().lower()
        terms = [term for term in query_lower.split() if term]
        if not terms:
            return []

        matches = []
        for tool_name in self._get_on_demand_tool_names():
            if tool_name not in self.library:
                continue
            tool = self.library[tool_name]
            name_parts = tool_name.lower().replace("__", " ").replace("_", " ")
            description = (tool.get_module_description() or "").lower()
            score = 0
            if query_lower == tool_name.lower():
                score += 100
            if query_lower in name_parts:
                score += 40
            for term in terms:
                if term in name_parts:
                    score += 15
                if description and term in description:
                    score += 5
            if score > 0:
                matches.append((score, tool_name))

        matches.sort(key=lambda item: (-item[0], item[1]))
        return [tool_name for _, tool_name in matches[:max_results]]

    def _build_background_dispatch_result(
        self,
        *,
        task_id: str,
        tool_name: str,
        task_kind: str,
    ) -> str:
        actions = ["`task_status`", "`task_stop`", "`task_wait`", "`task_output`"]
        if task_kind == "agent":
            actions.insert(1, "`task_activity`")
            actions.insert(2, "`task_message`")
        return (
            f"The `{tool_name}` tool is running in the background with "
            f"task_id='{task_id}'. Use that task_id with "
            + ", ".join(actions[:-1])
            + f", or {actions[-1]}."
        )

    @staticmethod
    def _parse_utc_timestamp(value: Optional[str]) -> Optional[float]:
        if not isinstance(value, str) or not value:
            return None
        try:
            normalized = value.replace("Z", "+00:00")
            return (
                datetime.fromisoformat(normalized).astimezone(timezone.utc).timestamp()
            )
        except ValueError:
            return None

    @staticmethod
    def _truncate_activity_text(value: str, *, limit: int = 140) -> str:
        text = " ".join(str(value).split())
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _register_task_future(self, task_id: str, future: Any) -> None:
        with self._task_futures_lock:
            self._task_futures[task_id] = future

    def _get_task_future(self, task_id: str) -> Optional[Any]:
        with self._task_futures_lock:
            return self._task_futures.get(task_id)

    def _cleanup_task_future(self, task_id: str, future: Any) -> None:
        with self._task_futures_lock:
            current = self._task_futures.get(task_id)
            if current is future:
                self._task_futures.pop(task_id, None)

    def _register_task_inbox(self, task_id: str, inbox: AgentInbox) -> None:
        with self._task_inboxes_lock:
            self._task_inboxes[task_id] = inbox

    def _get_task_inbox(self, task_id: str) -> Optional[AgentInbox]:
        with self._task_inboxes_lock:
            return self._task_inboxes.get(task_id)

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
            call_params["notification"] = self._build_notification_handle(
                tool_name=tool_name
            )

        if _uses_library_injection(config):
            call_params["tool_library"] = ToolLibraryHandle(self)

        return call_params

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

    def _build_call_parameters_for_response(
        self, params: Optional[Mapping[str, Any]]
    ) -> Optional[Dict[str, Any]]:
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
            "tool_library",
            "tool_call_id",
        ):
            parameters.pop(key, None)
        return parameters

    def _build_notification_handle(
        self,
        *,
        tool_name: str,
        ref: Optional[str] = None,
        agent_inbox: Optional[AgentInbox] = None,
    ) -> ToolNotificationHandle:
        execution_context = get_execution_context()
        inbox = agent_inbox or execution_context.get("agent_inbox") or self.agent_inbox
        metadata = {"tool": tool_name}
        return ToolNotificationHandle(
            inbox,
            ref=ref,
            metadata=metadata,
        )

    # --- Background Task Execution ---

    def _run_background_tool(
        self,
        *,
        tool: Tool,
        task_handle: TaskHandle,
        tool_name: str,
        call_params: Dict[str, Any],
        execution_scope: Optional[Dict[str, Any]] = None,
        agent_inbox: Optional[AgentInbox] = None,
    ) -> Any:
        scope = execution_scope or {}
        with execution_context(**scope):
            task_handle.set_running()
            try:
                result = tool(**call_params)
            except TaskStopRequestedError as exc:
                task_handle.stop(reason=str(exc))
                self._publish_task_notification(
                    task_id=task_handle.task_id,
                    tool_name=tool_name,
                    status="stopped",
                    hint=(
                        f"Use task_status(task_id='{task_handle.task_id}') "
                        "if you need stop details."
                    ),
                    agent_inbox=agent_inbox,
                )
                raise
            except TaskPauseRequestedError as exc:
                task_handle.pause(reason=str(exc))
                self._publish_task_notification(
                    task_id=task_handle.task_id,
                    tool_name=tool_name,
                    status="paused",
                    hint=(
                        f"Use task_message(task_id='{task_handle.task_id}', "
                        "message='...') to resume the paused task."
                    ),
                    agent_inbox=agent_inbox,
                )
                raise
            except Exception as exc:
                task_handle.fail(exc)
                self._publish_task_notification(
                    task_id=task_handle.task_id,
                    tool_name=tool_name,
                    status="failed",
                    hint=(
                        f"Use task_status(task_id='{task_handle.task_id}') "
                        "if you need error details."
                    ),
                    agent_inbox=agent_inbox,
                )
                raise
            task_handle.complete(result)
            self._publish_task_notification(
                task_id=task_handle.task_id,
                tool_name=tool_name,
                status="completed",
                hint=(
                    f"Use task_output(task_id='{task_handle.task_id}') "
                    "if you need the result."
                ),
                agent_inbox=agent_inbox,
            )
            return result

    def _resume_background_agent_task(self, *, task: Any, message: str) -> str:
        tool_name = task.tool_name
        if tool_name not in self.library:
            raise ValueError(f"The tool `{tool_name}` is no longer available.")
        tool = self.library[tool_name]
        checkpoint_namespace = (
            tool.impl.get_module_name()
            if hasattr(tool, "impl") and hasattr(tool.impl, "get_module_name")
            else tool.get_module_name()
        )

        checkpoint_store = get_execution_context().get("checkpoint_store")
        session_id = task.metadata.get("checkpoint_session_id") or task.metadata.get(
            "session_id"
        )
        run_id = task.metadata.get("checkpoint_run_id") or task.task_id
        restored_messages = ChatMessages()
        restored_vars: Dict[str, Any] = {}
        restored_model_preference = None

        if checkpoint_store is not None and isinstance(session_id, str) and session_id:
            state = checkpoint_store.load_state(
                checkpoint_namespace,
                session_id,
                run_id,
            )
            if state is not None:
                restored_messages._hydrate_state(state.get("messages", {}))
                restored_vars = state.get("vars", {}) or {}
                restored_model_preference = state.get("model_preference")

        root_inbox = get_execution_context().get("agent_inbox") or self.agent_inbox
        task_inbox = self._get_task_inbox(task.task_id)
        if task_inbox is None:
            task_inbox = root_inbox.fork(
                owner=f"{tool_name}:{task.task_id}",
                namespace=checkpoint_namespace,
                session_id=(
                    session_id if isinstance(session_id, str) and session_id else None
                ),
                run_id=run_id,
            )
            self._register_task_inbox(task.task_id, task_inbox)

        self.task_store.requeue(task.task_id)
        self.task_store.add_activity(
            task.task_id,
            kind="message",
            summary=f"Root message: {self._truncate_activity_text(message)}",
            metadata={"direction": "root_to_task", "resume": True},
        )

        execution_scope = {
            "session_id": session_id
            if isinstance(session_id, str) and session_id
            else None,
            "run_id": run_id,
            "parent_run_id": task.metadata.get("parent_run_id"),
            "root_run_id": task.metadata.get("root_run_id"),
            "checkpoint_store": checkpoint_store,
            "agent_inbox": task_inbox,
            "task_activity_recorder": TaskActivityRecorder(
                task.task_id, self.task_store
            ),
        }
        future = Executor.get_instance().submit(
            partial(
                self._run_background_tool,
                tool=tool,
                task_handle=TaskHandle(
                    task.task_id,
                    self.task_store,
                    tool_name=tool_name,
                    agent_inbox=root_inbox,
                ),
                tool_name=tool_name,
                call_params={
                    "messages": restored_messages,
                    "scope": ExecutionScope(
                        session_id=(
                            session_id if isinstance(session_id, str) else "default"
                        ),
                        namespace=checkpoint_namespace,
                        run_id=run_id,
                        parent_run_id=task.metadata.get("parent_run_id"),
                        root_run_id=task.metadata.get("root_run_id"),
                    ),
                    "model_preference": restored_model_preference,
                    "vars": restored_vars,
                    "tool_call_id": f"task_message_{task.task_id}",
                    "task": message,
                },
                execution_scope=execution_scope,
                agent_inbox=root_inbox,
            )
        )
        self._register_task_future(task.task_id, future)
        future.add_done_callback(partial(self._cleanup_task_future, task.task_id))
        future.add_done_callback(self._log_background_task_failure)
        return "Message scheduled and background agent resumed."

    def _log_background_task_failure(self, future: Any) -> None:
        try:
            future.result()
        except FutureCancelledError:
            return
        except TaskStopRequestedError:
            return
        except TaskPauseRequestedError:
            return
        except Exception as exc:
            logger.error(f"Background task error: {exc!s}", exc_info=True)

    # --- Background Task Dispatch ---

    def _dispatch_background_tool(
        self,
        *,
        tool: Tool,
        tool_id: str,
        tool_name: str,
        call_params: Dict[str, Any],
        config: Mapping[str, Any],
    ) -> ToolCall:
        task_kind = config.get("tool_kind", "tool")
        execution_context = get_execution_context()
        session_id = execution_context.get("session_id")
        parent_run_id = execution_context.get("run_id")
        root_run_id = execution_context.get("root_run_id")
        checkpoint_store = execution_context.get("checkpoint_store")
        root_agent_inbox = execution_context.get("agent_inbox") or self.agent_inbox
        task_id = uuid4().hex[:8]
        task_inbox = None
        if task_kind == "agent":
            task_inbox = root_agent_inbox.fork(
                owner=f"{tool_name}:{task_id}",
                namespace=tool_name,
                session_id=session_id if isinstance(session_id, str) else None,
                run_id=task_id,
            )
            self._register_task_inbox(task_id, task_inbox)
        task = self.task_store.create(
            task_id=task_id,
            tool_name=tool_name,
            metadata={
                "tool_call_id": tool_id,
                "task_kind": task_kind,
                "session_id": session_id,
                "parent_run_id": parent_run_id,
                "root_run_id": root_run_id,
                "checkpoint_session_id": session_id,
                "checkpoint_run_id": task_id if task_kind == "agent" else None,
                "supports_activity": task_kind == "agent",
                "supports_message": task_kind == "agent",
                "stop_requested": False,
            },
        )
        runner_params = dict(call_params)
        if config.get("inject_task", False) and task_kind != "agent":
            runner_params["task"] = TaskHandle(
                task.task_id,
                self.task_store,
                tool_name=tool_name,
                agent_inbox=root_agent_inbox,
            )
        if config.get("inject_notification", False) and task_kind != "agent":
            runner_params["notification"] = self._build_notification_handle(
                tool_name=tool_name,
                ref=task.task_id,
                agent_inbox=root_agent_inbox,
            )
        if task_kind == "agent":
            runner_params["scope"] = ExecutionScope(
                session_id=session_id if isinstance(session_id, str) else "default",
                namespace=tool_name,
                run_id=task.task_id,
                parent_run_id=(
                    parent_run_id
                    if isinstance(parent_run_id, str) and parent_run_id
                    else None
                ),
                root_run_id=(
                    root_run_id
                    if isinstance(root_run_id, str) and root_run_id
                    else task.task_id
                ),
            )
        runner_params["tool_call_id"] = tool_id
        execution_scope = {
            "session_id": session_id
            if isinstance(session_id, str) and session_id
            else None,
            "run_id": task.task_id,
            "parent_run_id": (
                parent_run_id
                if isinstance(parent_run_id, str) and parent_run_id
                else None
            ),
            "root_run_id": (
                root_run_id if isinstance(root_run_id, str) and root_run_id else None
            ),
            "checkpoint_store": checkpoint_store,
            "agent_inbox": task_inbox or root_agent_inbox,
            "task_handle": TaskHandle(
                task.task_id,
                self.task_store,
                tool_name=tool_name,
                agent_inbox=root_agent_inbox,
            ),
            "task_activity_recorder": TaskActivityRecorder(
                task.task_id, self.task_store
            ),
        }
        future = Executor.get_instance().submit(
            partial(
                self._run_background_tool,
                tool=tool,
                task_handle=TaskHandle(
                    task.task_id,
                    self.task_store,
                    tool_name=tool_name,
                    agent_inbox=root_agent_inbox,
                ),
                tool_name=tool_name,
                call_params=runner_params,
                execution_scope=execution_scope,
                agent_inbox=root_agent_inbox,
            )
        )
        self._register_task_future(task.task_id, future)
        future.add_done_callback(partial(self._cleanup_task_future, task.task_id))
        future.add_done_callback(self._log_background_task_failure)
        return ToolCall(
            id=tool_id,
            name=tool_name,
            parameters=self._build_call_parameters_for_response(call_params),
            result=self._build_background_dispatch_result(
                task_id=task.task_id,
                tool_name=tool_name,
                task_kind=task_kind,
            ),
        )

    # --- Background Task Notifications ---

    def _publish_task_notification(
        self,
        *,
        task_id: str,
        tool_name: str,
        status: str,
        hint: str,
        agent_inbox: Optional[AgentInbox] = None,
    ) -> Optional[AgentNotification]:
        inbox = agent_inbox or self.agent_inbox
        if inbox is None:
            return None
        return inbox.publish(
            AgentNotification(
                notification_id=uuid4().hex[:8],
                source="task",
                ref=task_id,
                status=status,
                hint=hint,
                metadata={"tool": tool_name},
                dedupe_key=f"task:{task_id}:{status}",
            )
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

            if config.get("background", False):
                return_directly = False
                tool_calls.append(
                    self._dispatch_background_tool(
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
                parameters = self._build_call_parameters_for_response(meta.params)
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

            if config.get("background", False):
                return_directly = False
                tool_calls.append(
                    self._dispatch_background_tool(
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
                parameters = self._build_call_parameters_for_response(meta.params)
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
