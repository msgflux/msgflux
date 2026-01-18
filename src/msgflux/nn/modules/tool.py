import asyncio
import inspect
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union

import msgspec

from msgflux.dotdict import dotdict
from msgflux.logger import logger
from msgflux.nn import functional as F
from msgflux.nn.modules.container import ModuleDict
from msgflux.nn.modules.module import Module
from msgflux.protocols.mcp import (
    MCPClient,
    convert_mcp_schema_to_tool_schema,
    extract_tool_result_text,
    filter_tools,
)
from msgflux.telemetry.span import (
    aset_tool_attributes,
    set_tool_attributes,
)
from msgflux.utils.chat import generate_tool_json_schema
from msgflux.utils.convert import convert_camel_to_snake_case
from msgflux.utils.inspect import fn_has_parameters
from msgflux.utils.tenacity import tool_retry


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

        # Store MCP-specific data
        self._mcp_client = mcp_client
        self._mcp_tool_info = mcp_tool_info
        self._namespace = namespace
        self._mcp_tool_name = name

        # Set description from MCP tool info
        if hasattr(mcp_tool_info, "description"):
            self.set_description(mcp_tool_info.description)

        # Store config
        self.register_buffer("tool_config", config or {})

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
    ):
        super().__init__()
        self.set_name(name)
        self.set_description(description)
        self.set_annotations(annotations)
        self.register_buffer("tool_config", tool_config)
        self.impl = impl  # Not a buffer for now

    @tool_retry
    @set_tool_attributes(execution_type="local")
    def forward(self, **kwargs):
        if inspect.iscoroutinefunction(self.impl):
            return F.wait_for(self.impl, **kwargs)
        return self.impl(**kwargs)

    @tool_retry
    @aset_tool_attributes(execution_type="local")
    async def aforward(self, *args, **kwargs):
        if hasattr(self.impl, "acall"):
            return await self.impl.acall(*args, **kwargs)
        elif inspect.iscoroutinefunction(self.impl):
            return await self.impl(*args, **kwargs)
        # Fall back to sync call in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.impl(*args, **kwargs))


def _convert_module_to_nn_tool(impl: Callable) -> Tool:  # noqa: C901
    """Convert a callable in nn.Tool."""
    tool_config = impl.__dict__.get("tool_config", dotdict())

    name_overridden = tool_config.pop("name_overridden", None)

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

        raw_name = (
            name_overridden
            or getattr(impl, "name", None)
            or getattr(impl, "__name__", None)
        )
        name = convert_camel_to_snake_case(raw_name)

        if inspect.isclass(impl):
            impl = impl()  # Initialized

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

    else:
        raise ValueError(
            "The given object is not a callable function, class, or instance"
        )

    if tool_config.get("handoff", False):
        name = "transfer_to_" + name
        annotations = {}  # pass only the model state

    if tool_config.get("background"):
        doc = "This tool will run in the background. \n" + doc

    return LocalTool(
        name=name,
        description=doc,
        annotations=annotations,
        tool_config=tool_config,
        impl=impl,
    )


class ToolLibrary(Module):
    """ToolLibrary is a Module type that manage tool calls over the tool library."""

    def __init__(
        self,
        name: str,
        tools: List[Callable],
        special_tools: Optional[List[str]] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
    ):
        """Args:
        name:
            Library name.
        tools:
            A list of callables.
        special_tools:
            Autonomy tools for the model.
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
        self.register_buffer("special_library", [])
        self.register_buffer("tool_configs", {})
        self.register_buffer("mcp_clients", {})
        for tool in tools:
            self.add(tool)
        if special_tools:
            for special_tool in special_tools:
                self.special_add(special_tool)
        if mcp_servers:
            self._initialize_mcp_clients(mcp_servers)

    def add(self, tool: Union[str, Callable]):
        """Add a local tool in library."""
        if isinstance(tool, str):
            if tool in self.special_library.keys():
                raise ValueError(
                    f"The special tool name `{tool}` is already in special tool library"
                )
            self.special_library.append(tool)
        else:
            name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
            if name in self.library.keys():
                raise ValueError(f"The tool name `{name}` is already in tool library")
            if not isinstance(tool, Tool):
                tool = _convert_module_to_nn_tool(tool)

            # Store tool config (may be empty dict for local tools)
            self.tool_configs[tool.name] = getattr(tool, "tool_config", {})

            self.library.update({tool.name: tool})

    def remove(self, tool_name: str):
        if tool_name in self.library.keys():
            self.library.pop(tool_name)
            self.tool_configs.pop(tool_name, None)
        elif tool_name in self.special_library:
            self.special_library.remove(tool_name)
        else:
            raise ValueError(f"The tool name `{tool_name}` is not in tool library")

    def clear(self):
        self.library.clear()
        self.special_library.clear()
        # TODO: clean mcp

    def _initialize_mcp_clients(self, mcp_servers: List[Dict[str, Any]]):
        """Initialize MCP clients from server configurations."""
        for server_config in mcp_servers:
            namespace = server_config.get("name")
            if not namespace:
                raise ValueError("MCP server config must include 'name' field")

            transport_type = server_config.get("transport", "stdio")

            # Create client based on transport type
            if transport_type == "stdio":
                client = MCPClient.from_stdio(
                    command=server_config.get("command"),
                    args=server_config.get("args"),
                    cwd=server_config.get("cwd"),
                    env=server_config.get("env"),
                    timeout=server_config.get("timeout", 30.0),
                )
            elif transport_type == "http":
                client = MCPClient.from_http(
                    base_url=server_config.get("base_url"),
                    timeout=server_config.get("timeout", 30.0),
                    headers=server_config.get("headers"),
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

        # Local tools
        for tool_name in self.library:
            schemas.append(self.library[tool_name].get_json_schema())

        # MCP tools
        if self.mcp_clients:
            for namespace, mcp_data in self.mcp_clients.items():
                for mcp_tool in mcp_data["tools"]:
                    schema = convert_mcp_schema_to_tool_schema(mcp_tool, namespace)
                    schemas.append(schema)

        return schemas

    def forward(  # noqa: C901
        self,
        tool_callings: List[Tuple[str, str, Any]],
        model_state: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> ToolResponses:
        """Executes tool calls with tool config logic.

        Args:
            tool_callings:
                A list of tuples containing the tool id, name and parameters.
                !!! example
                    [('123121', 'tool_name1', {'parameter1': 'value1'}),
                    ('322', 'tool_name2', '')]
            model_state:
                The current state of the Agent for the `handoff` functionality.
            vars:
                Extra kwargs to be used in tools.

        Returns:
            ToolResponses:
                Structured object containing all tool call results.
        """
        # TODO capturar no trace quando o modelo erra algo na tool
        # TODO capturar no trace o tool config
        if model_state is None:
            model_state = {}

        if vars is None:
            vars = {}

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

            # Handle inject_vars
            inject_vars = config.get("inject_vars", False)
            if inject_vars:
                if isinstance(inject_vars, list):
                    for key in inject_vars:
                        if key not in vars:
                            raise ValueError(
                                f"The tool `{tool_name}` requires the injected "
                                f"parameter `{key}`, but it was not found."
                            )
                        tool_params[key] = vars[key]
                elif inject_vars is True:
                    tool_params["vars"] = vars

            if config.get("background", False):
                return_directly = False
                F.background_task(tool, **(tool_params or {}))
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
                        result=f"""The `{tool_name}` tool was started in the background.
                        This tool will not generate a return""",
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

            if config.get("inject_model_state", False):  # Add model_state
                tool_params["task_messages"] = model_state

            if not config.get("return_direct", False):
                return_directly = False

            tool_params = tool_params or {}
            # Add tool_call_id for telemetry
            tool_params["tool_call_id"] = tool_id
            prepared_calls.append(partial(tool, **tool_params))

            call_metadata.append(
                dotdict(id=tool_id, name=tool_name, config=config, params=tool_params)
            )

        if prepared_calls:
            results = F.scatter_gather(prepared_calls)
            for meta, result in zip(call_metadata, results):
                if isinstance(meta.params, dict):
                    parameters = meta.params.to_dict()
                    parameters.pop("vars", None)
                    parameters.pop("tool_call_id", None)
                else:
                    parameters = None
                tool_calls.append(
                    ToolCall(
                        id=meta.id,
                        name=meta.name,
                        parameters=parameters,
                        result=result,
                    )
                )

        return ToolResponses(return_directly=return_directly, tool_calls=tool_calls)

    async def aforward(  # noqa: C901
        self,
        tool_callings: List[Tuple[str, str, Any]],
        model_state: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> ToolResponses:
        """Async version of forward. Executes tool calls with logic for `handoff`, `return_direct`.

        Args:
            tool_callings:
                A list of tuples containing the tool id, name and parameters.
                !!! example
                    [('123121', 'tool_name1', {'parameter1': 'value1'}),
                    ('322', 'tool_name2', '')]
            model_state:
                The current state of the Agent for the `handoff` functionality.
            vars:
                Extra kwargs to be used in tools.

        Returns:
            ToolResponses:
                Structured object containing all tool call results.
        """
        if model_state is None:
            model_state = {}

        if vars is None:
            vars = {}

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

            # Handle inject_vars
            inject_vars = config.get("inject_vars", False)
            if inject_vars:
                if isinstance(inject_vars, list):
                    for key in inject_vars:
                        if key not in vars:
                            raise ValueError(
                                f"The tool `{tool_name}` requires the injected "
                                f"parameter `{key}`, but it was not found."
                            )
                        tool_params[key] = vars[key]
                elif inject_vars is True:
                    tool_params["vars"] = vars

            if config.get("background", False):
                return_directly = False
                await F.abackground_task(tool.acall, **(tool_params or {}))
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
                        result=f"""The `{tool_name}` tool was started in the background.
                        This tool will not generate a return""",
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

            if config.get("inject_model_state", False):  # Add model_state
                tool_params["task_messages"] = model_state

            if not config.get("return_direct", False):
                return_directly = False

            tool_params = tool_params or {}
            # Add tool_call_id for telemetry
            tool_params["tool_call_id"] = tool_id
            prepared_calls.append(partial(tool.acall, **tool_params))

            call_metadata.append(
                dotdict(id=tool_id, name=tool_name, config=config, params=tool_params)
            )

        if prepared_calls:
            results = await F.ascatter_gather(prepared_calls)
            for meta, result in zip(call_metadata, results):
                if isinstance(meta.params, dict):
                    parameters = meta.params.to_dict()
                    parameters.pop("vars", None)
                    parameters.pop("tool_call_id", None)
                else:
                    parameters = None
                tool_calls.append(
                    ToolCall(
                        id=meta.id,
                        name=meta.name,
                        parameters=parameters,
                        result=result,
                    )
                )

        return ToolResponses(return_directly=return_directly, tool_calls=tool_calls)
