import inspect
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Union, Tuple

import msgspec
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from msgflux.dotdict import dotdict
from msgflux.logger import logger
from msgflux.nn import functional as F
from msgflux.nn.modules.container import ModuleDict
from msgflux.nn.modules.module import Module
from msgflux.protocols.mcp import (
    MCPClient, convert_mcp_schema_to_tool_schema, filter_tools
)
from msgflux.telemetry.span import (
    ainstrument_tool_library_call, instrument_tool_library_call
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


class ToolBase(Module):
    """Tool is Module type that provide a json schema to tools."""

    def get_json_schema(self):
        return generate_tool_json_schema(self)


def _convert_module_to_nn_tool(impl: Callable) -> ToolBase: # noqa: C901
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
            or
            getattr(impl, "__doc__", None)
            or
            getattr(impl.__call__, "__doc__", None)            
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
            or
            getattr(impl, "__annotations__", None)
            or
            getattr(impl.__call__, "__annotations__", None)            
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
            or
            getattr(impl, "name", None)
            or
            getattr(impl, "__name__", None)
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

    class Tool(ToolBase):
        def __init__(self):
            super().__init__()
            self.set_name(name)
            self.set_description(doc)
            self.set_annotations(annotations)
            self.register_buffer("tool_config", tool_config)
            self.impl = impl  # Not a buffer for now

        @tool_retry
        def forward(self, *args, **kwargs):
            if inspect.iscoroutinefunction(self.impl):
                return F.wait_for(self.impl, *args, **kwargs)
            return self.impl(*args, **kwargs)

        @tool_retry
        async def aforward(self, *args, **kwargs):
            # Check if impl has acall method first (for Module instances)
            if hasattr(self.impl, "acall"):
                return await self.impl.acall(*args, **kwargs)
            # Then check if it's a coroutine function
            elif inspect.iscoroutinefunction(self.impl):
                return await self.impl(*args, **kwargs)
            # Fall back to sync call
            else:
                return self.impl(*args, **kwargs)

    return Tool()


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
        if isinstance(tool, str):
            if tool in self.special_library.keys():
                raise ValueError(
                    f"The special tool name `{tool}` is already in special tool library"
                )
            self.special_library.append(tool)
        else:
            name = (getattr(tool, "name", None) or getattr(tool, "__name__", None))
            if name in self.library.keys():
                raise ValueError(f"The tool name `{name}` is already in tool library")
            if not isinstance(tool, ToolBase):
                tool = _convert_module_to_nn_tool(tool)

            self.tool_configs[tool.name] = tool.tool_config

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
                    timeout=server_config.get("timeout", 30.0)
                )
            elif transport_type == "http":
                client = MCPClient.from_http(
                    base_url=server_config.get("base_url"),
                    timeout=server_config.get("timeout", 30.0),
                    headers=server_config.get("headers")
                )
            else:
                raise ValueError(
                    f"Unknown transport type: {transport_type}. "
                    "Supported types: 'stdio', 'http'"
                )

            # Connect and list tools synchronously using executor
            F.wait_for(client.connect)
            all_tools = F.wait_for(client.list_tools, use_cache=False)

            # Apply filters
            include_tools = server_config.get("include_tools")
            exclude_tools = server_config.get("exclude_tools")
            filtered_tools = filter_tools(all_tools, include_tools, exclude_tools)

            # Store client and tools
            tool_configs = server_config.get("tool_config", {})
            self.mcp_clients[namespace] = {
                "client": client,
                "tools": filtered_tools,
                "tool_config": tool_configs
            }

    def get_tools(self) -> Iterator[Dict[str, ToolBase]]:
        return self.library.items()

    def get_tool_names(self) -> List[str]:
        """Get names of all local tools."""
        return list(self.library.keys())

    def get_mcp_tool_names(self) -> List[str]:
        """Get names of all MCP tools (with namespace)."""
        tool_names = []
        for namespace, mcp_data in self.mcp_clients.items():
            for tool in mcp_data["tools"]:
                tool_names.append(f"{namespace}__{tool.name}")
        return tool_names

    def get_all_tool_names(self) -> List[str]:
        """Get names of all tools (local + MCP)."""
        return self.get_tool_names() + self.get_mcp_tool_names()

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

    @instrument_tool_library_call
    def forward(  # noqa: C901
        self,
        tool_callings: List[Tuple[str, str, Any]],
        model_state: Optional[List[Dict[str, Any]]] = None,
        vars: Optional[Mapping[str, Any]] = None,
    ) -> ToolResponses:
        """Executes tool calls with logic for `handoff`, `return_direct`.

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
            # Check if it's an MCP tool (has namespace prefix)
            if "__" in tool_name:
                namespace, mcp_tool_name = tool_name.split("__", 1)

                if namespace in self.mcp_clients:
                    # Execute MCP tool
                    mcp_data = self.mcp_clients[namespace]
                    client = mcp_data["client"]
                    tool_config = mcp_data["tool_config"].get(mcp_tool_name, {})

                    # Apply inject_vars if configured
                    inject_vars = tool_config.get("inject_vars", False)
                    if inject_vars:
                        if isinstance(inject_vars, list):
                            for key in inject_vars:
                                if key in vars:
                                    tool_params[key] = vars[key]
                        elif inject_vars is True:
                            tool_params["vars"] = vars

                    # Check return_direct config
                    if not tool_config.get("return_direct", False):
                        return_directly = False

                    # Call MCP tool
                    try:
                        from msgflux.protocols.mcp import extract_tool_result_text
                        result = F.wait_for(client.call_tool, mcp_tool_name, tool_params)

                        if result.isError:
                            error_text = extract_tool_result_text(result)
                            tool_calls.append(
                                ToolCall(
                                    id=tool_id,
                                    name=tool_name,
                                    parameters=tool_params,
                                    error=error_text
                                )
                            )
                        else:
                            result_text = extract_tool_result_text(result)
                            tool_calls.append(
                                ToolCall(
                                    id=tool_id,
                                    name=tool_name,
                                    parameters=tool_params,
                                    result=result_text
                                )
                            )
                    except Exception as e:
                        error_msg = f"MCP tool execution error: {str(e)}"

                        # Log error
                        logger.error(
                            "MCP tool execution failed",
                            extra={
                                "tool_id": tool_id,
                                "tool_name": tool_name,
                                "namespace": namespace,
                                "mcp_tool_name": mcp_tool_name,
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                            exc_info=True
                        )

                        # Record telemetry
                        span = trace.get_current_span()
                        if span.is_recording():
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, error_msg))
                            span.set_attribute("mcp.tool.error", True)
                            span.set_attribute("mcp.tool.namespace", namespace)
                            span.set_attribute("mcp.tool.name", mcp_tool_name)

                        tool_calls.append(
                            ToolCall(
                                id=tool_id,
                                name=tool_name,
                                parameters=tool_params,
                                error=error_msg
                            )
                        )
                    continue

            # Local tool execution
            if tool_name not in self.library:
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
                        error=f"Error: Tool `{tool_name}` not found."
                    )
                )
                return_directly = False
                continue

            tool = self.library[tool_name]
            config = self.tool_configs.get(tool_name, {})

            inject_vars = config.get("inject_vars", False)
            if inject_vars != False:
                if not inject_vars:
                    raise ValueError(
                        f"The tool `{tool_name}` expects injected vars "
                        f"(`{inject_vars}`), but none were provided."
                    )
                if isinstance(inject_vars, list):
                    for key in inject_vars:
                        if key not in vars:
                            raise ValueError(
                                f"The tool `{tool_name}` requires the injected "
                                f"parameter `{key}`, but it was not found."
                            )
                        tool_params[key] = vars[key]
                elif inject_vars == True:
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
                        This tool will not generate a return"""
                    )
                )
                continue

            if config.get("call_as_response", False):  # return function call as response
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

    @ainstrument_tool_library_call
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
            # Check if it's an MCP tool (has namespace prefix)
            if "__" in tool_name:
                namespace, mcp_tool_name = tool_name.split("__", 1)

                if namespace in self.mcp_clients:
                    # Execute MCP tool
                    mcp_data = self.mcp_clients[namespace]
                    client = mcp_data["client"]
                    tool_config = mcp_data["tool_config"].get(mcp_tool_name, {})

                    # Apply inject_vars if configured
                    inject_vars = tool_config.get("inject_vars", False)
                    if inject_vars:
                        if isinstance(inject_vars, list):
                            for key in inject_vars:
                                if key in vars:
                                    tool_params[key] = vars[key]
                        elif inject_vars is True:
                            tool_params["vars"] = vars

                    # Check return_direct config
                    if not tool_config.get("return_direct", False):
                        return_directly = False

                    # Call MCP tool asynchronously
                    try:
                        from msgflux.protocols.mcp import extract_tool_result_text
                        result = await client.call_tool(mcp_tool_name, tool_params)

                        if result.isError:
                            error_text = extract_tool_result_text(result)
                            tool_calls.append(
                                ToolCall(
                                    id=tool_id,
                                    name=tool_name,
                                    parameters=tool_params,
                                    error=error_text
                                )
                            )
                        else:
                            result_text = extract_tool_result_text(result)
                            tool_calls.append(
                                ToolCall(
                                    id=tool_id,
                                    name=tool_name,
                                    parameters=tool_params,
                                    result=result_text
                                )
                            )
                    except Exception as e:
                        error_msg = f"MCP tool execution error: {str(e)}"

                        # Log error
                        logger.error(
                            "MCP tool execution failed",
                            extra={
                                "tool_id": tool_id,
                                "tool_name": tool_name,
                                "namespace": namespace,
                                "mcp_tool_name": mcp_tool_name,
                                "error": str(e),
                                "error_type": type(e).__name__,
                            },
                            exc_info=True
                        )

                        # Record telemetry
                        span = trace.get_current_span()
                        if span.is_recording():
                            span.record_exception(e)
                            span.set_status(Status(StatusCode.ERROR, error_msg))
                            span.set_attribute("mcp.tool.error", True)
                            span.set_attribute("mcp.tool.namespace", namespace)
                            span.set_attribute("mcp.tool.name", mcp_tool_name)

                        tool_calls.append(
                            ToolCall(
                                id=tool_id,
                                name=tool_name,
                                parameters=tool_params,
                                error=error_msg
                            )
                        )
                    continue

            # Local tool execution
            if tool_name not in self.library:
                tool_calls.append(
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
                        error=f"Error: Tool `{tool_name}` not found."
                    )
                )
                return_directly = False
                continue

            tool = self.library[tool_name]
            config = self.tool_configs.get(tool_name, {})

            inject_vars = config.get("inject_vars", False)
            if inject_vars != False:
                if not inject_vars:
                    raise ValueError(
                        f"The tool `{tool_name}` expects injected vars "
                        f"(`{inject_vars}`), but none were provided."
                    )
                if isinstance(inject_vars, list):
                    for key in inject_vars:
                        if key not in vars:
                            raise ValueError(
                                f"The tool `{tool_name}` requires the injected "
                                f"parameter `{key}`, but it was not found."
                            )
                        tool_params[key] = vars[key]
                elif inject_vars == True:
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
                        This tool will not generate a return"""
                    )
                )
                continue

            if config.get("call_as_response", False):  # return function call as response
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
