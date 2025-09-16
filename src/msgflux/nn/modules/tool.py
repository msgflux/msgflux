import inspect
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import msgspec

from msgflux.dotdict import dotdict
from msgflux.nn import functional as F
from msgflux.nn.modules.container import ModuleDict
from msgflux.nn.modules.module import Module
from msgflux.telemetry.span import instrument_tool_library_call
from msgflux.utils.chat import generate_tool_json_schema
from msgflux.utils.convert import convert_camel_to_snake_case
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
            raise NotImplementedError(
                "To transform a class in `nn.Tool` is necessary "
                "to implement annotations of types hint in "
                "`self.annotations`, `self.__annotations__` or in `def __call__`"
            )

        raw_name = (getattr(impl, "name", None) or getattr(impl, "__name__", None))
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

        if hasattr(impl, "__annotations__"):
            annotations = impl.__annotations__
        else:
            raise NotImplementedError(
                "To transform a function into a `nn.Tool` "
                "is necessary to implement parameters "
                "annotations of types hint "
            )

        name = impl.__name__

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

    return Tool()


class ToolLibrary(Module):
    """ToolLibrary is a Module type that manage tool calls over the tool library."""

    def __init__(
        self,
        name: str,
        tools: List[Callable],
        special_tools: Optional[List[str]] = None,
    ):
        """Args:
        name:
            Library name.
        tools:
            A list of callables.
        special_tools:
            Autonomy tools for the model.
        """
        super().__init__()
        self.set_name(f"{name}_tool_library")
        self.library = ModuleDict()
        self.register_buffer("special_library", [])
        self.register_buffer("tool_configs", {})
        for tool in tools:
            self.add(tool)
        if special_tools:
            for special_tool in special_tools:
                self.special_add(special_tool)

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

    def get_tools(self) -> Iterator[Dict[str, ToolBase]]:
        return self.library.items()

    def get_tool_names(self) -> List[str]:
        return list(self.library.keys())

    def get_tool_json_schemas(self) -> List[Dict[str, Any]]:
        """Returns a list of JSON schemas from functions."""
        # TODO: support to especial and mcp tool schemas
        return [self.library[tool_name].get_json_schema() for tool_name in self.library]

    @instrument_tool_library_call
    def forward(  # noqa: C901
        self,
        tool_callings: List[Tuple[str, str, Any]],
        model_state: Optional[List[Dict[str, Any]]] = None,
        injected_kwargs: Optional[Dict[str, Any]] = None,
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
            injected_kwargs:
                Extra kwargs to be used in tools.

        Returns:
            ToolResponses:
                Structured object containing all tool call results.
        """
        if injected_kwargs is None:
            injected_kwargs = {}

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
                        error=f"Error: Tool `{tool_name}` not found."
                    )
                )
                return_directly = False
                continue

            tool = self.library[tool_name]
            config = self.tool_configs.get(tool_name, {})

            inject_kwargs = config.get("inject_kwargs", False)
            if inject_kwargs is not False:
                if not injected_kwargs:
                    raise ValueError(
                        f"The tool `{tool_name}` expects injected parameters "
                        f"(`{inject_kwargs}`), but none were provided."
                    )
                if isinstance(inject_kwargs, list):
                    for key in inject_kwargs:
                        if key not in injected_kwargs:
                            raise ValueError(
                                f"The tool `{tool_name}` requires the injected "
                                f"parameter `{key}`, but it was not found."
                            )
                        tool_params[key] = injected_kwargs[key]
                elif inject_kwargs is True:
                    tool_params.update(injected_kwargs)

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
                    ToolCall(
                        id=tool_id,
                        name=tool_name,
                        parameters=tool_params,
                        result=None  # intentionally left None
                    )
                )
                return_directly = True
                continue

            if config.get("handoff", False):  # Add model_state
                tool_params["task_messages"] = model_state
                tool_params["message"] = None

            if not config.get("return_direct", False):
                return_directly = False

            if tool_params:
                prepared_calls.append(partial(tool, **tool_params))
            else:
                prepared_calls.append(partial(tool, None))

            call_metadata.append(
                dotdict({"id": tool_id, "name": tool_name, "config": config, "params": tool_params})
            )

        if prepared_calls:                     
            results = F.scatter_gather(prepared_calls)
            for meta, result in zip(call_metadata, results):
                tool_calls.append(
                    ToolCall(
                        id=meta.id,
                        name=meta.name,
                        parameters=meta.params.to_dict(),
                        result=result,
                    )
                )

        return ToolResponses(return_directly=return_directly, tool_calls=tool_calls)
