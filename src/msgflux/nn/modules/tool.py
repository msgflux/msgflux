import inspect
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


class ToolBase(Module):
    """Tool is Module type that provide a json schema to tools."""

    def get_json_schema(self):
        return generate_tool_json_schema(self)


def _convert_module_to_nn_tool(impl: Callable) -> ToolBase: # noqa: C901
    """Convert a callable in nn.Tool."""
    tool_config = impl.__dict__.get("tool_config", {})

    # Case 1: Uninitialized or initialized class
    if inspect.isclass(impl) or callable(impl):
        if not callable(impl):
            raise NotImplementedError(
                "To transform a class in `nn.Tool`"
                " is necessary implement a `def __call__`"
            )

        if hasattr(impl, "docstring") and impl.docstring is not None:
            doc = impl.docstring
        elif hasattr(impl, "__doc__") and impl.__doc__ is not None:
            doc = impl.__doc__
        elif hasattr(impl.__call__, "__doc__") and impl.__call__.__doc__ is not None:
            doc = impl.__call__.__doc__
        else:
            raise NotImplementedError(
                "To transform a class into a `nn.Tool` "
                "it is necessary to implement a docstring. "
                "Can be: a cls attr `self.docstring`, or"
                "a docstring in the class or in `def __call__`"
            )

        if hasattr(impl, "annotations"):
            annotations = impl.annotations
        elif hasattr(impl, "__annotations__"):
            annotations = impl.__annotations__
        elif hasattr(impl.__call__, "__annotations__"):
            annotations = impl.__call__.__annotations__
        else:
            raise NotImplementedError(
                "To transform a class in `nn.Tool` is necessary "
                "to implement annotations of types hint in "
                "`self.annotations`, `self.__annotations__` or in `def __call__`"
            )

        name = convert_camel_to_snake_case(impl.__name__)

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

    if tool_config.handoff:
        name = "transfer_to_" + name

    if tool_config.background:
        doc = "This tool will run in the background. \n" + doc

    class Tool(ToolBase):
        def __init__(self):
            super().__init__()
            self.set_name(name)
            self.set_description(doc)
            self._set_annotations(annotations)
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
            name = tool.name if isinstance(tool, ToolBase) else tool.__name__
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
    def forward( # noqa: C901
        self,
        tool_callings: List[Tuple[str, str, Any]],
        model_state: Optional[List[Dict[str, Any]]] = None,
        injected_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
            A dict containing `return_directly` and `responses`.
            Where responses will be a mapping from `tool_name` to
            `tool_response` if `return_directly=True` or `tool_id`
            to `tool_response` if `return_directly=False`.
        """
        if injected_kwargs is None:
            injected_kwargs = {}
        prepared_calls = []
        call_metadata = []
        responses = {}
        return_directly = True if tool_callings else False

        for tool_id, tool_name, tool_params in tool_callings:
            if tool_name not in self.library:
                responses[tool_id] = f"Error: Tool `{tool_name}` not found."
                return_directly = False  # Errors should always be returned to the model
                continue

            tool = self.library[tool_name]
            config = self.tool_configs.get(tool_name)

            inject_kwargs = config.get("inject_kwargs", False)
            if inject_kwargs is not False:
                if not injected_kwargs:
                    raise ValueError(
                        f"The tool `{tool_name}` expect injected parameters "
                        f"(`{inject_kwargs}`), but no one were provided."
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
                background_tool_params = tool_params or {}
                F.background_task(tool, **background_tool_params)
                result = f"""The `{tool_name}` tool was started in the background.
                This tool will not generate a return"""
                responses[tool_id] = result
                continue

            if config.get("call_as_response", False):  # Return direct
                responses[tool_name] = {
                    "tool_name": tool_name,
                    "parameters": tool_params,
                }
                return_directly = True
                continue

            if config.get("handoff", False):  # Add model_state
                tool_params.task_messages = model_state  # Will ALWAYS have 'message'

            if not config.get("return_direct", False):
                return_directly = False  # Disable direct return

            if tool_params:
                prepared_calls.append(partial(tool, **tool_params))
            else:
                prepared_calls.append(partial(tool, None))  # No params

            call_metadata.append(
                dotdict({"id": tool_id, "name": tool_name, "config": config})
            )

        if prepared_calls:
            results = F.scatter_gather(prepared_calls)
            for meta, result in zip(call_metadata, results):
                if return_directly:  # tool_name -> result
                    responses[meta.name] = result
                else:  # tool_id -> result
                    encoded_result = None
                    if not isinstance(result, str):
                        encoded_result = msgspec.json.encode(result).decode("utf-8")
                    responses[meta.id] = encoded_result or result

        return dotdict({"return_directly": return_directly, "responses": responses})
