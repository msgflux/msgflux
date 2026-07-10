from functools import wraps
from types import FunctionType, MethodType
from typing import Any, Callable, Collection, Dict, List, Optional, Union

from msgflux.core.dotdict import dotdict
from msgflux.tools.helpers import normalize_background_capabilities


def _normalize_configured_background_capabilities(
    *,
    background: Optional[bool],
    allow_background: Optional[bool],
    capabilities: Optional[Collection[str]],
) -> tuple[str, ...] | None:
    if capabilities is None:
        return None
    if not (background or allow_background):
        raise ValueError(
            "`background_capabilities` requires `background=True` or "
            "`allow_background=True`."
        )
    return normalize_background_capabilities(capabilities)


def tool_config(
    *,
    display_name: Optional[str] = None,
    usage_guidance: Optional[str] = None,
    return_direct: Optional[bool] = False,
    call_as_response: Optional[bool] = False,
    spawn: Optional[bool] = False,
    background: Optional[bool] = False,
    allow_background: Optional[bool] = False,
    background_capabilities: Optional[Collection[str]] = None,
    disable_input: Optional[bool] = False,
    on_demand: Optional[bool] = False,
    inject_message: Optional[bool] = False,
    inject_messages: Optional[bool] = False,
    inject_handle: Optional[bool] = False,
    inject_vars: Optional[Union[bool, List[str]]] = False,
    handoff: Optional[bool] = False,
    tool_kind: Optional[str] = None,
    name_override: Optional[str] = None,
    retry: Optional[Any] = None,
) -> Callable:
    """Decorator to inject meta-properties into functions, classes, or instances.

    This decorator adds metadata properties to control tool behavior such as whether
    results are returned directly or passed for further handling, and optionally
    override the tool's registered name.

    Behavior depends on what is being decorated:
    - **Functions**: Wraps the function and injects properties into the wrapper
    - **Classes**: Modifies the class's __init__ to inject properties into all
      future instances. This allows classes with required parameters to be decorated.
    - **Instances**: Directly injects properties into the instance

    Args:
        return_direct:
            If True, the tool will return its output directly without additional
            processing.
        display_name:
            Human-readable name for UI/events. If omitted, the tool name is used.
        usage_guidance:
            Optional guidance describing when and how an agent should use the tool.
            Agents may render this in their system prompt.
        call_as_response:
            If True, returns the tool call as its result. This property requires
            `return_direct = True` and will automatically change it to True if it
            is passed as false.
        spawn:
            If True, the tool will be dispatched without waiting for a result.
            The model receives a confirmation that the task was started.
        background:
            If True, the tool runs in the background and returns a `task_id`
            immediately. The result can be retrieved later via `task_status`
            and `task_output`.
        allow_background:
            If True, the model can choose whether to run the tool in the
            background by setting the reserved `run_in_background` tool
            argument. When false or null, the tool runs normally. Manual
            callers may omit the argument, which is treated as false.
        background_capabilities:
            Optional task controls supported by this background tool. Valid
            values are `activity` and `message`. Agents receive their defaults
            when this option is omitted.
        disable_input:
            If True, removes public input parameters from the tool schema. The model
            will call the tool with no explicit arguments, and any arguments supplied
            by the model are ignored at runtime. This does not inject any runtime
            context by itself.
        on_demand:
            If True, keep the tool registered in the library but hide its schema
            from the model until it is loaded through `tool_search`.
        inject_message:
            If True, the tool receives the original `message` passed to the Agent
            at runtime. This injected parameter does not become part of the tool
            schema exposed to the model.
        inject_messages:
            If True, the tool receives the current conversation history as
            `messages` at runtime. This injected parameter does not become part of
            the tool schema exposed to the model.
        inject_handle:
            If True, the tool receives the current `ToolLibraryHandle` as
            `handle` at runtime. This injected parameter does not become part of
            the tool schema exposed to the model.
        inject_vars:
            Indicates if the tool should receive vars. If True, the tool receives all
            vars as a named argument `vars`. If a list of vars is passed, only those
            vars will be passed.
        handoff:
            If True, indicates that this function will receive the `messages`
            from the Agent.
        tool_kind:
            Optional kind used by `ToolBucket` to group related tools.
        name_override:
            A custom name to override the default tool name derived from the function
            or class. If not provided, the original name is used.
        retry:
            Retry configuration for this tool. Accepts a tenacity retry decorator
            for custom retry behavior, False to disable retry, or None (default)
            to use the default retry from envs.

    Returns:
        A decorator that modifies the target by injecting the specified properties.
        - For functions: returns a wrapped function with properties
        - For classes: returns the modified class (all instances will have properties)
        - For instances: returns the instance with injected properties

    Raises:
        ValueError:
           `spawn=True` is not compatible with `return_direct=True`
           and `call_as_response=True`.
        ValueError:
           `background=True` is not compatible with `return_direct=True`,
           `call_as_response=True`, `spawn=True`, and `handoff=True`.
        ValueError:
           `allow_background=True` is not compatible with `return_direct=True`,
           `call_as_response=True`, `spawn=True`, and `handoff=True`.
        ValueError:
           `inject_vars=True` is not compatible with `call_as_response=True`.

    Examples:
        Decorating a function:
            >>> @tool_config(return_direct=True)
            ... def my_tool(query: str) -> str:
            ...     return f"Result: {query}"
            >>> my_tool.tool_config.return_direct
            True

        Decorating a class (all instances will have tool_config):
            >>> @tool_config(return_direct=True)
            ... class SentimentClassifier(nn.Agent):
            ...     def __init__(self, model):
            ...         super().__init__(model=model)
            >>> classifier = SentimentClassifier(model=my_model)
            >>> classifier.tool_config.return_direct
            True

        Decorating an instance:
            >>> classifier = SentimentClassifier(model=my_model)
            >>> classifier = tool_config(return_direct=True)(classifier)
            >>> classifier.tool_config.return_direct
            True
    """

    def decorator(f):
        _return_direct = return_direct  # Local copy
        _inject_message = inject_message  # Local copy
        _inject_messages = inject_messages  # Local copy

        if call_as_response is True and _return_direct is False:
            _return_direct = True

        if handoff:
            _return_direct = True
            _inject_messages = True

        if spawn and (_return_direct or call_as_response):
            raise ValueError(
                "`spawn=True` is not compatible with `return_direct=True`"
                " and `call_as_response=True`."
            )

        if background and (_return_direct or call_as_response or spawn or handoff):
            raise ValueError(
                "`background=True` is not compatible with `return_direct=True`,"
                " `call_as_response=True`, `spawn=True`, and `handoff=True`."
            )

        if allow_background and (
            _return_direct or call_as_response or spawn or handoff
        ):
            raise ValueError(
                "`allow_background=True` is not compatible with "
                "`return_direct=True`, `call_as_response=True`, `spawn=True`, "
                "and `handoff=True`."
            )

        normalized_background_capabilities = (
            _normalize_configured_background_capabilities(
                background=background,
                allow_background=allow_background,
                capabilities=background_capabilities,
            )
        )

        if inject_vars is not False and call_as_response is True:
            raise ValueError(
                "`inject_vars` is not compatible with `call_as_response=True`"
            )

        tool_config = {
            "tool_config": dotdict(
                {
                    "spawn": spawn,
                    "background": background,
                    "allow_background": allow_background,
                    "background_capabilities": normalized_background_capabilities,
                    "display_name": display_name,
                    "usage_guidance": usage_guidance,
                    "call_as_response": call_as_response,
                    "handoff": handoff,
                    "disable_input": disable_input,
                    "on_demand": on_demand,
                    "inject_message": _inject_message,
                    "inject_messages": _inject_messages,
                    "inject_handle": inject_handle,
                    "inject_vars": inject_vars,
                    "return_direct": _return_direct,
                    "tool_kind": tool_kind,
                    "name_overridden": name_override,
                    "retry": retry,
                }
            )
        }
        if isinstance(f, (FunctionType, MethodType)):
            return decorate_function(f, tool_config)
        if isinstance(f, type):  # Is a class (not an instance)
            # Create a new subclass with tool_config injected
            return decorate_class(f, tool_config)
        # Is an instance
        return decorate_instance(f, tool_config)

    return decorator


def decorate_function(
    func: Union[FunctionType, MethodType],
    tool_config: Dict[str, Union[bool, str]],
) -> Union[FunctionType, MethodType]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__dict__.update(tool_config)
    return wrapper


def decorate_class(cls: type, tool_config: Dict[str, Union[bool, str]]) -> type:
    """Decorates a class by injecting tool_config as a class attribute.

    Injects tool_config directly into the class to make it accessible both
    from the class itself and from instances. This is compatible with AutoParams
    and doesn't interfere with __init__.

    Args:
        cls: The class to decorate
        tool_config: Dictionary containing tool configuration properties

    Returns:
        The class with tool_config injected as a class attribute
    """
    # Inject tool_config directly as class attribute
    # This works for both class-level access (when used as tool) and instance access
    cls.tool_config = tool_config["tool_config"]

    return cls


def decorate_instance(
    instance: Callable, tool_config: Dict[str, Union[bool, str]]
) -> Callable:
    instance.__dict__.update(tool_config)
    return instance
