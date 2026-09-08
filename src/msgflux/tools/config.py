from functools import wraps
from types import FunctionType, MethodType
from typing import Any, Callable, Collection, Dict, Optional, Union

from msgflux.core.dotdict import dotdict
from msgflux.runtime.permissions import normalize_permissions
from msgflux.tools.helpers import normalize_background_capabilities
from msgflux.tools.runtime import FeedbackSpec
from msgflux.tools.specs import ContextBinding, ContextSpec


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


def _normalize_feedback(
    feedback: Optional[Union[str, FeedbackSpec]],
    *,
    return_direct: Optional[bool],
    handoff: Optional[bool],
    call_as_response: Optional[bool],
) -> FeedbackSpec | None:
    if feedback is None:
        return None
    if return_direct or handoff or call_as_response:
        raise ValueError(
            "`feedback` cannot be combined with `return_direct=True`, "
            "`handoff=True`, or `call_as_response=True`."
        )
    return FeedbackSpec.coerce(feedback)


def _normalize_dispatch(
    dispatch: Optional[str],
    *,
    background: Optional[bool],
    allow_background: Optional[bool],
    detached: Optional[bool],
) -> str | None:
    if dispatch is not None and (not isinstance(dispatch, str) or not dispatch.strip()):
        raise ValueError("`dispatch` must be a non-empty string or None")
    if dispatch is not None and (background or allow_background or detached):
        raise ValueError(
            "`dispatch` cannot be combined with `background=True`, "
            "`allow_background=True`, or `detached=True`."
        )
    return dispatch


def _normalize_runtime_inputs(
    runtime_inputs: ContextSpec | Collection[ContextBinding | str] | None,
    *,
    handoff: Optional[bool],
) -> ContextSpec:
    if runtime_inputs is not None:
        bindings = list(ContextSpec.coerce(runtime_inputs).bindings)
    else:
        bindings = []
    if handoff and not any(binding.source == "messages" for binding in bindings):
        bindings.append(ContextBinding(source="messages"))
    return ContextSpec(bindings=tuple(bindings))


def tool_config(
    *,
    description: Optional[str] = None,
    display_name: Optional[str] = None,
    usage_guidance: Optional[str] = None,
    feedback: Optional[Union[str, FeedbackSpec]] = None,
    dispatch: Optional[str] = None,
    return_direct: Optional[bool] = False,
    call_as_response: Optional[bool] = False,
    detached: Optional[bool] = False,
    background: Optional[bool] = False,
    allow_background: Optional[bool] = False,
    background_capabilities: Optional[Collection[str]] = None,
    disable_input: Optional[bool] = False,
    defer_loading: Optional[bool] = False,
    runtime_inputs: ContextSpec | Collection[ContextBinding | str] | None = None,
    handoff: Optional[bool] = False,
    tool_kind: Optional[str] = None,
    name_override: Optional[str] = None,
    retry: Optional[Any] = None,
    required_permissions: Collection[str] = (),
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
        description:
            Optional model-facing description. When set, it overrides the callable's
            class attribute or docstring without mutating the callable.
        return_direct:
            If True, the tool will return its output directly without additional
            processing.
        display_name:
            Human-readable name for UI/events. If omitted, the tool name is used.
        usage_guidance:
            Optional guidance describing when and how an agent should use the tool.
            Agents may render this in their system prompt.
        feedback:
            Optional Agent feedback mode selected after execution. Use a string for
            an application-defined mode or FeedbackSpec for a mode with options.
            It cannot be combined with the legacy return_direct, handoff, or
            call_as_response aliases.
        dispatch:
            Optional registered dispatch name. It cannot be combined with the
            background, allow_background, or detached convenience options.
        call_as_response:
            If True, returns the tool call as its result. This property requires
            `return_direct = True` and will automatically change it to True if it
            is passed as false.
        detached:
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
        defer_loading:
            If True, keep the tool registered in the library but hide its schema
            from the model until it is loaded through `tool_search`.
        runtime_inputs:
            Runtime context sources injected outside the model-visible schema.
            Accepts source names or ContextBinding values for custom parameters
            and provider-specific options.
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
        required_permissions:
            Exact capabilities required from the live execution scope. The runtime
            blocks missing grants before tool preparation; this is not a sandbox.

    Returns:
        A decorator that modifies the target by injecting the specified properties.
        - For functions: returns a wrapped function with properties
        - For classes: returns the modified class (all instances will have properties)
        - For instances: returns the instance with injected properties

    Raises:
        ValueError:
           `detached=True` is not compatible with `return_direct=True`
           and `call_as_response=True`.
        ValueError:
           `background=True` is not compatible with `return_direct=True`,
           `call_as_response=True`, `detached=True`, and `handoff=True`.
        ValueError:
           `allow_background=True` is not compatible with `return_direct=True`,
           `call_as_response=True`, `detached=True`, and `handoff=True`.
        ValueError:
           `runtime_inputs` is not compatible with `call_as_response=True`.

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
    normalized_permissions = normalize_permissions(required_permissions)

    def decorator(f):
        _return_direct = return_direct  # Local copy

        normalized_feedback = _normalize_feedback(
            feedback,
            return_direct=_return_direct,
            handoff=handoff,
            call_as_response=call_as_response,
        )

        normalized_dispatch = _normalize_dispatch(
            dispatch,
            background=background,
            allow_background=allow_background,
            detached=detached,
        )

        if call_as_response is True and _return_direct is False:
            _return_direct = True

        if handoff:
            _return_direct = True

        normalized_context = _normalize_runtime_inputs(
            runtime_inputs,
            handoff=handoff,
        )

        if detached and (_return_direct or call_as_response):
            raise ValueError(
                "`detached=True` is not compatible with `return_direct=True`"
                " and `call_as_response=True`."
            )

        if background and (_return_direct or call_as_response or detached or handoff):
            raise ValueError(
                "`background=True` is not compatible with `return_direct=True`,"
                " `call_as_response=True`, `detached=True`, and `handoff=True`."
            )

        if allow_background and (
            _return_direct or call_as_response or detached or handoff
        ):
            raise ValueError(
                "`allow_background=True` is not compatible with "
                "`return_direct=True`, `call_as_response=True`, `detached=True`, "
                "and `handoff=True`."
            )

        normalized_background_capabilities = (
            _normalize_configured_background_capabilities(
                background=background,
                allow_background=allow_background,
                capabilities=background_capabilities,
            )
        )

        if normalized_context.bindings and call_as_response is True:
            raise ValueError(
                "`runtime_inputs` is not compatible with `call_as_response=True`"
            )

        tool_config = {
            "tool_config": dotdict(
                {
                    "description": description,
                    "dispatch": normalized_dispatch,
                    "detached": detached,
                    "background": background,
                    "allow_background": allow_background,
                    "background_capabilities": normalized_background_capabilities,
                    "display_name": display_name,
                    "usage_guidance": usage_guidance,
                    "feedback": normalized_feedback,
                    "call_as_response": call_as_response,
                    "handoff": handoff,
                    "disable_input": disable_input,
                    "defer_loading": defer_loading,
                    "runtime_inputs": normalized_context,
                    "return_direct": _return_direct,
                    "tool_kind": tool_kind,
                    "name_overridden": name_override,
                    "retry": retry,
                    "required_permissions": normalized_permissions,
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
