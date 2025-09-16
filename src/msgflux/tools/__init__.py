from functools import wraps
from types import FunctionType, MethodType
from typing import Callable, Dict, List, Optional, Union

from msgflux.dotdict import dotdict


class ToolFlowControl:
    """Base class for creating custom tool flow controls
    based on generation schamas.

    Each generation schema, such as ReAct, can be
    treated as a custom tool flow control by
    inheriting from this class.
    """


def tool_config(
    *,
    return_direct: Optional[bool] = False,
    background: Optional[bool] = False,
    handoff: Optional[bool] = False,
    call_as_response: Optional[bool] = False,
    inject_kwargs: Optional[Union[bool, List[str]]] = False,
    name_override: Optional[str] = None,
) -> Callable:
    """Decorator to inject meta-properties into a function or class instance.

    This decorator adds metadata properties to a function or an instance of a
    class, allowing tools to control behavior such as whether results are
    returned directly or passed for further handling, and optionally override
    the tool's registered name.

    Args:
        return_direct:
            If True, the tool will return its output directly without additional
            processing.
        background:
            If True, the tool will be executed in the background and a message
            that the task has been scheduled will be the response to the model.
        handoff:
            If True, indicates that this function will receive the `model_state`
            from the Agent.
        call_as_response:
            If True, returns the tool call as its result. This property requires
            `return_direct = True` and will automatically change it to True if it
            is passed as false.
        inject_kwargs:
            Indicates if the tool should receive kwargs. If True, the tool receives all
            kwargs. If a list of kwargs is passed, only those kwargs will be passed.
        name_override:
            A custom name to override the default tool name derived from the function
            or class. If not provided, the original name is used.

    Returns:
        A decorator that modifies the target function or class instance
        by injecting the specified properties.

    Raises:
        ValueError:
           `background=True` is not compatible with `return_direct=True`,
           `call_as_response=True` and `handoff=True`.
        ValueError:
           `handoff=True` is not compatible with `call_as_response=True`.
        ValueError:
           `inject_kwargs=True` is not compatible with `call_as_response=True`.
    """

    def decorator(f):
        _return_direct = return_direct  # Local copy

        if background is True and (return_direct is True or handoff is True):
            raise ValueError(
                "`background=True` is not compatible with `return_direct=True`"
                ", `call_as_response=True` and `handoff=True`"
            )

        if handoff is True and call_as_response is True:
            raise ValueError(
                "`handoff=True` is not compatible with `call_as_response=True`"
            )

        if inject_kwargs is not False and call_as_response is True:
            raise ValueError(
                "`inject_kwargs=True` is not compatible with `call_as_response=True`"
            )

        if call_as_response is True and _return_direct is False:
            _return_direct = True

        tool_config = {
            "tool_config": dotdict(
                {
                    "background": background,
                    "call_as_response": call_as_response,
                    "handoff": handoff,
                    "inject_kwargs": inject_kwargs,
                    "return_direct": _return_direct,
                }
            )
        }
        if isinstance(f, (FunctionType, MethodType)):
            return decorate_function(f, name_override, tool_config)
        if isinstance(f, type):  # Not initialized class
            f = f()  # Init class
        return decorate_instance(f, name_override, tool_config)

    return decorator


def decorate_function(
    func: Union[FunctionType, MethodType],
    override_name: str,
    tool_config: Dict[str, Union[bool, str]],
) -> Union[FunctionType, MethodType]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    if override_name:
        wrapper.__name__ = override_name
        wrapper.__qualname__ = override_name

    wrapper.__dict__.update(tool_config)
    return wrapper


def decorate_instance(
    instance: Callable, override_name: str, tool_config: Dict[str, Union[bool, str]]
) -> Callable:
    if override_name:
        instance.__name__ = override_name
        instance.__qualname__ = override_name

    instance.__dict__.update(tool_config)
    return instance
