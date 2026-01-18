from functools import wraps
from types import FunctionType, MethodType
from typing import Callable, Dict, List, Optional, Union

from msgflux.dotdict import dotdict


def tool_config(
    *,
    return_direct: Optional[bool] = False,
    call_as_response: Optional[bool] = False,
    background: Optional[bool] = False,
    inject_model_state: Optional[bool] = False,
    inject_vars: Optional[Union[bool, List[str]]] = False,
    handoff: Optional[bool] = False,
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
        call_as_response:
            If True, returns the tool call as its result. This property requires
            `return_direct = True` and will automatically change it to True if it
            is passed as false.
        background:
            If True, the tool will be executed in the background and a message
            that the task has been scheduled will be the response to the model.
        inject_model_state:
            If true, the tool automatically sets `inject_model_state` and `return_direct`
            to `True`. Additionally, the tool will be renamed to `transfer_to_<name>`.
            Any input parameters for this tool will be removed. The tool will **only**
            receive `model_state` as a parameter.
        inject_vars:
            Indicates if the tool should receive vars. If True, the tool receives all
            vars as a named argument `vars`. If a list of vars is passed, only those
            vars will be passed.
        handoff:
            If True, indicates that this function will receive the `model_state`
            from the Agent.
        name_override:
            A custom name to override the default tool name derived from the function
            or class. If not provided, the original name is used.

    Returns:
        A decorator that modifies the target function or class instance
        by injecting the specified properties.

    Raises:
        ValueError:
           `background=True` is not compatible with `return_direct=True` and
           `call_as_response=True`.
        ValueError:
           `inject_vars=True` is not compatible with `call_as_response=True`.
    """

    def decorator(f):
        _return_direct = return_direct  # Local copy
        _inject_model_state = inject_model_state  # Local copy

        if call_as_response is True and _return_direct is False:
            _return_direct = True

        if handoff:
            _return_direct = True
            _inject_model_state = True

        if background and (_return_direct or call_as_response):
            raise ValueError(
                "`background=True` is not compatible with `return_direct=True`"
                " and `call_as_response=True`."
            )

        if inject_vars is not False and call_as_response is True:
            raise ValueError(
                "`inject_vars` is not compatible with `call_as_response=True`"
            )

        tool_config = {
            "tool_config": dotdict(
                {
                    "background": background,
                    "call_as_response": call_as_response,
                    "handoff": handoff,
                    "inject_model_state": _inject_model_state,
                    "inject_vars": inject_vars,
                    "return_direct": _return_direct,
                    "name_overridden": name_override,
                }
            )
        }
        if isinstance(f, (FunctionType, MethodType)):
            return decorate_function(f, tool_config)
        if isinstance(f, type):  # Not initialized class
            f = f()  # Init class
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


def decorate_instance(
    instance: Callable, tool_config: Dict[str, Union[bool, str]]
) -> Callable:
    instance.__dict__.update(tool_config)
    return instance
