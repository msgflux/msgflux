import asyncio
import functools
import inspect
import re
import weakref
from collections import OrderedDict, namedtuple
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import msgspec
try:
    from code2mermaid import code_to_mermaid
except ImportError:
    code_to_mermaid = None
from jinja2 import Template
from opentelemetry import trace

from msgflux._private.executor import Executor
from msgflux.dotdict import dotdict
from msgflux.envs import envs
from msgflux.exceptions import UnsafeModelResponseError, UnsafeUserInputError
from msgflux.logger import logger
from msgflux.message import Message
from msgflux.models.gateway import ModelGateway
from msgflux.models.model import Model
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.nn.parameter import Parameter
from msgflux.telemetry.span import spans
from msgflux.utils.convert import convert_camel_snake_to_title
from msgflux.utils.encode import encode_data_to_base64
from msgflux.utils.hooks import RemovableHandle
from msgflux.utils.mermaid import plot_mermaid
from msgflux.utils.msgspec import StructFactory
from msgflux.utils.validation import is_base64, is_builtin_type, is_subclass_of

__all__ = [
    "Module",
    "register_module_buffer_registration_hook",
    "register_module_forward_hook",
    "register_module_forward_pre_hook",
    "register_module_module_registration_hook",
    "register_module_parameter_registration_hook",
]

MSGFLUX_DESERIALIZABLE_CLS: Dict[str, Type] = {
    "model": Model,
    "model_gateway": ModelGateway
}


T = TypeVar("T", bound="Module")

# TODO para serializar basta verificar se o obj tem msgflux_type
# isso resolve em vez de ficar add manualmente quais são


class _IncompatibleKeys(  # TODO tirar. tirar nao porra. tem que ficar
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"]),
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super().__repr__()

    __str__ = __repr__


def _addindent(s_, num_spaces: int):
    s = s_.split("\n")
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(num_spaces * " ") + line for line in s]
    s = "\n".join(s)
    s = first + "\n" + s
    return s


def get_callable_name(callable: Callable) -> str: # noqa: A002
    if isinstance(callable, Module):
        return callable.get_module_name()
    elif inspect.isfunction(callable):
        return callable.__name__
    else:
        return callable.__class__.__name__


r"""This tracks hooks common to all modules that are executed immediately before
.registering the buffer/module/parameter"""
_global_buffer_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_module_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_parameter_registration_hooks: Dict[int, Callable] = OrderedDict()


class _WrappedHook:
    def __init__(self, hook: Callable, module: Optional["Module"] = None):
        self.hook: Callable = hook
        functools.update_wrapper(self, hook)

        self.with_module: bool = False

        if module is not None:
            self.module: weakref.ReferenceType[Module] = weakref.ref(module)
            self.with_module = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.with_module:
            module = self.module()
            if module is None:
                raise RuntimeError("You are trying to call the hook of a dead Module!")
            return self.hook(module, *args, **kwargs)
        return self.hook(*args, **kwargs)

    def __getstate__(self) -> Dict:
        result = {"hook": self.hook, "with_module": self.with_module}
        if self.with_module:
            result["module"] = self.module()

        return result

    def __setstate__(self, state: Dict):
        self.hook = state["hook"]
        self.with_module = state["with_module"]

        if self.with_module:
            if state["module"] is None:
                raise RuntimeError(
                    "You are trying to revive the hook of a dead Module!"
                )
            self.module = weakref.ref(state["module"])


r"""This tracks hooks common to all modules that are executed before/after
calling forward. This is global state used for debugging/profiling
purposes"""
_global_forward_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_forward_hooks: Dict[int, Callable] = OrderedDict()
_global_forward_hooks_always_called: Dict[int, bool] = OrderedDict()
_global_forward_hooks_with_kwargs: Dict[int, bool] = OrderedDict()


def register_module_buffer_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    """Register a buffer registration hook common to all modules.

    !!! warning

        This adds global state to the `nn.Module` module

    The hook will be called every time `register_buffer` is invoked.
    It should have the following signature:

        hook(module, name, buffer) -> None or new buffer

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        msgflux.nn.utils.hooks.RemovableHandle: A handle that can be used
        to remove the added hook by calling ``handle.remove()``
    """
    handle = RemovableHandle(_global_buffer_registration_hooks)
    _global_buffer_registration_hooks[handle.id] = hook
    return handle


def register_module_module_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    """Register a module registration hook common to all modules.

    !!! warning

        This adds global state to the `nn.Module` module

    The hook will be called every time `register_module` is invoked.
    It should have the following signature::

        hook(module, name, submodule) -> None or new submodule

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        msgflux.nn.utils.hooks.RemovableHandle: A handle that can be used
        to remove the added hook by calling ``handle.remove()``
    """
    handle = RemovableHandle(_global_module_registration_hooks)
    _global_module_registration_hooks[handle.id] = hook
    return handle


def register_module_parameter_registration_hook(
    hook: Callable[..., None],
) -> RemovableHandle:
    """Register a parameter registration hook common to all modules.

    !!! warning

        This adds global state to the `nn.Module` module

    The hook will be called every time :func:`register_parameter` is invoked.
    It should have the following signature::

        hook(module, name, param) -> None or new parameter

    The hook can modify the input or return a single modified value in the hook.

    Returns:
        msgflux.nn.utils.hooks.RemovableHandle: A handle that can be used to remove
        the added hook by calling ``handle.remove()``.
    """
    handle = RemovableHandle(_global_parameter_registration_hooks)
    _global_parameter_registration_hooks[handle.id] = hook
    return handle


def register_module_forward_pre_hook(hook: Callable[..., None]) -> RemovableHandle:
    """Register a forward pre-hook common to all modules.

    !!! warning

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time before :func:`forward` is invoked.
    It should have the following signature::

        hook(module, input) -> None or modified input

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the input. User can either return a tuple or a
    single modified value in the hook. We will wrap the value into a tuple
    if a single value is returned(unless that value is already a tuple).

    This hook has precedence over the specific module hooks registered with
    ``register_forward_pre_hook``.

    Returns:
        :class:`msgflux.nn.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = RemovableHandle(_global_forward_pre_hooks)
    _global_forward_pre_hooks[handle.id] = hook
    return handle


def register_module_forward_hook(
    hook: Callable[..., None],
    *,
    with_kwargs: bool = False,
    always_call: bool = False,
) -> RemovableHandle:
    """Register a global forward hook for all the modules.

    !!! warning

        This adds global state to the `nn.module` module
        and it is only intended for debugging/profiling purposes.

    The hook will be called every time after :func:`forward` has computed an output.
    It should have the following signature::

        hook(module, input, output) -> None or modified output

    The input contains only the positional arguments given to the module.
    Keyword arguments won't be passed to the hooks and only to the ``forward``.
    The hook can modify the output. It can modify the input inplace but
    it will not have effect on forward since this is called after
    :func:`forward` is called.

    Args:
        hook:
            The user defined hook to be registered.
        always_call:
            If ``True`` the ``hook`` will be run regardless of
            whether an exception is raised while calling the Module.
            Default: ``False``
    Returns:
        :class:`msgflux.nn.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``

    This hook will be executed before specific module hooks registered with
    ``register_forward_hook``.
    """
    handle = RemovableHandle(
        _global_forward_hooks, extra_dict=_global_forward_hooks_always_called
    )
    _global_forward_hooks[handle.id] = hook
    if with_kwargs:
        _global_forward_hooks_with_kwargs[handle.id] = True
    if always_call:
        _global_forward_hooks_always_called[handle.id] = True
    return handle


def _forward_unimplemented(self, *inputs: Any) -> None:
    """Define the computation performed at every call.

    Should be overridden by all subclasses.

    !!! note

        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError(
        f"Module [{type(self).__name__}] is missing the required `forward` function"
    )


async def _aforward_unimplemented(self, *inputs: Any) -> None:
    """Define the async computation performed at every acall.

    Should be overridden by all subclasses that want async support.

    !!! note

        Although the recipe for async forward pass needs to be defined within
        this function, one should call the :class:`Module` instance's acall
        method afterwards instead of this since the former takes care of running
        the registered hooks while the latter silently ignores them.
    """
    raise NotImplementedError(
        f"Module [{type(self).__name__}] is missing the required `aforward` function"
    )


class Module:
    training: bool
    _version: int = 1
    """This allows better BC support for :meth:`load_state_dict`. In
    :meth:`state_dict`, the version number will be saved as in the attribute
    `_metadata` of the returned state dict, and thus pickled. `_metadata` is a
    dictionary with keys that follow the naming convention of state dict. See
    ``_load_from_state_dict`` on how to use this information in loading.

    If new parameters/buffers are added/removed from a module, this number shall
    be bumped, and the module's `_load_from_state_dict` method can compare the
    version number and do appropriate changes if the state dict is from before
    the change."""
    _parameters: Dict[str, Optional[Parameter]] = OrderedDict()
    _buffers: Dict[str, Optional[Any]] = OrderedDict()
    _modules: Dict[str, Optional["Module"]] = OrderedDict()
    _forward_hooks: Dict[int, Callable]
    _forward_hooks_with_kwargs: Dict[int, bool]
    _forward_hooks_always_called: Dict[int, bool]
    _forward_pre_hooks: Dict[int, Callable]
    _forward_pre_hooks_with_kwargs: Dict[int, bool]
    _load_state_dict_post_hooks: Dict[int, Callable]
    _load_state_dict_pre_hooks: Dict[int, Callable]
    _state_dict_hooks: Dict[int, Callable]
    _state_dict_pre_hooks: Dict[int, Callable]
    call_super_init: bool = False

    def __init__(self, *args, **kwargs) -> None:
        # Backward compatibility: no args used to be allowed when call_super_init=False
        if self.call_super_init is False and bool(kwargs):
            raise TypeError(
                f"{type(self).__name__}.__init__() got an unexpected "
                f"keyword argument `{next(iter(kwargs))}`"
            )

        if self.call_super_init is False and bool(args):
            raise TypeError(
                f"{type(self).__name__}.__init__() takes 1 positional "
                f"argument but {len(args) + 1} were"
            )

        """
        Calls super().__setattr__('a', a) instead of the typical self.a = a
        to avoid Module.__setattr__ overhead. Module's __setattr__ has special
        handling for parameters, submodules, and buffers but simply calls into
        super().__setattr__ for all other attributes.
        """
        super().__setattr__("training", True)
        super().__setattr__("_parameters", {})
        super().__setattr__("_buffers", {})
        super().__setattr__("_non_persistent_buffers_set", set())  # ?
        super().__setattr__("_forward_pre_hooks", OrderedDict())
        super().__setattr__("_forward_hooks", OrderedDict())
        super().__setattr__("_forward_hooks_always_called", OrderedDict())
        super().__setattr__("_state_dict_hooks", OrderedDict())
        super().__setattr__("_state_dict_pre_hooks", OrderedDict())
        super().__setattr__("_load_state_dict_pre_hooks", OrderedDict())
        super().__setattr__("_load_state_dict_post_hooks", OrderedDict())
        super().__setattr__("_modules", {})
        super().__setattr__("_spans", spans)

        if self.call_super_init:
            super().__init__(*args, **kwargs)

    forward: Callable[..., Any] = _forward_unimplemented
    aforward: Callable[..., Any] = _aforward_unimplemented

    # msgflux funcs

    def _get_mermaid(
        self,
        title: Optional[str] = None,
        orientation: Optional[str] = "TD",
        *,
        remove_self: Optional[bool] = True,
    ) -> str:
        if code_to_mermaid is None:
            raise ImportError(
                "`mermaid` client is not available. "
                "Install with `pip install msgflux[plot]`."
            )
        mermaid = code_to_mermaid(
            inspect.getsource(self.forward),
            remove_self=remove_self,
            title=title,
            orientation=orientation,
        )
        return mermaid

    def plot(
        self,
        title: Optional[str] = None,
        orientation: Optional[str] = "TD",
        *,
        remove_self: Optional[bool] = True,
    ) -> "Mermaid":
        """Generates and renders a Mermaid diagram of the `forward` method.

        This method extracts the source code of the `forward` method and converts
        it into a Mermaid diagram for visualization. Optionally, it can clean up
        the code by removing references to `self` to produce a cleaner diagram.

        Args:
            title:
                Title to display at the top of the Mermaid diagram.
            orientation:
                Diagram orientation. Options include "TD" (top-down),
                "LR" (left-right), etc.
            remove_self:
                Whether to remove references to `self` from the code before
                generating the diagram. Useful for cleaner output.

        Returns:
            The rendered Mermaid diagram.
        """
        mermaid = self._get_mermaid(title, orientation, remove_self)
        return plot_mermaid(mermaid)

    def _get_content_from_or_input(self, path: str, message: Message) -> Any:
        """Returns the first valid content from OR input."""
        content = None
        for single_path in path:
            content = message.get(single_path)
            if content is not None:
                break
        return content

    def _get_content_from_message(self, path: str, message: Message):
        content = None
        if isinstance(message, Message):
            if isinstance(path, tuple): # OR inputs
                content = self._get_content_from_or_input(path, message)
            else:
                content = message.get(path)
        return content

    def _extract_message_values(
        self, paths: Union[str, List[str], Dict[str, str]], message: Message
    ) -> Optional[Union[str, Dict[str, Any], List[Any], None]]:
        """Process inputs based on their type (str, dict, list)
        by extracting content from the message.
        """
        if isinstance(paths, str):
            return self._get_content_from_message(paths, message)
        elif isinstance(paths, dict):
            return dotdict(
                {
                    key: self._get_content_from_message(path, message)
                    for key, path in paths.items()
                }
            )
        elif isinstance(paths, list):
            return [
                self._get_content_from_message(path, message)
                for path in paths
                if self._get_content_from_message(path, message) is not None
            ]
        return None

    def _prepare_data_uri(
        self, source: str, force_encode: Optional[bool] = False # noqa: FBT001, FBT002
    ) -> Optional[str]:
        """Prepares a data string (URL or Data URI base64).
        If force_encode=True, always tries to download and encode URL.
        Otherwise, keeps the URL if it is HTTP and not base64.
        Returns None in case of encoding/download error.
        """
        if not source:
            return None

        if is_base64(source):
            # If it is already base64, assume it is ready (no prefix)
            # Prefix will be added by formatter if needed
            return source

        is_url = source.startswith("http")

        if is_url and not force_encode:
            # Keep the URL as is if you don't force the encoding
            return source

        # Need to encode (either local or force_encode=True for URL)
        try:
            return encode_data_to_base64(source)
        except Exception as e:
            logger.error(f"Failed to encode source {source}: {e}")
            return None

    def _format_task_template(self, content: Union[str, Dict[str, Any]]) -> str:
        return self._format_template(content, self.task_template)

    def _format_response_template(self, content: str) -> str:
        return self._format_template(content, self.response_template)

    def _format_template(
        self, content: Union[str, Dict[str, Any]], raw_template: str
    ) -> str:
        if isinstance(content, str):
            rendered = raw_template.format(content)
        elif isinstance(content, dict):
            template = Template(raw_template)
            rendered = template.render(content)
        else:
            raise ValueError("Unsupported content type for template formatting")
        rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
        return rendered        

    def set_name(self, name: str):
        if isinstance(name, str):
            if name != "":
                # Validate snake_case pattern
                if not re.match(r"^[a-z][a-z0-9_]*$", name):
                    raise ValueError(
                        f"`name` must follow snake_case pattern (lowercase letters, "
                        f"numbers, and underscores only, starting with a letter). "
                        f"Got: '{name}'"
                    )
                self.register_buffer("name", name)
            else:
                raise ValueError("`name` requires a string not empty")
        else:
            raise TypeError(f"`name` need be a `str` given {type(name)}")

    def set_annotations(self, annotations: Dict[str, type]):
        if isinstance(annotations, dict):
            super().__setattr__("annotations", annotations)
        else:
            raise TypeError(f"`annotations` need be a `dict` given {type(annotations)}")

    def _set_task_template(self, task_template: Optional[str] = None):
        if isinstance(task_template, str) or task_template is None:
            if isinstance(task_template, str) and task_template == "":
                raise ValueError(
                    f"`task_template` requires a string not empty given {task_template}"
                )
            self.register_buffer("task_template", task_template)
        else:
            raise TypeError(
                "`task_template` requires a string or None"
                f"given `{type(task_template)}`"
            )

    def _set_response_template(self, response_template: Optional[str] = None):
        if isinstance(response_template, str) or response_template is None:
            if isinstance(response_template, str) and response_template == "":
                raise ValueError("`response_template` cannot be an empty str")
            self.register_buffer("response_template", response_template)
        else:
            raise TypeError(
                "`response_template` requires a string or None "
                f"given `{type(response_template)}`"
            )

    def _set_response_mode(self, response_mode: str):
        if isinstance(response_mode, str):
            if response_mode == "":
                raise ValueError("`response_mode` requires a not empty string")
            self.register_buffer("response_mode", response_mode)
        else:
            raise TypeError(
                f"`response_mode` requires a string given `{type(response_mode)}`"
            )

    def _set_prompt(self, prompt: Optional[str] = None):
        if isinstance(prompt, str) or prompt is None:
            self.register_buffer("prompt", prompt)
        else:
            raise TypeError(f"`prompt` need be a str or None given `{type(prompt)}`")

    def _set_stream(self, stream: bool): # noqa: FBT001
        if isinstance(stream, bool):
            self.register_buffer("stream", stream)
        else:
            raise TypeError(f"`stream` need be a bool given `{type(stream)}`")

    def _set_execution_kwargs(self, execution_kwargs: Optional[Dict[str, Any]] = None):
        if isinstance(execution_kwargs, dict) or execution_kwargs is None:
            if isinstance(execution_kwargs, dict):
                execution_kwargs = dotdict(execution_kwargs)
            self.register_buffer("execution_kwargs", execution_kwargs)
        else:
            raise TypeError(
                "`execution_kwargs` need be a dict or None "
                f"given `{type(execution_kwargs)}`"
            )

    def _set_verbose(self, verbose: bool): # noqa: FBT001
        if isinstance(verbose, bool):
            super().__setattr__("verbose", verbose)
        else:
            raise TypeError(f"`verbose` need be a `bool` given {type(verbose)}")

    def _extract_raw_response(
        self, model_response: Union[ModelResponse, ModelStreamResponse]
    ) -> Any:
        if isinstance(model_response, ModelResponse):
            return model_response.consume()
        elif isinstance(model_response, ModelStreamResponse):
            return model_response
        else:
            raise ValueError(f"Unsupported `model_response={type(model_response)}`")

    def _prepare_response(self, raw_response: Any, message: Any) -> Any:
        if not isinstance(raw_response, ModelStreamResponse) and (
            hasattr(self, "response_template") and self.response_template is not None
        ):
            response = self._format_response_template(raw_response)
        else:
            response = raw_response
        return self._define_response_mode(response, message)

    def _define_response_mode(self, response: Any, message: Any) -> Any:
        if self.response_mode == "plain_response":
            return response
        elif isinstance(message, Message):
            message.set(self.response_mode, response)
            return message
        else:
            raise ValueError(
                "For non-Message objects is required `response_mode=='plain_response'`"
            )

    def _set_task_inputs(
        self, task_inputs: Optional[Union[str, Dict[str, str], Tuple[str, ...]]] = None
    ):
        # TODO: suporte para lista de inputs ["outputs.text1", "outputs.text2"]
        if isinstance(task_inputs, (str, dict, tuple)) or task_inputs is None:
            if isinstance(task_inputs, str) and task_inputs == "":
                raise ValueError(
                    f"`task_inputs` requires a string not empty given `{task_inputs}`"
                )
            if isinstance(task_inputs, (dict, tuple)) and not task_inputs:
                raise ValueError(
                    "`task_inputs` requires a dict or tuple not empty "
                    f"given `{task_inputs}`"
                )
            self.register_buffer("task_inputs", task_inputs)
        else:
            raise TypeError(
                "`task_inputs` requires a string, dict or None, "
                f"given `{type(task_inputs)}`"
            )

    def _set_task_multimodal_inputs(
        self, task_multimodal_inputs: Optional[Dict[str, List[str]]] = None
    ):
        # TODO permitir passar em vez de uma lista passar so um valor se for unico
        if isinstance(task_multimodal_inputs, dict) or task_multimodal_inputs is None:
            if not task_multimodal_inputs and task_multimodal_inputs is not None:
                raise ValueError(
                    "`task_multimodal_inputs` requires a dict not empty"
                    f"given `{task_multimodal_inputs}`"
                )
            self.register_buffer("task_multimodal_inputs", task_multimodal_inputs)
        else:
            raise TypeError(
                "`task_multimodal_inputs` requires a dict "
                f"given `{type(task_multimodal_inputs)}`"
            )

    def _set_model_preference(self, model_preference: Optional[str] = None):
        if isinstance(model_preference, str) or model_preference is None:
            self.register_buffer("model_preference", model_preference)
        else:
            raise TypeError(
                "`model_preference` need be a string or None, "
                f"given `{type(model_preference)}`"
            )

    def _set_input_guardrail(self, input_guardrail: Optional[Callable] = None):
        if isinstance(input_guardrail, Callable) or input_guardrail is None:
            if (
                inspect.isclass(input_guardrail)
                and hasattr(input_guardrail, "serialize")
            ) or None:
                self.register_buffer("input_guardrail", input_guardrail)
            elif isinstance(input_guardrail, self.__class__):
                self.input_guardrail = input_guardrail
            else:
                super().__setattr__("input_guardrail", input_guardrail)
        else:
            raise TypeError(
                "`input_guardrail` need be a callable or None, "
                f"given `{type(input_guardrail)}`"
            )

    def _set_output_guardrail(self, output_guardrail: Optional[Callable] = None):
        if isinstance(output_guardrail, Callable) or output_guardrail is None:
            if (
                inspect.isclass(output_guardrail)
                and hasattr(output_guardrail, "serialize")
            ) or None:
                self.register_buffer("output_guardrail", output_guardrail)
            elif isinstance(output_guardrail, self.__class__):
                self.output_guardrail = output_guardrail
            else:
                super().__setattr__("output_guardrail", output_guardrail)
        else:
            raise TypeError(
                "`output_guardrail` need be a callable or None, "
                f"given `{type(output_guardrail)}`"
            )

    def _execute_input_guardrail(self, model_execution_params: Dict[str, Any]):
        guardrail_params = self._prepare_input_guardrail_execution(
            model_execution_params
        )
        guardrail_response = self.input_guardrail(**guardrail_params)

        if isinstance(guardrail_response, ModelResponse):
            guardrail_response = self._extract_raw_response(guardrail_response)

        if not guardrail_response["safe"]:
            raise UnsafeUserInputError()  # TODO

    async def _aexecute_input_guardrail(self, model_execution_params: Dict[str, Any]):
        guardrail_params = self._prepare_input_guardrail_execution(
            model_execution_params
        )

        # Check if guardrail has acall method or is a coroutine function
        if hasattr(self.input_guardrail, 'acall'):
            guardrail_response = await self.input_guardrail.acall(**guardrail_params)
        elif inspect.iscoroutinefunction(self.input_guardrail):
            guardrail_response = await self.input_guardrail(**guardrail_params)
        else:
            # Fallback to sync call
            guardrail_response = self.input_guardrail(**guardrail_params)

        if isinstance(guardrail_response, ModelResponse):
            guardrail_response = self._extract_raw_response(guardrail_response)

        if not guardrail_response["safe"]:
            raise UnsafeUserInputError()  # TODO

    def _execute_output_guardrail(self, model_response: Dict[str, Any]):
        guardrail_params = self._prepare_output_guardrail_execution(model_response)
        guardrail_response = self.output_guardrail(**guardrail_params)

        if isinstance(guardrail_response, ModelResponse):
            guardrail_response = self._extract_raw_response(guardrail_response)

        if not guardrail_response["safe"]:
            raise UnsafeModelResponseError()  # TODO

    async def _aexecute_output_guardrail(self, model_response: Dict[str, Any]):
        guardrail_params = self._prepare_output_guardrail_execution(model_response)

        # Check if guardrail has acall method or is a coroutine function
        if hasattr(self.output_guardrail, 'acall'):
            guardrail_response = await self.output_guardrail.acall(**guardrail_params)
        elif inspect.iscoroutinefunction(self.output_guardrail):
            guardrail_response = await self.output_guardrail(**guardrail_params)
        else:
            # Fallback to sync call
            guardrail_response = self.output_guardrail(**guardrail_params)

        if isinstance(guardrail_response, ModelResponse):
            guardrail_response = self._extract_raw_response(guardrail_response)

        if not guardrail_response["safe"]:
            raise UnsafeModelResponseError()  # TODO

    def get_model_preference_from_message(self, message: Message) -> Optional[str]:
        if isinstance(message, Message) and isinstance(self.model_preference, str):
            return message.get(self.model_preference)
        else:
            return None

    def set_description(self, description: Optional[str] = None):
        if isinstance(description, str) or description is None:
            self.register_buffer("description", description)
        else:
            raise ValueError("`description` requires a string not empty")

    def get_module_name(self):
        module_name = getattr(self, "name", None)
        if module_name is None:
            module_name = self._get_name()
        return module_name

    def get_module_description(self):
        module_description = getattr(self, "description", None)
        if module_description is None:
            module_description = self.__class__.__doc__
        return module_description

    def get_module_annotations(self):
        module_annotations = getattr(self, "annotations", None)
        if module_annotations is None:
            module_annotations = self.__class__.__annotations__
        return module_annotations

    # msgflux END

    def register_buffer(self, name: str, data: Any) -> None:
        # TODO: muito trabalho pra ajeitar a docstring
        # mudei de tensor para data
        """Add a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name:
                Name of the buffer. The buffer can be accessed
                from this module using the given name
            data:
                buffer to be registered.
        Example::
            >>> self.register_buffer("name", "agent")
        """
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(f"buffer name should be a string. Got {type(name)}")
        elif "." in name:
            raise KeyError("buffer name can't contain '.'")
        elif name == "":
            raise KeyError("buffer name can't be empty string")
        else:
            for hook in _global_buffer_registration_hooks.values():
                output = hook(self, name, data)
                if output is not None:
                    data = output

            self._buffers[name] = data
            self._non_persistent_buffers_set.discard(name)

    def register_parameter(self, name: str, param: Parameter) -> None:
        """Add a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if "_parameters" not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call"
            )

        elif not isinstance(name, str):
            raise TypeError(f"parameter name should be a string. Got {type(name)}")
        elif "." in name:
            raise KeyError("parameter name can't contain '.'")
        elif name == "":
            raise KeyError("parameter name can't be empty string")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")
        elif param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(
                f"cannot assign `{type(param)}` object to parameter `{name}` "
                "(msgflux.nn.Parameter required)"
            )
        else:
            for hook in _global_parameter_registration_hooks.values():
                output = hook(self, name, param)
                if output is not None:
                    param = output
            self._parameters[name] = param

    def add_module(self, name: str, module: "Module") -> None:
        """Add a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (str): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{type(module)} is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(f"module name should be a string. Got {type(name)}")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute `{name}` already exists")
        elif "." in name:
            raise KeyError(f"module name can't contain '.', got: {name}")
        elif name == "":
            raise KeyError("module name can't be empty string ")
        for hook in _global_module_registration_hooks.values():
            output = hook(self, name, module)
            if output is not None:
                module = output
        self._modules[name] = module

    def register_module(self, name: str, module: "Module") -> None:
        """Alias for :func:`add_module`."""
        self.add_module(name, module)

    def get_submodule(self, target: str) -> "Module":
        """Return the submodule given by ``target``
        if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded by the degree
        of module nesting in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            msgflux.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: Module = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            mod = getattr(mod, item)

            if not isinstance(mod, Module):
                raise AttributeError("`" + item + "` is not an nn.Module")

        return mod

    def set_submodule(self, target: str, module: "Module") -> None:
        """Set the submodule given by ``target`` if it exists, otherwise throw an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To overide the ``Conv2d`` with a new submodule ``Linear``, you
        would call
        ``set_submodule("net_b.net_c.conv", nn.Linear(33, 16))``.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)
            module: The module to set the submodule to.

        Raises:
            ValueError: If the target string is empty
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
        if target == "":
            raise ValueError("Cannot set the submodule without a target name!")

        atoms: List[str] = target.split(".")
        name = atoms.pop(-1)
        mod: Module = self

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(
                    mod._get_name() + " has no attribute `" + item + "`"
                )

            mod = getattr(mod, item)

            # Use isinstance instead of type here to also handle subclass of nn.Module
            if not isinstance(mod, Module):
                raise AttributeError("`" + item + "` is not an nn.Module")

        setattr(mod, name, module)

    def get_parameter(self, target: str) -> "Parameter":
        """Return the parameter given by ``target``if
        it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the Parameter
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            msgflux.nn.Parameter: The Parameter referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Parameter``
        """
        module_path, _, param_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + param_name + "`"
            )

        param: Parameter = getattr(mod, param_name)

        if not isinstance(param, Parameter):
            raise AttributeError("`" + param_name + "` is not an nn.Parameter")

        return param

    def get_buffer(self, target: str) -> Any:
        """Return the buffer given by ``target`` if it exists, otherwise throw an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the buffer
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns: TODO
            torch.Tensor: The buffer referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not a
                buffer
        """
        module_path, _, buffer_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(
                mod._get_name() + " has no attribute `" + buffer_name + "`"
            )

        buffer: Any = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer

    # revisar

    def register_forward_pre_hook(
        self,
        hook: Callable[
            [T, Tuple[Any, ...], Dict[str, Any]],
            Optional[Tuple[Any, Dict[str, Any]]],
        ],
        *,
        prepend: bool = False,
    ) -> RemovableHandle:
        r"""Register a forward pre-hook on the module.

        The hook will be called every time before :func:`forward` is invoked.


        If ``with_kwargs`` is false or not specified, the input contains only
        the positional arguments given to the module. Keyword arguments won't be
        passed to the hooks and only to the ``forward``. The hook can modify the
        input. User can either return a tuple or a single modified value in the
        hook. We will wrap the value into a tuple if a single value is returned
        (unless that value is already a tuple). The hook should have the
        following signature::

            hook(module, args) -> None or modified input

        If ``with_kwargs`` is true, the forward pre-hook will be passed the
        kwargs given to the forward function. And if the hook modifies the
        input, both the args and kwargs should be returned. The hook should have
        the following signature::

            hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If true, the provided ``hook`` will be fired before
                all existing ``forward_pre`` hooks on this
                :class:`msgflux.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward_pre`` hooks
                on this :class:`msgflux.nn.modules.Module`. Note that global
                ``forward_pre`` hooks registered with
                :func:`register_module_forward_pre_hook` will fire before all
                hooks registered by this method.
                Default: ``False``

        Returns:
            :class:`msgflux.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(self._forward_pre_hooks)
        self._forward_pre_hooks[handle.id] = hook
        if prepend:
            self._forward_pre_hooks.move_to_end(handle.id, last=False)
        return handle

    def register_forward_hook(
        self,
        hook: Union[
            Callable[[T, tuple[Any, ...], Any], Optional[Any]],
            Callable[[T, tuple[Any, ...], dict[str, Any], Any], Optional[Any]],
        ],
        *,
        prepend: bool = False,
        always_call: bool = False,
    ) -> RemovableHandle:
        """Register a forward hook on the module.

        The hook will be called every time after :func:`forward`
        has computed an output. The hook receives both positional
        arguments (`args`), keyword arguments (`kwargs`), and the
        output of the forward call. The hook can modify `args`,
        `kwargs`, and the output. It should have the following signature::

            hook(module, args, kwargs, output) -> None, modified_output,
            or (modified_args, modified_kwargs, modified_output)

        Args:
            hook (Callable): The user defined hook to be registered.
            prepend (bool): If ``True``, the provided ``hook`` will be fired
                before all existing ``forward`` hooks on this
                :class:`msgflux.nn.modules.Module`. Otherwise, the provided
                ``hook`` will be fired after all existing ``forward`` hooks on
                this :class:`msgflux.nn.modules.Module`. Note that global
                ``forward`` hooks registered with
                :func:`register_module_forward_hook` will fire before all hooks
                registered by this method.
                Default: ``False``
            always_call (bool): If ``True`` the ``hook`` will be run regardless of
                whether an exception is raised while calling the Module.
                Default: ``False``

        Returns:
            :class:`msgflux.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(
            self._forward_hooks,
            extra_dict=[self._forward_hooks_always_called],
        )
        self._forward_hooks[handle.id] = hook
        if always_call:
            self._forward_hooks_always_called[handle.id] = True
        if prepend:
            self._forward_hooks.move_to_end(handle.id, last=False)
        return handle

    def _call_impl(self, *args, **kwargs):
        if not (self._forward_hooks or self._forward_pre_hooks):
            return self._call(*args, **kwargs)

        for hook in self._forward_pre_hooks.values():
            hook_result = hook(self, args, kwargs)
            if hook_result is not None:
                if isinstance(hook_result, tuple) and len(hook_result) == 2:
                    args, kwargs = hook_result
                else:
                    raise RuntimeError(
                        "forward pre-hook must return None or a tuple of (args, kwargs)"
                    )

        result = self._call(*args, **kwargs)

        for hook in self._forward_hooks.values():
            hook_result = hook(self, args, kwargs, result)
            if hook_result is not None:
                result = hook_result

        return result

    def _call(self, *args, **kwargs):
        # Search for a Message in args and kwargs
        message = next(
            (arg for arg in args if isinstance(arg, Message)),
            next((v for v in kwargs.values() if isinstance(v, Message)), None),
        )

        if message is not None:
            # Check if this module has already processed the message
            # If it is True, skip the module
            if envs.state_checkpoint and message.in_msg(self.name):
                return message

        module_name = self.get_module_name()
        module_name_title = convert_camel_snake_to_title(module_name)

        encoded_state_dict = None
        if envs.telemetry_capture_state_dict:
            state_dict = self.state_dict()
            encoded_state_dict = msgspec.json.encode(state_dict)

        # Trace capture
        current_span = trace.get_current_span()
        # If there is no active span or it is not recording, this is the root module
        if current_span is None or not current_span.is_recording():
            with self._spans.init_flow(
                module_name_title, message, encoded_state_dict
            ):
                module_output = self.forward(*args, **kwargs)
        else:
            with self._spans.init_module(module_name_title):
                module_output = self.forward(*args, **kwargs)
        return module_output

    async def _acall_impl(self, *args, **kwargs):
        if not (self._forward_hooks or self._forward_pre_hooks):
            return await self._acall(*args, **kwargs)

        for hook in self._forward_pre_hooks.values():
            hook_result = hook(self, args, kwargs)
            if hook_result is not None:
                if isinstance(hook_result, tuple) and len(hook_result) == 2:
                    args, kwargs = hook_result
                else:
                    raise RuntimeError(
                        "forward pre-hook must return None or a tuple of (args, kwargs)"
                    )

        result = await self._acall(*args, **kwargs)

        for hook in self._forward_hooks.values():
            hook_result = hook(self, args, kwargs, result)
            if hook_result is not None:
                result = hook_result

        return result

    async def _acall(self, *args, **kwargs):
        # Search for a Message in args and kwargs
        message = next(
            (arg for arg in args if isinstance(arg, Message)),
            next((v for v in kwargs.values() if isinstance(v, Message)), None),
        )

        if message is not None:
            # Check if this module has already processed the message
            # If it is True, skip the module
            if envs.state_checkpoint and message.in_msg(self.name):
                return message

        module_name = self.get_module_name()
        module_name_title = convert_camel_snake_to_title(module_name)

        encoded_state_dict = None
        if envs.telemetry_capture_state_dict:
            state_dict = self.state_dict()
            encoded_state_dict = msgspec.json.encode(state_dict)

        # Trace capture
        current_span = trace.get_current_span()
        # If there is no active span or it is not recording, this is the root module
        if current_span is None or not current_span.is_recording():
            with self._spans.init_flow(
                module_name_title, message, encoded_state_dict
            ):
                module_output = await self.aforward(*args, **kwargs)
        else:
            with self._spans.init_module(module_name_title):
                module_output = await self.aforward(*args, **kwargs)
        return module_output

    __call__: Callable[..., Any] = _call_impl

    async def acall(self, *args, **kwargs):
        """Async interface using aforward.

        If aforward is not implemented, falls back to running __call__ in executor.
        """
        # Check if aforward is implemented by comparing with the unimplemented version
        if type(self).aforward is _aforward_unimplemented:
            # Fallback to executor for backward compatibility
            loop = asyncio.get_event_loop()
            executor = Executor.get_instance()
            return await loop.run_in_executor(executor, lambda: self.__call__(*args, **kwargs))
        else:
            # Use native async implementation
            return await self._acall_impl(*args, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: Dict[str, Any]):
        self.__dict__.update(state)

        # Support loading old checkpoints that don't have the following attrs:
        if "_forward_pre_hooks" not in self.__dict__:
            self._forward_pre_hooks = OrderedDict()
        if "_forward_hooks_always_called" not in self.__dict__:
            self._forward_hooks_always_called = OrderedDict()
        if "_state_dict_hooks" not in self.__dict__:
            self._state_dict_hooks = OrderedDict()
        if "_state_dict_pre_hooks" not in self.__dict__:
            self._state_dict_pre_hooks = OrderedDict()
        if "_load_state_dict_pre_hooks" not in self.__dict__:
            self._load_state_dict_pre_hooks = OrderedDict()
        if "_load_state_dict_post_hooks" not in self.__dict__:
            self._load_state_dict_post_hooks = OrderedDict()
        if "_non_persistent_buffers_set" not in self.__dict__:
            self._non_persistent_buffers_set = set()

    def __getattr__(self, name: str) -> Union[Any, "Module"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            f"`{type(self).__name__}` object has no attribute `{name}`"
        )

    def __setattr__(self, name: str, value: Union[Any, "Module"]) -> None: # noqa: C901
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get("_parameters")
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call"
                )
            remove_from(
                self.__dict__,
                self._buffers,
                self._modules,
                self._non_persistent_buffers_set,
            )
            self.register_parameter(name, value)
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(
                    f"cannot assign '{type(value)}' as parameter '{name}' "
                    "(msgflux.nn.Parameter or None expected)"
                )
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get("_modules")
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call"
                    )
                remove_from(
                    self.__dict__,
                    self._parameters,
                    self._buffers,
                    self._non_persistent_buffers_set,
                )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(
                        f"cannot assign `{type(value)}` as child module `{name}` "
                        "(msgflux.nn.Module or None expected)"
                    )
                for hook in _global_module_registration_hooks.values():
                    output = hook(self, name, value)
                    if output is not None:
                        value = output
                modules[name] = value
            else:
                super().__setattr__(name, value)

    def __delattr__(self, name: str):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def _register_state_dict_hook(self, hook: Callable):
        """Register a post-hook for the :meth:`~msgflux.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata) -> None or state_dict

        The registered hooks can modify the ``state_dict`` inplace or return a new one.
        If a new ``state_dict`` is returned, it will only be respected if it is the root
        module that :meth:`~nn.Module.state_dict` is called from.
        """
        if getattr(hook, "_from_public_api", False):
            raise RuntimeError(
                "Cannot register the same function as the state dict post"
                "hook that was previously registered via "
                "register_state_dict_post_hook"
            )
        handle = RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def register_state_dict_post_hook(self, hook: Callable):
        """Register a post-hook for the :meth:`~msgflux.nn.Module.state_dict` method.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata) -> None

        The registered hooks can modify the ``state_dict`` inplace.
        """
        hook._from_public_api = True
        handle = RemovableHandle(self._state_dict_hooks)
        self._state_dict_hooks[handle.id] = hook
        return handle

    def register_state_dict_pre_hook(self, hook: Callable):
        """Register a pre-hook for the :meth:`~msgflux.nn.Module.state_dict`
        method.

        It should have the following signature::
            hook(module, prefix, keep_vars) -> None

        The registered hooks can be used to perform pre-processing
        before the ``state_dict`` call is made.
        """
        handle = RemovableHandle(self._state_dict_pre_hooks)
        self._state_dict_pre_hooks[handle.id] = hook
        return handle

    def _get_serializable_value(self, obj: object):
        """Get serializable value from an object."""
        if is_builtin_type(obj):
            return obj
        elif is_subclass_of(obj, msgspec.Struct):
            return msgspec.json.schema(obj)
        elif hasattr(obj, "serialize"):
            return obj.serialize()
        else:
            return None  # Fallback

    def _save_to_state_dict(self, destination: str, prefix: str):
        """Save parameters and buffers to state dict."""
        # Save parameters (only the data string)
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.data

        # Save buffers (handle different data types)
        for name, buf in self._buffers.items():
            destination[prefix + name] = self._get_serializable_value(buf)

    def state_dict(
        self, destination: Optional[Dict[str, Any]] = None, prefix: Optional[str] = ""
    ):
        """Returns a dictionary containing module's state.

        Args:
            destination:
                If provided, the state will be updated into
                the given dict. Default: None
            prefix:
                Prefix added to parameter and buffer names.
                Default: ""
        """
        if destination is None:
            destination = {}

        # Save current module's state
        self._save_to_state_dict(destination, prefix)

        # Save states from child modules
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + ".")

        return destination

    def _register_load_state_dict_pre_hook(
        self, hook: Callable, *, with_module: Optional[bool] = False
    ):
        """See :meth:`~msgflux.nn.Module.register_load_state_dict_pre_hook` for details.

        A subtle difference is that if ``with_module`` is set to ``False``, then the
        hook will not take the ``module`` as the first argument whereas
        :meth:`~msgflux.nn.Module.register_load_state_dict_pre_hook` always takes the
        ``module`` as the first argument.

        Args:
            hook (Callable): Callable hook that will be invoked before
                loading the state dict.
            with_module (bool, optional): Whether or not to pass the module
                instance to the hook as the first parameter.
        """
        handle = RemovableHandle(self._load_state_dict_pre_hooks)
        self._load_state_dict_pre_hooks[handle.id] = _WrappedHook(
            hook, self if with_module else None
        )
        return handle

    def register_load_state_dict_pre_hook(self, hook: Callable):
        """Register a pre-hook to be run before module's
        :meth:`~nn.Module.load_state_dict` is called.

        It should have the following signature::
            hook(module, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

        Args:
            hook:
                Callable hook that will be invoked before
                loading the state dict.
        """
        return self._register_load_state_dict_pre_hook(hook, with_module=True)

    def register_load_state_dict_post_hook(self, hook: Callable):
        """Register a post-hook to be run after module's
        :meth:`~nn.Module.load_state_dict` is called.

        It should have the following signature::
            hook(module, incompatible_keys) -> None

        The ``module`` argument is the current module that this hook is registered
        on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
        of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
        is a ``list`` of ``str`` containing the missing keys and
        ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

        The given incompatible_keys can be modified inplace if needed.

        Note that the checks performed when calling :func:`load_state_dict` with
        ``strict=True`` are affected by modifications the hook makes to
        ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
        set of keys will result in an error being thrown when ``strict=True``, and
        clearing out both missing and unexpected keys will avoid an error.

        Returns:
            :class:`msgflux.utils.hooks.RemovableHandle`:
                a handle that can be used to remove the added hook by calling
                ``handle.remove()``
        """
        handle = RemovableHandle(self._load_state_dict_post_hooks)
        self._load_state_dict_post_hooks[handle.id] = hook
        return handle

    def _load_from_state_dict( # noqa: C901
        self, state_dict: Dict[str, Any], prefix: Optional[str] = ""
    ) -> None:
        """Loads the module state from a state dict.

        Args:
            state_dict: Dictionary containing the state
            prefix: Prefix used for parameter/buffer names
        """
        # Load parameters
        for name, param in self._parameters.items():
            if param is not None:
                key = prefix + name
                if key in state_dict:
                    self._parameters[name].copy_to_data(state_dict[key])

        # Load buffers
        for name, _ in self._buffers.items():
            key = prefix + name
            if key in state_dict:
                data = state_dict[key]
                # Check if it is a msgflux serializable class
                if isinstance(data, dict) and "msgflux_type" in data:
                    msgflux_type = data.pop("msgflux_type")
                    if msgflux_type in MSGFLUX_DESERIALIZABLE_CLS:
                        # TODO not recreate if same type
                        cls = MSGFLUX_DESERIALIZABLE_CLS[msgflux_type]
                        instance = cls.from_serialized(**data)
                        self._buffers[name] = instance
                    elif msgflux_type == "generation_schema":
                        state = data.pop("state")
                        generation_schema = StructFactory.from_schema(state)
                        self._buffers[name] = generation_schema
                else:  # Otherwise, load the value directly
                    self._buffers[name] = data

        # Load submodules recursively
        for name, module in self._modules.items():
            if module is not None:
                module_prefix = prefix + name + "."
                module_dict = {
                    k.replace(module_prefix, ""): v
                    for k, v in state_dict.items()
                    if k.startswith(module_prefix)
                }
                if module_dict:
                    module._load_from_state_dict(module_dict)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Loads the state of the module and its submodules.

        Args:
            state_dict: Dictionary containing the complete state
        """
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"`state_dict` to be dict, given {type(state_dict).__name__}"
            )

        self._load_from_state_dict(state_dict)

    def _named_members(  # ok?
        self,
        get_members_fn,
        prefix: Optional[str] = "",
        *,
        recurse: Optional[bool] = True,
        remove_duplicate: Optional[bool] = True,
    ):
        """Help yield various names + members of modules."""
        memo = set()
        modules = (
            self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
            if recurse
            else [(prefix, self)]
        )
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                if remove_duplicate:
                    memo.add(v)
                name = module_prefix + ("." if module_prefix else "") + k
                yield name, v

    def parameters(self, *, recurse: Optional[bool] = True) -> Iterator[Parameter]:
        """Return an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse: If True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Returns:
            Parameter: module parameter

        !!! example
            # TODO
            ```python
            for param in model.parameters():
                print(type(param), param.size())
            >>> <class 'torch.Tensor'> (20L,)
            >>> <class 'torch.Tensor'> (20L, 1L, 5L, 5L)
            ```
        """
        for _name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self, prefix: str = "", *, recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        # TODO: docstring
        """Return an iterator over module parameters, yielding both the name of the
            parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                parameters in the result. Defaults to True.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, param in self.named_parameters():
            >>>     if name in ['bias']:
            >>>         print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def buffers(self, *, recurse: Optional[bool] = True) -> Iterator[Any]:  # TODO doc
        """Return an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Returns:
            torch.Tensor: module buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(
        self,
        prefix: Optional[str] = "",
        *,
        recurse: Optional[bool] = True,
        remove_duplicate: Optional[bool] = True
    ) -> Iterator[Tuple[str, Any]]:  # TODO docstring
        """Return an iterator over module buffers, yielding both the name of the
            buffer as well as the buffer itself.

        Args:
            prefix:
                Prefix to prepend to all buffer names.
            recurse:
                If True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module. Defaults to True.
            remove_duplicat:
                Whether to remove the duplicated buffers in the result.
                Defaults to True.

        Yields:
            Tuple containing the name and buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, buf in self.named_buffers():
            >>>     if name in ['running_var']:
            >>>         print(buf.size())

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )
        yield from gen

    def children(self) -> Iterator["Module"]:
        """Return an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for _, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        """Return an iterator over immediate children modules, yielding
            both the name of the module as well as the module itself.

        Yields:
            (str, Module): Tuple containing a name and child module

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator["Module"]:  # TODO DOC
        """Return an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
            ...     print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: Optional[str] = "",
        *,
        remove_duplicate: Optional[bool] = True,
    ) -> Iterator[Tuple[str, "Module"]]:
        """Return an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo:
                A memo to store the set of modules already added to the result.
            prefix:
                A prefix that will be added to the name of the module.
            remove_duplicate:
                Whether to remove the duplicated module instances in the result
                or not.

        Yields:
            Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.
        """
        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ("." if prefix else "") + name
                yield from module.named_modules(
                    memo, submodule_prefix, remove_duplicate=remove_duplicate
                )

    def train(self: T, *, mode: Optional[bool] = True) -> T:
        """Set the module in training mode.

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode.

        Args:
            mode:
                Whether to set training mode (``True``) or evaluation
                mode (``False``). Default: ``True``.

        Returns:
            Self.
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode=mode)
        return self

    def eval(self: T) -> T:
        """Set the module in evaluation mode.

        This has an effect only on certain modules. See the documentation of
        particular modules for details of their behaviors in training/evaluation
        mode, i.e. whether they are affected.

        This is equivalent with :meth:`self.train(False) <msgflux.nn.Module.train>`.

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.eval()` and several similar mechanisms that may be confused with it.

        Returns:
            Self.
        """
        return self.train(mode=False)

    def requires_grad_(self: T, *, requires_pgrad: Optional[bool] = True) -> T:
        """Change if autograd should record operations on parameters in this module.

        This method sets the parameters' :attr:`requires_grad` attributes
        in-place.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually (e.g., GAN training).

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.requires_grad_()` and several similar mechanisms that may be confused with it.

        Args:
            requires_grad:
                Whether autograd should record operations on
                parameters in this module.

        Returns:
            Module: self
        """
        for p in self.parameters():
            p.requires_grad_(requires_grad=requires_pgrad)
        return self

    def zero_pgrad(
        self, *, set_to_none: Optional[bool] = True
    ) -> None:  # TODO isso é interessante mas vai mudar
        """Reset gradients of all model parameters.

        See similar function under :class:`msgflux.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`msgflux.optim.Optimizer.zero_grad` for details.
        """
        for p in self.parameters():
            if p.pgrad is not None:
                if set_to_none:
                    p.pgrad = None
                else:  # TODO revisar abaixo
                    if p.pgrad.grad_fn is not None:
                        p.pgrad.detach_()
                    else:
                        p.pgrad.requires_grad_(False)
                    p.pgrad.zero_()

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        """Return the extra representation of the module.

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ""

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)
