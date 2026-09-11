import asyncio
import functools
import inspect
import weakref
from collections import OrderedDict
from typing import Any, Callable, Literal, Optional, Tuple

__all__ = ["Hook", "RemovableHandle"]


class RemovableHandle:
    """A handle which provides the capability to remove a hook."""

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        """Args:
        hooks_dict:
            A dictionary of hooks, indexed by hook `id`.
        extra_dict:
            An additional dictionary or list of dictionaries whose keys
            will be deleted when the same keys are removed from `hooks_dict`.
        """
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        self.extra_dict_ref: Tuple = ()
        if isinstance(extra_dict, dict):
            self.extra_dict_ref = (weakref.ref(extra_dict),)
        elif isinstance(extra_dict, list):
            self.extra_dict_ref = tuple(weakref.ref(d) for d in extra_dict)

    def remove(self) -> None:
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        for ref in self.extra_dict_ref:
            extra_dict = ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]

    def __getstate__(self):
        if self.extra_dict_ref is None:
            return (self.hooks_dict_ref(), self.id)
        else:
            return (
                self.hooks_dict_ref(),
                self.id,
                tuple(ref() for ref in self.extra_dict_ref),
            )

    def __setstate__(self, state) -> None:
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

        if len(state) < 3 or state[2] is None:
            self.extra_dict_ref = ()
        else:
            self.extra_dict_ref = tuple(weakref.ref(d) for d in state[2])

    def __enter__(self) -> "RemovableHandle":
        return self

    def __exit__(self, dtype: Any, value: Any, tb: Any) -> None:
        self.remove()


class Hook:
    """Base class for declarative hooks registrable via ``hooks`` param.

    Subclasses must implement ``__call__`` (sync) and optionally
    override ``acall`` (async). By default ``acall`` runs ``__call__``
    in an executor.

    A hook uses exactly one registration style:

    - ``event=...`` registers a stable lifecycle hook.
    - ``on=...`` keeps the lower-level forward/method hook behavior.

    Args:
        event: Stable lifecycle event name. Lifecycle handlers receive the
            event payload and may return a replacement payload.
        handler: Optional callable used by lifecycle hooks. Subclasses may
            instead override :meth:`handle` and :meth:`ahandle`.
        on: ``"pre"`` (before execution) or ``"post"`` (after execution).
        target: Submodule attribute name to register the hook on.
            ``None`` registers on the module itself.
        method: Optional method name to register on. ``None`` targets the
            module execution boundary (`forward`).
    """

    _VALID_ON = {"pre", "post"}

    def __init__(
        self,
        *,
        event: Optional[str] = None,
        handler: Optional[Callable[[Any], Any]] = None,
        on: Optional[Literal["pre", "post"]] = None,
        target: Optional[str] = None,
        method: Optional[str] = None,
    ):
        if event is not None:
            if not isinstance(event, str) or not event.strip():
                raise ValueError("`event` must be a non-empty string")
            if on is not None or method is not None:
                raise ValueError("`event` cannot be combined with `on` or `method`")
            if handler is not None and not callable(handler):
                raise TypeError(f"`handler` must be callable, given `{type(handler)}`")
        elif on not in self._VALID_ON:
            raise ValueError(f"`on` must be one of {self._VALID_ON}, given `{on!r}`")
        elif handler is not None:
            raise ValueError("`handler` is only supported with lifecycle hooks")

        self.event = event
        self.handler = handler
        self.on = on
        self.target = target
        self.method = method

    @property
    def is_lifecycle(self) -> bool:
        """Whether this hook targets a stable lifecycle event."""
        return self.event is not None

    def __call__(self, module: Any, args: tuple, kwargs: dict, output: Any = None):
        """Sync hook — called by ``_call_impl``. Subclasses must override."""
        raise NotImplementedError

    async def acall(self, module: Any, args: tuple, kwargs: dict, output: Any = None):
        """Async hook — called by ``_acall_impl``.

        Default implementation runs ``__call__`` in an executor.
        """
        if self.on == "pre":
            return await asyncio.to_thread(
                functools.partial(self, module, args, kwargs)
            )
        return await asyncio.to_thread(
            functools.partial(self, module, args, kwargs, output)
        )

    def handle(self, payload: Any) -> Any:
        """Handle a lifecycle event synchronously.

        Subclasses may override this method. When ``handler`` was supplied to
        the constructor it is used directly.
        """
        if not self.is_lifecycle:
            raise RuntimeError("`handle` is only available for lifecycle hooks")
        if self.handler is None:
            raise NotImplementedError
        result = self.handler(payload)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            raise TypeError(
                "Lifecycle handler returned an awaitable during synchronous "
                "execution; use `ahandle`/`acall` instead"
            )
        return result

    async def ahandle(self, payload: Any) -> Any:
        """Handle a lifecycle event asynchronously."""
        if not self.is_lifecycle:
            raise RuntimeError("`ahandle` is only available for lifecycle hooks")

        if self.handler is not None:
            if hasattr(self.handler, "acall"):
                return await self.handler.acall(payload)
            if inspect.iscoroutinefunction(self.handler):
                return await self.handler(payload)

        if type(self).handle is not Hook.handle:
            return await asyncio.to_thread(self.handle, payload)

        if self.handler is None:
            raise NotImplementedError
        result = await asyncio.to_thread(self.handle, payload)
        if inspect.isawaitable(result):
            return await result
        return result

    def register(self, module: Any) -> "RemovableHandle":
        """Register this hook on *module*."""
        if self.is_lifecycle:
            return module.register_lifecycle_hook(self.event, self)
        if self.method is not None:
            if self.on == "pre":
                return module.register_method_pre_hook(self.method, self)
            return module.register_method_hook(self.method, self)
        if self.on == "pre":
            return module.register_forward_pre_hook(self)
        return module.register_forward_hook(self)

    @property
    def processor_key(self) -> Optional[str]:
        """Key used to match processors in ``_set_hooks``. ``None`` = no processor."""
        return None
