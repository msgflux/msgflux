import asyncio
import functools
import weakref
from collections import OrderedDict
from typing import Any, Optional, Tuple

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

    Args:
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
        on: str,
        target: Optional[str] = None,
        method: Optional[str] = None,
    ):
        if on not in self._VALID_ON:
            raise ValueError(f"`on` must be one of {self._VALID_ON}, given `{on!r}`")
        self.on = on
        self.target = target
        self.method = method

    def __call__(self, module: Any, args: tuple, kwargs: dict, output: Any = None):
        """Sync hook — called by ``_call_impl``. Subclasses must override."""
        raise NotImplementedError

    async def acall(self, module: Any, args: tuple, kwargs: dict, output: Any = None):
        """Async hook — called by ``_acall_impl``.

        Default implementation runs ``__call__`` in an executor.
        """
        loop = asyncio.get_event_loop()
        if self.on == "pre":
            return await loop.run_in_executor(
                None, functools.partial(self, module, args, kwargs)
            )
        return await loop.run_in_executor(
            None, functools.partial(self, module, args, kwargs, output)
        )

    def register(self, module: Any) -> "RemovableHandle":
        """Register this hook on *module*."""
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
