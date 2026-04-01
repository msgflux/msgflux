import asyncio
import inspect
from typing import Any, Callable, Dict, Optional

from msgflux.exceptions import (
    UnsafeModelResponseError,
    UnsafeUserInputError,
    _GuardInterrupt,
)
from msgflux.models.response import ModelResponse
from msgflux.nn.hooks.hook import Hook

_DEFAULT_GUARD_TARGET = object()


class Guard(Hook):
    """Validates input/output of a Module via the hook system.

    The validator receives ``data`` as a positional argument and must return
    either a dict with ``"safe"`` (bool) or a ``ModelResponse`` (auto-consumed).

    Args:
        validator: Callable that receives ``data`` and returns
            ``{"safe": bool}`` or a ``ModelResponse``.
        on: ``"pre"`` (before forward) or ``"post"`` (after forward).
        message: Response to return when ``safe=False``. If ``None``,
            an exception is raised instead.
        target: Submodule attribute name to register the hook on.
            Defaults to ``"generator"`` for forward hooks and ``None`` for
            method hooks.
        method: Optional method name to register the guard on. ``None``
            targets the module execution boundary (`forward`).
        include_data: If ``True``, the data that triggered the guard is
            attached to the raised exception via ``exc.data``. Defaults
            to ``False`` for security (the data may contain unsafe content).
    """

    def __init__(
        self,
        validator: Callable[..., Dict[str, Any]],
        *,
        on: str,
        message: Optional[str] = None,
        target: Any = _DEFAULT_GUARD_TARGET,
        method: Optional[str] = None,
        include_data: bool = False,
    ):
        if not callable(validator):
            raise TypeError(f"`validator` must be callable, given `{type(validator)}`")
        resolved_target = target
        if target is _DEFAULT_GUARD_TARGET:
            resolved_target = None if method is not None else "generator"

        super().__init__(on=on, target=resolved_target, method=method)
        self.validator = validator
        self.message = message
        self.include_data = include_data
        self.processor: Optional[Callable[..., Any]] = None
        self._has_async_validator = inspect.iscoroutinefunction(validator) or hasattr(
            validator, "acall"
        )

    def __call__(
        self,
        module: Any,  # noqa: ARG002
        args: tuple,  # noqa: ARG002
        kwargs: dict,
        output: Any = None,
    ) -> None:
        """Sync validation — called by ``_call_impl``."""
        data = self._apply_processor(kwargs if self.on == "pre" else output)
        result = self.validator(data)
        self._check_result(result, data)

    async def acall(
        self,
        module: Any,  # noqa: ARG002
        args: tuple,  # noqa: ARG002
        kwargs: dict,
        output: Any = None,
    ) -> None:
        """Async validation — called by ``_acall_impl``."""
        data = self._apply_processor(kwargs if self.on == "pre" else output)
        if self._has_async_validator:
            result = await self._acall_validator(data)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.validator(data))
        self._check_result(result, data)

    @property
    def processor_key(self) -> Optional[str]:
        """Key used to match processors in ``_set_hooks``."""
        if self.method is not None:
            return None
        return f"guard_{self.on}"

    def _check_result(self, result: Any, data: Any = None) -> None:
        if isinstance(result, ModelResponse):
            result = result.consume()
        if not result.get("safe", True):
            exc_data = data if self.include_data else None
            if self.message is not None:
                raise _GuardInterrupt(self.message)
            if self.on == "pre":
                raise UnsafeUserInputError(data=exc_data)
            raise UnsafeModelResponseError(data=exc_data)

    def _apply_processor(self, data: Any) -> Any:
        if self.processor is not None:
            return self.processor(data)
        return data

    async def _acall_validator(self, data: Any) -> Any:
        if hasattr(self.validator, "acall"):
            return await self.validator.acall(data)
        elif inspect.iscoroutinefunction(self.validator):
            return await self.validator(data)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.validator(data))
