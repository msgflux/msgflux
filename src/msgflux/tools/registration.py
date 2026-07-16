from typing import Callable

from msgflux.logger import logger


class ToolRegistrationTransaction:
    """Undo journal shared by one complete tool registration tree."""

    def __init__(self) -> None:
        self._undo: list[Callable[[], None]] = []
        self.reconcile_background = False

    def record(self, callback: Callable[[], None]) -> None:
        self._undo.append(callback)

    def rollback(self) -> None:
        rollback_errors = []
        for callback in reversed(self._undo):
            try:
                callback()
            except Exception as exc:  # pragma: no cover - defensive containment
                rollback_errors.append(exc)
        self._undo.clear()
        if rollback_errors:
            details = "; ".join(str(exc) for exc in rollback_errors)
            logger.error(f"Tool registration rollback errors: {details}")
