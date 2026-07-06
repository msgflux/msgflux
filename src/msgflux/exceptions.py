from typing import Any, List, Optional, Tuple


class ToolCallTimeOutError(Exception):
    pass


class ModelRouterError(Exception):
    def __init__(
        self,
        exceptions: List[Exception],
        model_info: List[Tuple[str, str, Exception]],
        message: Optional[str] = "Model routing failed",
    ):
        self.exceptions = exceptions
        self.model_info = model_info
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        details = "\n".join(
            [
                f"  - Model: {m_id} ({prov}): {type(exc).__name__}: {exc}"
                for m_id, prov, exc in self.model_info
            ]
        )
        return f"{self.message}\nCaptured Exceptions:\n{details}"


class _GuardInterrupt(Exception):  # noqa: N818
    """Internal exception for short-circuit when message is provided."""

    def __init__(self, response: str):
        self.response = response


class AbortRequestedError(Exception):
    """Raised when a local runtime abort signal is observed."""

    def __init__(self, message: Optional[str] = None):
        super().__init__(message or "Execution abort requested.")


class UnsafeUserInputError(Exception):
    def __init__(self, message: Optional[str] = None, data: Any = None):
        super().__init__(message or "Unsafe user input detected")
        self.data = data


class UnsafeModelResponseError(Exception):
    def __init__(self, message: Optional[str] = None, data: Any = None):
        super().__init__(message or "Unsafe model response detected")
        self.data = data


class TypedParserNotFoundError(ValueError):
    """Raised when a requested typed parser is not registered."""


class TaskError:
    """Wraps an exception from a failed task in a gather/scatter operation.

    Returned in place of ``None`` when a callable fails during parallel
    execution (e.g. ``scatter_gather``, ``map_gather``, ``bcast_gather``).
    This allows callers to inspect failures without losing successful results.

    Attributes:
        exception: The original exception raised by the task.
        index: The positional index of the failed task in the callable list.

    Examples:
        results = F.scatter_gather([agent_a, agent_b])
        for result in results:
            if isinstance(result, TaskError):
                print(f"Failed: {result.exception}")
            else:
                print(f"Success: {result}")
    """

    def __init__(self, exception: Exception, index: int):
        self.exception = exception
        self.index = index

    def __repr__(self) -> str:
        return f"TaskError(index={self.index}, exception={self.exception!r})"

    def __str__(self) -> str:
        return f"Task {self.index} failed: {self.exception}"

    def __bool__(self) -> bool:
        return False
