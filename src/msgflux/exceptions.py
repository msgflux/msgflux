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


class ModelProviderHTTPError(RuntimeError):
    """Structured HTTP failure returned by a model provider."""

    def __init__(
        self,
        *,
        status_code: int,
        description: str,
        provider: Optional[str] = None,
        model_id: Optional[str] = None,
        code: Any = None,
        error_type: Optional[str] = None,
        param: Optional[str] = None,
        request_id: Optional[str] = None,
        response: Any = None,
    ):
        self.status_code = status_code
        self.description = description
        self.provider = provider
        self.model_id = model_id
        self.code = code
        self.error_type = error_type
        self.param = param
        self.request_id = request_id
        self.response = response

        source = provider or "Model provider"
        message = f"{source} request failed with HTTP {status_code}: {description}"
        details = []
        if code is not None:
            details.append(f"code={code}")
        if error_type:
            details.append(f"type={error_type}")
        if param:
            details.append(f"param={param}")
        if request_id:
            details.append(f"request_id={request_id}")
        if details:
            message = f"{message} ({', '.join(details)})"
        super().__init__(message)


class TaskInterruptRequestedError(Exception):
    """Raised when a cooperative background task receives an interrupt request."""

    def __init__(self, task_id: str, message: Optional[str] = None):
        self.task_id = task_id
        super().__init__(message or f"Task `{task_id}` interrupt requested.")


class TaskPauseRequestedError(Exception):
    """Raised when a cooperative task or agent receives a pause request."""

    def __init__(self, task_id: Optional[str] = None, message: Optional[str] = None):
        self.task_id = task_id
        label = f"`{task_id}`" if task_id else "execution"
        super().__init__(message or f"Task {label} pause requested.")


class TaskIdCollisionError(Exception):
    """Raised when a task store receives an id already assigned to another task."""

    def __init__(self, task_id: str):
        super().__init__(f"Task id `{task_id}` already exists.")
        self.task_id = task_id


class UnsafeUserInputError(Exception):
    def __init__(self, message: Optional[str] = None, data: Any = None):
        super().__init__(message or "Unsafe user input detected")
        self.data = data


class UnsafeModelResponseError(Exception):
    def __init__(self, message: Optional[str] = None, data: Any = None):
        super().__init__(message or "Unsafe model response detected")
        self.data = data


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
