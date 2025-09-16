from typing import List, Optional, Tuple


# Temporary
class KeyExhaustedError(Exception):
    """Exception raised when all API keys have been tried and failed."""

    pass


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


class UnsafeUserInputError(Exception):
    pass


class UnsafeModelResponseError(Exception):
    pass


class TypedParserNotFoundError(ValueError):
    """Raised when a requested typed parser is not registered."""
