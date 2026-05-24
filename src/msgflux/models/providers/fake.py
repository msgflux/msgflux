from typing import Any

from msgflux.models.base import BaseModel
from msgflux.models.registry import register_model
from msgflux.models.types import ChatCompletionModel


class FakeModelExecutionError(RuntimeError):
    """Raised when a fake model is executed."""


@register_model
class FakeChatCompletion(BaseModel, ChatCompletionModel):
    provider = "fake"

    def __init__(self, model_id: str = "placeholder", **kwargs: Any) -> None:
        self.model_id = model_id
        self.kwargs = dict(kwargs)
        self.client = None
        self.aclient = None
        self.model = None
        self.processor = None

    def _initialize(self) -> None:
        self.client = None
        self.aclient = None

    def __call__(self, *_args: Any, **_kwargs: Any) -> None:
        raise FakeModelExecutionError(
            "FakeChatCompletion cannot execute requests. Replace it with a real "
            "chat_completion model before running this module."
        )

    async def acall(self, *_args: Any, **_kwargs: Any) -> None:
        self.__call__()
