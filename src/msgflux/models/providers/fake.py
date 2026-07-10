from typing import Any

from msgflux.models.base import BaseModel
from msgflux.models.registry import register_model
from msgflux.models.types import (
    ChatCompletionModel,
    ImageEmbedderModel,
    ImageTextToImageModel,
    ModerationModel,
    SpeechToTextModel,
    TextEmbedderModel,
    TextRerankerModel,
    TextToImageModel,
    TextToSpeechModel,
)


class FakeModelExecutionError(RuntimeError):
    """Raised when a fake model is executed."""


class _BaseFakeModel(BaseModel):
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
            f"{self.__class__.__name__} cannot execute requests. Replace it with a "
            f"real {self.model_type} model before running this module."
        )

    async def acall(self, *_args: Any, **_kwargs: Any) -> None:
        self.__call__()


@register_model
class FakeChatCompletion(_BaseFakeModel, ChatCompletionModel):
    pass


@register_model
class FakeImageEmbedder(_BaseFakeModel, ImageEmbedderModel):
    pass


@register_model
class FakeImageTextToImage(_BaseFakeModel, ImageTextToImageModel):
    pass


@register_model
class FakeModeration(_BaseFakeModel, ModerationModel):
    pass


@register_model
class FakeSpeechToText(_BaseFakeModel, SpeechToTextModel):
    pass


@register_model
class FakeTextEmbedder(_BaseFakeModel, TextEmbedderModel):
    pass


@register_model
class FakeTextReranker(_BaseFakeModel, TextRerankerModel):
    pass


@register_model
class FakeTextToImage(_BaseFakeModel, TextToImageModel):
    pass


@register_model
class FakeTextToSpeech(_BaseFakeModel, TextToSpeechModel):
    pass
