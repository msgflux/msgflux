from typing import Any, Mapping, Type

from msgflux.models.base import BaseModel
from msgflux.models.registry import model_registry
from msgflux.models.types import (
    ChatCompletionModel,
    ImageClassifierModel,
    ImageEmbedderModel,
    ImageTextToImageModel,
    ModerationModel,
    SpeechToTextModel,
    TextClassifierModel,
    TextEmbedderModel,
    TextRerankerModel,
    TextToImageModel,
    TextToSpeechModel,
)


class Model:

    @classmethod
    def providers(cls):
        return {k: list(v.keys()) for k, v in model_registry.items()}

    @classmethod
    def model_types(cls):
        return list(model_registry.keys())

    @classmethod
    def _model_path_parser(cls, model_id: str) -> tuple[str, str]:
        provider, model_id = model_id.split("/", 1)
        return provider, model_id

    @classmethod
    def _get_model_class(cls, model_type: str, provider: str) -> Type[BaseModel]:
        if model_type not in model_registry:
            raise ValueError(f"Model type `{model_type}` is not supported")
        if provider not in model_registry[model_type]:
            raise ValueError(
                f"Provider `{provider}` not registered for type `{model_type}`"
            )
        model_cls = model_registry[model_type][provider]
        return model_cls

    @classmethod
    def _create_model(
        cls, model_type: str, model_path: str, **kwargs
    ) -> Type[BaseModel]:
        provider, model_id = cls._model_path_parser(model_path)
        model_cls = cls._get_model_class(model_type, provider)
        return model_cls(model_id=model_id, **kwargs)

    @classmethod
    def from_serialized(
        cls, provider: str, model_type: str, state: Mapping[str, Any]
    ) -> Type[BaseModel]:
        """Creates a model instance from serialized parameters.

        Args:
            provider:
                The model provider (e.g., "openai", "google").
            model_type:
                The type of model (e.g., "chat_completion", "text_embedder").
            state:
                Dictionary containing the serialized model parameters.

        Returns:
            An instance of the appropriate model class with restored state
        """
        model_cls = cls._get_model_class(model_type, provider)
        # Create instance without calling __init__
        instance = object.__new__(model_cls)
        # Restore the instance state
        instance.from_serialized(state)
        return instance

    @classmethod
    def chat_completion(cls, model_path: str, **kwargs) -> ChatCompletionModel:
        return cls._create_model("chat_completion", model_path, **kwargs)

    @classmethod
    def image_classifier(cls, model_path: str, **kwargs) -> ImageClassifierModel:
        return cls._create_model("image_classifier", model_path, **kwargs)

    @classmethod
    def image_embedder(cls, model_path: str, **kwargs) -> ImageEmbedderModel:
        return cls._create_model("image_embedder", model_path, **kwargs)

    @classmethod
    def image_text_to_image(cls, model_path: str, **kwargs) -> ImageTextToImageModel:
        return cls._create_model("image_text_to_image", model_path, **kwargs)

    @classmethod
    def moderation(cls, model_path: str, **kwargs) -> ModerationModel:
        return cls._create_model("moderation", model_path, **kwargs)

    @classmethod
    def speech_to_text(cls, model_path: str, **kwargs) -> SpeechToTextModel:
        return cls._create_model("speech_to_text", model_path, **kwargs)

    @classmethod
    def text_classifier(cls, model_path: str, **kwargs) -> TextClassifierModel:
        return cls._create_model("text_classifier", model_path, **kwargs)

    @classmethod
    def text_embedder(cls, model_path: str, **kwargs) -> TextEmbedderModel:
        return cls._create_model("text_embedder", model_path, **kwargs)

    @classmethod
    def text_reranker(cls, model_path: str, **kwargs) -> TextRerankerModel:
        return cls._create_model("text_reranker", model_path, **kwargs)

    @classmethod
    def text_to_image(cls, model_path: str, **kwargs) -> TextToImageModel:
        return cls._create_model("text_to_image", model_path, **kwargs)

    @classmethod
    def text_to_speech(cls, model_path: str, **kwargs) -> TextToSpeechModel:
        return cls._create_model("text_to_speech", model_path, **kwargs)
