from typing import Any, Dict, Type

from msgflux.models.base import BaseModel
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
from msgflux.utils.imports import import_module_from_lib

_SUPPORTED_MODEL_TYPES = [
    "chat_completion",
    "image_classfier",
    "image_embedder",
    "image_text_to_image",
    "moderation",
    "speech_to_text",
    "text_classifier",
    "text_embedder",
    "text_reranker",
    "text_to_image",
    "text_to_speech",
]

_MODEL_NAMESPACE_TRANSLATOR = {
    "jinaai": "JinaAI",
    "ollama": "Ollama",
    "openai": "OpenAI",
    "openrouter": "OpenRouter",
    "sambanova": "SambaNova",
    "timm": "TIMM",
    "together": "Together",
    "vllm": "VLLM",
}

_CHAT_COMPLETION_PROVIDERS = [
    "ollama",
    "openai",
    "openrouter",
    "sambanova",
    "together",
    "vllm",
]
_IMAGE_CLASSIFIER_PROVIDERS = ["jinaai"]
_IMAGE_EMBEDDER_PROVIDERS = ["jinaai"]
_IMAGE_TEXT_TO_IMAGE_PROVIDERS = ["openai"]
_MODERATION_PROVIDERS = ["openai"]
_SPEECH_TO_TEXT_PROVIDERS = ["openai", "vllm"]
_TEXT_CLASSIFIER_PROVIDERS = ["jinaai", "vllm"]
_TEXT_EMBEDDER_PROVIDERS = ["jinaai", "ollama", "openai", "together", "vllm"]
_TEXT_RERANKER_PROVIDERS = ["jinaai", "vllm"]
_TEXT_TO_IMAGE = ["openai", "together"]
_TEXT_TO_SPEECH_PROVIDERS = ["openai", "together"]


_PROVIDERS_BY_MODEL_TYPE = {
    "chat_completion": _CHAT_COMPLETION_PROVIDERS,
    "image_classifier": _IMAGE_CLASSIFIER_PROVIDERS,
    "image_embedder": _IMAGE_EMBEDDER_PROVIDERS,
    "image_text_to_image": _IMAGE_TEXT_TO_IMAGE_PROVIDERS,
    "moderation": _MODERATION_PROVIDERS,
    "speech_to_text": _SPEECH_TO_TEXT_PROVIDERS,
    "text_classifier": _TEXT_CLASSIFIER_PROVIDERS,
    "text_embedder": _TEXT_EMBEDDER_PROVIDERS,
    "text_reranker": _TEXT_RERANKER_PROVIDERS,
    "text_to_image": _TEXT_TO_IMAGE,
    "text_to_speech": _TEXT_TO_SPEECH_PROVIDERS,
}


class Model:
    supported_model_types = _SUPPORTED_MODEL_TYPES
    providers_by_model_type = _PROVIDERS_BY_MODEL_TYPE

    @classmethod
    def _model_path_parser(cls, model_id: str) -> tuple[str, str]:
        provider, model_id = model_id.split("/", 1)
        return provider, model_id

    @classmethod
    def _get_model_class(cls, model_type: str, provider: str) -> Type[BaseModel]:
        if model_type not in cls.supported_model_types:
            raise ValueError(f"Model type `{model_type}` is not supported")

        providers = cls.providers_by_model_type[model_type]
        if provider not in providers:
            raise ValueError(f"Provider `{provider}` is not supported for {model_type}")

        if len(model_type) <= 3:
            model_type = model_type.upper()
        else:
            model_type = model_type.title().replace("_", "")

        provider_class_name = f"{_MODEL_NAMESPACE_TRANSLATOR[provider]}{model_type}"
        module_name = f"msgflux.models.providers.{provider}"
        return import_module_from_lib(provider_class_name, module_name)

    @classmethod
    def _create_model(
        cls, model_type: str, model_path: str, **kwargs
    ) -> Type[BaseModel]:
        provider, model_id = cls._model_path_parser(model_path)
        model_cls = cls._get_model_class(model_type, provider)
        return model_cls(model_id=model_id, **kwargs)

    @classmethod
    def from_serialized(
        cls, provider: str, model_type: str, state: Dict[str, Any]
    ) -> Type[BaseModel]:
        """Creates a model instance from serialized parameters without calling __init__.

        Args:
            provider: The model provider (e.g., "openai", "google")
            model_type: The type of model (e.g., "chat_completation", "text_embedder")
            state: Dictionary containing the serialized model parameters

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
    def chat_completion(cls, model_path: str, **kwargs) -> Type[ChatCompletionModel]:
        return cls._create_model("chat_completion", model_path, **kwargs)

    @classmethod
    def image_classifier(cls, model_path: str, **kwargs) -> Type[ImageClassifierModel]:
        return cls._create_model("image_classifier", model_path, **kwargs)

    @classmethod
    def image_embedder(cls, model_path: str, **kwargs) -> Type[ImageEmbedderModel]:
        return cls._create_model("image_embedder", model_path, **kwargs)

    @classmethod
    def image_text_to_image(
        cls, model_path: str, **kwargs
    ) -> Type[ImageTextToImageModel]:
        return cls._create_model("image_text_to_image", model_path, **kwargs)

    @classmethod
    def moderation(cls, model_path: str, **kwargs) -> Type[ModerationModel]:
        return cls._create_model("moderation", model_path, **kwargs)

    @classmethod
    def speech_to_text(cls, model_path: str, **kwargs) -> Type[SpeechToTextModel]:
        return cls._create_model("speech_to_text", model_path, **kwargs)

    @classmethod
    def text_classifier(cls, model_path: str, **kwargs) -> Type[TextClassifierModel]:
        return cls._create_model("text_classifier", model_path, **kwargs)

    @classmethod
    def text_embedder(cls, model_path: str, **kwargs) -> Type[TextEmbedderModel]:
        return cls._create_model("text_embedder", model_path, **kwargs)

    @classmethod
    def text_reranker(cls, model_path: str, **kwargs) -> Type[TextRerankerModel]:
        return cls._create_model("text_reranker", model_path, **kwargs)

    @classmethod
    def text_to_image(cls, model_path: str, **kwargs) -> Type[TextToImageModel]:
        return cls._create_model("text_to_image", model_path, **kwargs)

    @classmethod
    def text_to_speech(cls, model_path: str, **kwargs) -> Type[TextToSpeechModel]:
        return cls._create_model("text_to_speech", model_path, **kwargs)
