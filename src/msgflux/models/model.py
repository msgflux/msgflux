from typing import TYPE_CHECKING, Any, Mapping

from msgflux.models.catalog import (
    get_current_provider_catalog,
    validate_serialized_source,
)
from msgflux.models.registry import model_registry

if TYPE_CHECKING:
    from msgflux.models.base import BaseModel
    from msgflux.models.catalog import ProviderCatalog

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


def _resolve_catalog(
    provider_catalog: "ProviderCatalog | None",
) -> "ProviderCatalog | None":
    return (
        provider_catalog
        if provider_catalog is not None
        else get_current_provider_catalog()
    )


class Model:
    @classmethod
    def providers(cls, *, provider_catalog=None):
        catalog = _resolve_catalog(provider_catalog)
        if catalog is not None:
            return catalog.providers()
        return {k: list(v.keys()) for k, v in model_registry.items()}

    @classmethod
    def model_types(cls):
        return list(model_registry.keys())

    @classmethod
    def _model_path_parser(cls, model_id: str) -> tuple[str, str]:
        provider, model_id = model_id.split("/", 1)
        return provider, model_id

    @classmethod
    def _get_model_class(
        cls,
        model_type: str,
        provider: str,
        provider_catalog: "ProviderCatalog | None" = None,
    ) -> type["BaseModel"]:
        catalog = _resolve_catalog(provider_catalog)
        if catalog is not None:
            return catalog.get_model_class(model_type, provider)
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
        cls,
        model_type: str,
        model_path: str,
        *,
        provider_catalog: "ProviderCatalog | None" = None,
        **kwargs,
    ) -> type["BaseModel"]:
        provider, model_id = cls._model_path_parser(model_path)
        catalog = _resolve_catalog(kwargs.pop("provider_catalog", provider_catalog))
        if catalog is not None:
            return catalog.create_model(model_type, provider, model_id, **kwargs)
        model_cls = cls._get_model_class(model_type, provider)
        return model_cls(model_id=model_id, **kwargs)

    @classmethod
    def from_serialized(
        cls,
        provider: str,
        model_type: str,
        state: Mapping[str, Any],
        *,
        provider_catalog: "ProviderCatalog | None" = None,
    ) -> type["BaseModel"]:
        """Creates a model instance from serialized parameters.

        Args:
            provider:
                The model provider (e.g., "openai", "google").
            model_type:
                The type of model (e.g., "chat_completion", "text_embedder").
            state:
                Dictionary containing the serialized model parameters.
            provider_catalog:
                Trusted catalog used to resolve and validate configured
                providers. When omitted, the catalog active via
                ``ProviderCatalog.use()`` is used, if any.

        Returns:
            An instance of the appropriate model class with restored state
        """
        catalog = _resolve_catalog(provider_catalog)
        if catalog is not None:
            return catalog.from_serialized(provider, model_type, state)
        validate_serialized_source(provider, state, None)
        model_cls = cls._get_model_class(model_type, provider)
        return model_cls.from_serialized(state)

    @classmethod
    def chat_completion(
        cls, model_path: str, *, provider_catalog=None, **kwargs
    ) -> ChatCompletionModel:
        return cls._create_model(
            "chat_completion", model_path, provider_catalog=provider_catalog, **kwargs
        )

    @classmethod
    def image_classifier(
        cls, model_path: str, *, provider_catalog=None, **kwargs
    ) -> ImageClassifierModel:
        return cls._create_model(
            "image_classifier", model_path, provider_catalog=provider_catalog, **kwargs
        )

    @classmethod
    def image_embedder(
        cls, model_path: str, *, provider_catalog=None, **kwargs
    ) -> ImageEmbedderModel:
        return cls._create_model(
            "image_embedder", model_path, provider_catalog=provider_catalog, **kwargs
        )

    @classmethod
    def image_text_to_image(
        cls, model_path: str, *, provider_catalog=None, **kwargs
    ) -> ImageTextToImageModel:
        return cls._create_model(
            "image_text_to_image",
            model_path,
            provider_catalog=provider_catalog,
            **kwargs,
        )

    @classmethod
    def moderation(
        cls, model_path: str, *, provider_catalog=None, **kwargs
    ) -> ModerationModel:
        return cls._create_model(
            "moderation", model_path, provider_catalog=provider_catalog, **kwargs
        )

    @classmethod
    def speech_to_text(
        cls, model_path: str, *, provider_catalog=None, **kwargs
    ) -> SpeechToTextModel:
        return cls._create_model(
            "speech_to_text", model_path, provider_catalog=provider_catalog, **kwargs
        )

    @classmethod
    def text_classifier(
        cls, model_path: str, *, provider_catalog=None, **kwargs
    ) -> TextClassifierModel:
        return cls._create_model(
            "text_classifier", model_path, provider_catalog=provider_catalog, **kwargs
        )

    @classmethod
    def text_embedder(
        cls, model_path: str, *, provider_catalog=None, **kwargs
    ) -> TextEmbedderModel:
        return cls._create_model(
            "text_embedder", model_path, provider_catalog=provider_catalog, **kwargs
        )

    @classmethod
    def text_reranker(
        cls, model_path: str, *, provider_catalog=None, **kwargs
    ) -> TextRerankerModel:
        return cls._create_model(
            "text_reranker", model_path, provider_catalog=provider_catalog, **kwargs
        )

    @classmethod
    def text_to_image(
        cls, model_path: str, *, provider_catalog=None, **kwargs
    ) -> TextToImageModel:
        return cls._create_model(
            "text_to_image", model_path, provider_catalog=provider_catalog, **kwargs
        )

    @classmethod
    def text_to_speech(
        cls, model_path: str, *, provider_catalog=None, **kwargs
    ) -> TextToSpeechModel:
        return cls._create_model(
            "text_to_speech", model_path, provider_catalog=provider_catalog, **kwargs
        )
