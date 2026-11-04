from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.models.catalog import (
        ProviderCatalog,
        ProviderConfig,
        get_current_provider_catalog,
    )
    from msgflux.models.chat_capabilities import (
        ChatAPIModeCapabilities,
        ChatModelCapabilities,
        ChatProviderCapabilities,
    )
    from msgflux.models.chat_context import (
        ChatContextAdapter,
        OpenAIResponsesContextAdapter,
    )
    from msgflux.models.chat_extensions import ChatModelExtension, ChatRequestContext
    from msgflux.models.compaction import ContextTokenEstimate, ModelCompaction
    from msgflux.models.configuration import ModelConfiguration
    from msgflux.models.model import Model
    from msgflux.models.model_credentials import (
        APIKeyCredentialResolver,
        BearerTokenCredentialResolver,
        ModelCredentialResolver,
        ResolvedModelCredentials,
    )
    from msgflux.models.registry import register_model

__all__ = [
    "APIKeyCredentialResolver",
    "BearerTokenCredentialResolver",
    "ChatAPIModeCapabilities",
    "ChatContextAdapter",
    "ChatModelExtension",
    "ChatModelCapabilities",
    "ChatRequestContext",
    "ChatProviderCapabilities",
    "ContextTokenEstimate",
    "Model",
    "ModelConfiguration",
    "ModelCompaction",
    "ModelCredentialResolver",
    "OpenAIResponsesContextAdapter",
    "ProviderCatalog",
    "ProviderConfig",
    "ResolvedModelCredentials",
    "get_current_provider_catalog",
    "register_model",
]


def __getattr__(name: str):
    if name in {
        "APIKeyCredentialResolver",
        "BearerTokenCredentialResolver",
        "ModelCredentialResolver",
        "ResolvedModelCredentials",
    }:
        value = getattr(import_module("msgflux.models.model_credentials"), name)
        globals()[name] = value
        return value
    if name in {
        "ChatAPIModeCapabilities",
        "ChatModelCapabilities",
        "ChatProviderCapabilities",
    }:
        value = getattr(import_module("msgflux.models.chat_capabilities"), name)
        globals()[name] = value
        return value
    if name in {"ChatContextAdapter", "OpenAIResponsesContextAdapter"}:
        value = getattr(import_module("msgflux.models.chat_context"), name)
        globals()[name] = value
        return value
    if name in {"ChatModelExtension", "ChatRequestContext"}:
        value = getattr(import_module("msgflux.models.chat_extensions"), name)
        globals()[name] = value
        return value
    if name in {"ContextTokenEstimate", "ModelCompaction"}:
        value = getattr(import_module("msgflux.models.compaction"), name)
        globals()[name] = value
        return value
    if name == "ModelConfiguration":
        value = getattr(import_module("msgflux.models.configuration"), name)
        globals()[name] = value
        return value
    if name in {
        "ProviderCatalog",
        "ProviderConfig",
        "get_current_provider_catalog",
    }:
        value = getattr(import_module("msgflux.models.catalog"), name)
        globals()[name] = value
        return value
    if name == "register_model":
        value = getattr(import_module("msgflux.models.registry"), name)
        globals()[name] = value
        return value
    if name != "Model":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module("msgflux.models.model"), name)
    globals()[name] = value
    return value
