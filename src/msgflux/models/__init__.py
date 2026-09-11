from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from msgflux.models.chat_capabilities import (
        ChatAPIModeCapabilities,
        ChatProviderCapabilities,
    )
    from msgflux.models.chat_context import (
        ChatContextAdapter,
        OpenAIResponsesContextAdapter,
    )
    from msgflux.models.chat_extensions import ChatModelExtension, ChatRequestContext
    from msgflux.models.compaction import ContextTokenEstimate, ModelCompaction
    from msgflux.models.model import Model
    from msgflux.models.model_credentials import (
        BearerTokenCredentialResolver,
        ModelCredentialResolver,
        ResolvedModelCredentials,
    )

__all__ = [
    "BearerTokenCredentialResolver",
    "ChatAPIModeCapabilities",
    "ChatContextAdapter",
    "ChatModelExtension",
    "ChatRequestContext",
    "ChatProviderCapabilities",
    "ContextTokenEstimate",
    "Model",
    "ModelCompaction",
    "ModelCredentialResolver",
    "OpenAIResponsesContextAdapter",
    "ResolvedModelCredentials",
]


def __getattr__(name: str):
    if name in {
        "BearerTokenCredentialResolver",
        "ModelCredentialResolver",
        "ResolvedModelCredentials",
    }:
        value = getattr(import_module("msgflux.models.model_credentials"), name)
        globals()[name] = value
        return value
    if name in {"ChatAPIModeCapabilities", "ChatProviderCapabilities"}:
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
    if name != "Model":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module("msgflux.models.model"), name)
    globals()[name] = value
    return value
