import re
from typing import TYPE_CHECKING, Any

from msgflux.models.chat_capabilities import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
)
from msgflux.models.openai_compatible import (
    OpenAIChatCompletionsAPI,
    OpenAICompatibleChatCompletion,
    OpenAIResponsesAPI,
)
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.reasoning import OpenAICompatibleReasoningCodec
from msgflux.utils.imports import AutoloadRegistry

if TYPE_CHECKING:
    from msgflux.models.base import BaseModel

model_registry = AutoloadRegistry("msgflux.models.providers")


def register_model(cls: type["BaseModel"]):
    model_type = getattr(cls, "model_type", None)
    provider = getattr(cls, "provider", None)

    if not model_type or not provider:
        raise ValueError(f"{cls.__name__} must define `model_type` and `provider`.")

    model_registry.setdefault(model_type, {})[provider] = cls
    return cls


_CHAT_API_MODES = ("chat_completions", "responses")


def register_chat_provider(
    provider: str,
    *,
    base_url: str,
    api_key_env: str,
    display_name: str | None = None,
    base_url_env: str | None = None,
    api_modes: tuple[str, ...] = ("chat_completions",),
    default_api_mode: str = "chat_completions",
    overwrite: bool = False,
) -> type["BaseModel"]:
    """Register an OpenAI-compatible chat provider without writing a class.

    Synthesizes the equivalent of a `_Base` env mixin plus an
    `OpenAICompatibleChatCompletion` subclass and registers it, so
    `Model.chat_completion(f"{provider}/<model-id>")` resolves. For
    parameter adaptation, reasoning codecs, or native transports, write a
    provider class instead.

    Args:
        provider: Registry name, also the `model_path` prefix. Must not
            contain `/`.
        base_url: Default endpoint base URL.
        api_key_env: Environment variable holding the API key.
        display_name: Human-readable name for errors. Defaults to `provider`.
        base_url_env: Optional environment variable overriding `base_url`.
        api_modes: Subset of `("chat_completions", "responses")`.
        default_api_mode: Mode used when none is requested.
        overwrite: Replace an existing registration instead of raising.

    Returns:
        The synthesized, registered model class.
    """
    if not isinstance(provider, str) or not provider or "/" in provider:
        raise ValueError("`provider` must be a non-empty string without `/`")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("`base_url` must be a non-empty string")
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise ValueError("`api_key_env` must be a non-empty string")
    if base_url_env is not None and (
        not isinstance(base_url_env, str) or not base_url_env.strip()
    ):
        raise ValueError("`base_url_env` must be a non-empty string or None")
    modes = tuple(api_modes)
    if not modes or any(mode not in _CHAT_API_MODES for mode in modes):
        raise ValueError(f"`api_modes` must be a non-empty subset of {_CHAT_API_MODES}")
    if default_api_mode not in modes:
        raise ValueError("`default_api_mode` must be one of `api_modes`")

    existing = model_registry.get("chat_completion", {}).get(provider)
    if existing is not None and not overwrite:
        raise ValueError(
            f"Provider `{provider}` is already registered for `chat_completion`. "
            "Pass `overwrite=True` to replace it."
        )

    adapters: dict[str, Any] = {
        "chat_completions": OpenAIChatCompletionsAPI(),
        "responses": OpenAIResponsesAPI(),
    }
    mode_capabilities = tuple(
        ChatAPIModeCapabilities(
            name=mode,
            adapter=adapters[mode],
            request_reasoning_effort=True,
        )
        for mode in modes
    )

    base_name = "".join(part.capitalize() for part in re.split(r"[_-]+", provider))
    env_base = type(
        f"_Base{base_name}",
        (ProviderEnvBase,),
        {
            "provider": provider,
            "display_name": display_name or provider,
            "api_key_env": api_key_env,
            "base_url_env": base_url_env,
            "base_url": base_url,
            "__module__": "msgflux.models.providers",
        },
    )
    cls: type[BaseModel] = type(
        f"{base_name}ChatCompletion",
        (env_base, OpenAICompatibleChatCompletion),
        {
            "capabilities": ChatProviderCapabilities(
                default_api_mode=default_api_mode,
                api_modes=mode_capabilities,
                default_reasoning_codec=OpenAICompatibleReasoningCodec(),
            ),
            "__module__": "msgflux.models.providers",
            "__doc__": f"{display_name or provider} chat completion (registered).",
        },
    )
    return register_model(cls)
