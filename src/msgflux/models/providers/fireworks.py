from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import register_model


class _BaseFireworks(ProviderEnvBase):
    """Configurations to use Fireworks models."""

    provider: str = "fireworks"
    display_name: str = "Fireworks"
    api_key_env: str = "FIREWORKS_API_KEY"
    base_url_env: str = "FIREWORKS_BASE_URL"
    base_url: str = "https://api.fireworks.ai/inference/v1"


@register_model
class FireworksChatCompletion(_BaseFireworks, OpenAICompatibleChatCompletion):
    """Fireworks chat completion.

    OpenAI-compatible `POST /v1/chat/completions` on
    `https://api.fireworks.ai/inference/v1`. Only `chat_completions` is
    declared. Reasoning models return clear-text `reasoning_content`
    alongside content, with reasoning token accounting in usage.
    """
