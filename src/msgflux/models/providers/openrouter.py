from os import getenv

from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.registry import register_model


class _BaseOpenRouter:
    """Configurations to use OpenRouter models."""

    provider: str = "openrouter"

    def _get_base_url(self):
        base_url = getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if base_url is None:
            raise ValueError("Please set `OPENROUTER_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        keys = getenv("OPENROUTER_API_KEY")
        self._api_key = [key.strip() for key in keys.split(",")]
        if not self._api_key:
            raise ValueError("No valid API keys found")

@register_model
class OpenRouterChatCompletion(_BaseOpenRouter, OpenAIChatCompletion):
    """OpenRouter Chat Completion."""
