from os import getenv

from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.registry import register_model


class _BaseCerebras:
    """Configurations to use Cerebras models."""

    provider: str = "cerebras"

    def _get_base_url(self):
        base_url = getenv("CEBEBRAS_BASE_URL", "https://api.cerebras.ai/v1")
        if base_url is None:
            raise ValueError("Please set `CEBEBRAS_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        keys = getenv("CEREBRAS_API_KEY")
        self._api_key = [key.strip() for key in keys.split(",")]
        if not self._api_key:
            raise ValueError("No valid API keys found")

@register_model
class CerebrasChatCompletion(_BaseCerebras, OpenAIChatCompletion):
    """Cerebras Chat Completion."""
