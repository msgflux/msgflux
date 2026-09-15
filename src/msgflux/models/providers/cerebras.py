from os import getenv

from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.registry import register_model


class _BaseCerebras:
    """Configurations to use Cerebras models."""

    provider: str = "cerebras"
    api_key_env: str = "CEREBRAS_API_KEY"

    def _get_base_url(self):
        base_url = getenv("CEBEBRAS_BASE_URL", "https://api.cerebras.ai/v1")
        if base_url is None:
            raise ValueError("Please set `CEBEBRAS_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv(self.api_key_env)
        if not key:
            raise ValueError(
                "The Cerebras API key is not available. "
                f"Please set `{self.api_key_env}`"
            )
        return key


@register_model
class CerebrasChatCompletion(_BaseCerebras, OpenAICompatibleChatCompletion):
    """Cerebras Chat Completion."""
