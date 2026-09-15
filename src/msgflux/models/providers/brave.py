from os import getenv

from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.registry import register_model


class _BaseBrave:
    """Configurations to use Brave models."""

    provider: str = "brave"
    api_key_env: str = "BRAVE_SEARCH_API_KEY"

    def _get_base_url(self):
        return "https://api.search.brave.com/res/v1"

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv(self.api_key_env)
        if not key:
            raise ValueError(
                f"The Brave API key is not available. Please set `{self.api_key_env}`"
            )
        return key


@register_model
class BraveChatCompletion(_BaseBrave, OpenAICompatibleChatCompletion):
    """Brave Chat Completion."""
