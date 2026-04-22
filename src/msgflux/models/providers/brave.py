from os import getenv

from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.registry import register_model


class _BaseBrave:
    """Configurations to use Brave models."""

    provider: str = "brave"

    def _get_base_url(self):
        return "https://api.search.brave.com/res/v1"

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("BRAVE_SEARCH_API_KEY")
        if not key:
            raise ValueError(
                "The Brave API key is not available. Please set `BRAVE_SEARCH_API_KEY`"
            )
        return key


@register_model
class BraveChatCompletion(_BaseBrave, OpenAIChatCompletion):
    """Brave Chat Completion."""
