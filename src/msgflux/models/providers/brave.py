from msgflux.models.openai_compatible import (
    OpenAICompatibleChatCompletion,
)
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import register_model


class _BaseBrave(ProviderEnvBase):
    """Configurations to use Brave models."""

    provider: str = "brave"
    display_name: str = "Brave"
    api_key_env: str = "BRAVE_SEARCH_API_KEY"
    base_url_env = None
    base_url: str = "https://api.search.brave.com/res/v1"


@register_model
class BraveChatCompletion(_BaseBrave, OpenAICompatibleChatCompletion):
    """Brave Chat Completion."""
