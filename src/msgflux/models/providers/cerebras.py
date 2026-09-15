from msgflux.models.openai_compatible import (
    OpenAICompatibleChatCompletion,
    ProviderEnvBase,
)
from msgflux.models.registry import register_model


class _BaseCerebras(ProviderEnvBase):
    """Configurations to use Cerebras models."""

    provider: str = "cerebras"
    display_name: str = "Cerebras"
    api_key_env: str = "CEREBRAS_API_KEY"
    base_url_env: str = "CEREBRAS_BASE_URL"
    base_url: str = "https://api.cerebras.ai/v1"


@register_model
class CerebrasChatCompletion(_BaseCerebras, OpenAICompatibleChatCompletion):
    """Cerebras Chat Completion."""
