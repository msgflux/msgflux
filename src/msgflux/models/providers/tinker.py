from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import register_model


class _BaseTinker(ProviderEnvBase):
    """Configurations to use Thinking Machines Tinker models."""

    provider: str = "tinker"
    display_name: str = "Tinker"
    api_key_env: str = "TINKER_API_KEY"
    base_url_env: str = "TINKER_BASE_URL"
    base_url: str = (
        "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
    )


@register_model
class TinkerChatCompletion(_BaseTinker, OpenAICompatibleChatCompletion):
    """Thinking Machines Tinker chat completion.

    `POST /chat/completions` on the Tinker inference endpoint. Only
    `chat_completions` is declared (`/completions` is a legacy prompt
    API, not a msgflux model type). Model ids are sampler checkpoint
    paths (`tinker://...`). Reasoning models return `reasoning_content`
    separately from `content` by default (`separate_reasoning`, true
    since June 2026); `reasoning_effort` accepts OpenAI strings or a
    float in [0.0, 0.99] on supporting models (400 otherwise).
    """
