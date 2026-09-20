from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import register_model


class _BaseDeepInfra(ProviderEnvBase):
    """Configurations to use DeepInfra models."""

    provider: str = "deepinfra"
    display_name: str = "DeepInfra"
    api_key_env: str = "DEEPINFRA_API_KEY"
    base_url_env: str = "DEEPINFRA_BASE_URL"
    base_url: str = "https://api.deepinfra.com/v1/openai"


@register_model
class DeepInfraChatCompletion(_BaseDeepInfra, OpenAICompatibleChatCompletion):
    """DeepInfra chat completion.

    OpenAI-compatible `POST /v1/openai/chat/completions` on
    `https://api.deepinfra.com/v1/openai`. Only `chat_completions` is
    declared. The schema accepts `reasoning_effort`, a `reasoning`
    object, native `reasoning_content` on assistant messages, and
    `prompt_cache_key` for prefix-cache reuse (thread-bound session
    injection is a follow-up).
    """
