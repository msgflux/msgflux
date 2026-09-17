from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import register_model


class _BaseRunInfra(ProviderEnvBase):
    """Configurations to use RunInfra models."""

    provider: str = "runinfra"
    display_name: str = "RunInfra"
    api_key_env: str = "RUNINFRA_API_KEY"
    base_url_env: str = "RUNINFRA_BASE_URL"
    base_url: str = "https://api.runinfra.ai/v1"


@register_model
class RunInfraChatCompletion(_BaseRunInfra, OpenAICompatibleChatCompletion):
    """RunInfra chat completion.

    OpenAI-compatible `POST /v1/chat/completions` on
    `https://api.runinfra.ai/v1`. Only `chat_completions` is declared.
    Reasoning models return clear-text `reasoning` alongside content,
    with reasoning token accounting in usage.
    """
