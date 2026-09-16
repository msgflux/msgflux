from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import register_model


class _BaseInference(ProviderEnvBase):
    """Configurations to use Inference.net models."""

    provider: str = "inference"
    display_name: str = "Inference"
    api_key_env: str = "INFERENCE_API_KEY"
    base_url_env: str = "INFERENCE_BASE_URL"
    base_url: str = "https://api.inference.net/v1"


@register_model
class InferenceChatCompletion(_BaseInference, OpenAICompatibleChatCompletion):
    """Inference.net chat completion.

    OpenAI-compatible `POST /v1/chat/completions` on
    `https://api.inference.net/v1`. Only `chat_completions` is declared;
    Responses support is unverified.
    """
