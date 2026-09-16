from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import register_model


class _BaseInferenceNet(ProviderEnvBase):
    """Configurations to use InferenceNet models."""

    provider: str = "inference-net"
    display_name: str = "InferenceNet"
    api_key_env: str = "INFERENCE_API_KEY"
    base_url_env: str = "INFERENCE_BASE_URL"
    base_url: str = "https://api.inference.net/v1"


@register_model
class InferenceNetChatCompletion(_BaseInferenceNet, OpenAICompatibleChatCompletion):
    """InferenceNet chat completion.

    OpenAI-compatible `POST /v1/chat/completions` on
    `https://api.inference.net/v1`. Only `chat_completions` is declared;
    Responses support is unverified.
    """
