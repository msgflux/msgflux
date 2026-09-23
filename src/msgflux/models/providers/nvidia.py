from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import register_model


class _BaseNVIDIA(ProviderEnvBase):
    """Configurations to use NVIDIA NIM models."""

    provider: str = "nvidia"
    display_name: str = "NVIDIA"
    api_key_env: str = "NVIDIA_API_KEY"
    base_url_env: str = "NVIDIA_BASE_URL"
    base_url: str = "https://integrate.api.nvidia.com/v1"


@register_model
class NVIDIAChatCompletion(_BaseNVIDIA, OpenAICompatibleChatCompletion):
    """NVIDIA NIM chat completion.

    OpenAI-compatible `POST /v1/chat/completions` on
    `https://integrate.api.nvidia.com/v1`. The hosted endpoint does not
    serve `/v1/responses` (404) and rejects `prompt_cache_key` (400), so
    only `chat_completions` is declared. Reasoning models return
    `reasoning_content`, which is extracted into history but not replayed.
    """
