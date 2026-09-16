from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import register_model


class _BaseGMICloud(ProviderEnvBase):
    """Configurations to use GMI Cloud models."""

    provider: str = "gmicloud"
    display_name: str = "GMICloud"
    api_key_env: str = "GMICLOUD_API_KEY"
    base_url_env: str = "GMICLOUD_BASE_URL"
    base_url: str = "https://api.gmi-serving.com/v1"


@register_model
class GMICloudChatCompletion(_BaseGMICloud, OpenAICompatibleChatCompletion):
    """GMI Cloud chat completion.

    OpenAI-compatible `POST /v1/chat/completions` on
    `https://api.gmi-serving.com/v1`. Only `chat_completions` is declared;
    Responses support is unverified. Reasoning models return native
    `reasoning_content`, which is extracted into history.
    """
