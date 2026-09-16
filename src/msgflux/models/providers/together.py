from typing import Any, Dict

from msgflux.models.openai_compatible import (
    OpenAICompatibleChatCompletion,
)
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.providers.openai import (
    OpenAITextEmbedder,
    OpenAITextToSpeech,
)
from msgflux.models.registry import register_model


class _BaseTogether(ProviderEnvBase):
    """Configurations to use Together models."""

    provider: str = "together"
    display_name: str = "Together"
    api_key_env: str = "TOGETHER_API_KEY"
    base_url_env: str = "TOGETHER_BASE_URL"
    base_url: str = "https://api.together.xyz/v1"


@register_model
class TogetherChatCompletion(_BaseTogether, OpenAICompatibleChatCompletion):
    """Together Chat Completion."""

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        response_format = params.pop("response_format", None)
        if response_format:
            params["response_format"] = {
                "type": "json_object",
                "schema": response_format,
            }
        tools = params.get("tools", None)
        if tools:  # Together supports 'strict' mode to tools
            for tool in tools:
                tool["function"]["strict"] = True
        return params


@register_model
class TogetherTextEmbedder(_BaseTogether, OpenAITextEmbedder):
    """Together Text Embedder."""


@register_model
class TogetherTextToSpeech(_BaseTogether, OpenAITextToSpeech):
    """Together Text to Speech."""
