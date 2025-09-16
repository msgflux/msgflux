from os import getenv
from typing import Any, Dict

from msgflux.models.providers.openai import (
    OpenAIChatCompletion,
    OpenAITextEmbedder,
    OpenAITextToSpeech,
)
from msgflux.models.registry import register_model


class _BaseTogether:
    """Configurations to use Together models."""

    provider: str = "together"

    def _get_base_url(self):
        base_url = getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
        if base_url is None:
            raise ValueError("Please set `TOGETHER_BASE_URL`")
        return base_url

    def _get_api_key(self):
        keys = getenv("TOGETHER_API_KEY")
        self._api_key = [key.strip() for key in keys.split(",")]
        if not self._api_key:
            raise ValueError("No valid API keys found")

@register_model
class TogetherChatCompletion(_BaseTogether, OpenAIChatCompletion):
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
class TogetherTextEmbedder(OpenAITextEmbedder, _BaseTogether):
    """Together Text Embedder."""

@register_model
class TogetherTextToSpeech(OpenAITextToSpeech, _BaseTogether):
    """Together Text to Speech."""
