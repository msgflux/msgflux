from os import getenv
from typing import Any, Dict

from msgflux.models.providers.openai import (
    OpenAIChatCompletation,
    OpenAITextEmbedder,
    OpenAITextToSpeech,
)


class _BaseTogether:
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


class TogetherChatCompletation(OpenAIChatCompletation, _BaseTogether):
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


class TogetherTextEmbedder(OpenAITextEmbedder, _BaseTogether):
    """Together Text Embedder."""


class TogetherTextToSpeech(OpenAITextToSpeech, _BaseTogether):
    """Together Text to Speech."""
