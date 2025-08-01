from os import getenv
from typing import Any, Dict

from msgflux.models.providers.openai import OpenAIChatCompletion, OpenAITextEmbedder


class _BaseOllama:
    """Configurations to use Ollama models."""

    provider: str = "ollama"

    def _get_base_url(self):
        base_url = getenv("OLLAMA_BASE_URL")
        if base_url is None:
            raise ValueError("Please set `OLLAMA_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        keys = getenv("OLAMA_API_KEY", "ollama")
        self._api_key = [key.strip() for key in keys.split(",")]
        if not self._api_key:
            raise ValueError("No valid API keys found")


class OllamaChatCompletion(OpenAIChatCompletion, _BaseOllama):
    """Ollama Chat Completion."""

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        response_format = params.pop("response_format", None)
        if response_format:
            params["extra_body"] = {"guided_json": response_format}
        return params


class OllamaTextEmbedder(OpenAITextEmbedder, _BaseOllama):
    """Ollama Text Embedder."""
