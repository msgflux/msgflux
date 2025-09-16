from os import getenv
from typing import Any, Dict

from msgflux.models.providers.openai import OpenAIChatCompletion, OpenAITextEmbedder
from msgflux.models.registry import register_model


class _BaseOllama:
    """Configurations to use Ollama models."""

    provider: str = "ollama"

    def _get_base_url(self):
        base_url = getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        if base_url is None:
            raise ValueError("Please set `OLLAMA_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        keys = getenv("OLAMA_API_KEY", "ollama")
        self._api_key = [key.strip() for key in keys.split(",")]
        if not self._api_key:
            raise ValueError("No valid API keys found")

@register_model
class OllamaChatCompletion(_BaseOllama, OpenAIChatCompletion):
    """Ollama Chat Completion."""

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        extra_body = params.get("extra_body", {})

        if self.enable_thinking is not None:
            extra_body["think"] = self.enable_thinking

        params["extra_body"] = extra_body
        return params

@register_model
class OllamaTextEmbedder(OpenAITextEmbedder, _BaseOllama):
    """Ollama Text Embedder."""
