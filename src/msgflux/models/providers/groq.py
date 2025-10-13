from os import getenv
from typing import Any, Dict

from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.registry import register_model


class _BaseGroq:
    """Configurations to use Groq models."""

    provider: str = "groq"

    def _get_base_url(self):
        base_url = getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        if base_url is None:
            raise ValueError("Please set `GROQ_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "The Grok API key is not available. Please set `GROQ_API_KEY`"
            )
        return key

@register_model
class GroqChatCompletion(_BaseGroq, OpenAIChatCompletion):
    """Groq Chat Completion."""

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        extra_body = params.get("extra_body", {})
        params["max_completion_tokens"] = params.pop("max_tokens")
        if params["tool_choice"] is None:
            if params["tools"] is not None:
                params["tool_choice"] = "auto"
            else:
                params["tool_choice"] = "none"
        if self.sampling_run_params.get("reasoning_effort", None):            
            extra_body["reasoning_format"] = "parsed"
        params["extra_body"] = extra_body
        return params
