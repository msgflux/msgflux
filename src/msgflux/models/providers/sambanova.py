from os import getenv
from typing import Any, Dict

from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.registry import register_model


class _BaseSambaNova:
    """Configurations to use SambaNova models."""

    provider: str = "sambanova"

    def _get_base_url(self):
        base_url = getenv("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1")
        if base_url is None:
            raise ValueError("Please set `SAMBANOVA_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("SAMBANOVA_API_KEY")
        if not key:
            raise ValueError(
                "The SambaNova API key is not available."
                "Please set `SAMBANOVA_API_KEY`"
            )
        return key


@register_model
class SambaNovaChatCompletion(_BaseSambaNova, OpenAIChatCompletion):
    """SambaNova Chat Completion."""

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        response_format = params.pop("response_format", None)
        if response_format:  # SambaNova NOT support strict=True
            response_format["json_schema"]["strict"] = False
            params["response_format"] = response_format
        return params
