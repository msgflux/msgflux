from os import getenv
from typing import Any, Dict

from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.registry import register_model


class _BaseSambaNova:
    """Configurations to use SambaNova models."""

    provider: str = "sambanova"
    api_key_env: str = "SAMBANOVA_API_KEY"

    def _get_base_url(self):
        base_url = getenv("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1")
        if base_url is None:
            raise ValueError("Please set `SAMBANOVA_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv(self.api_key_env)
        if not key:
            raise ValueError(
                "The SambaNova API key is not available."
                f"Please set `{self.api_key_env}`"
            )
        return key


@register_model
class SambaNovaChatCompletion(_BaseSambaNova, OpenAICompatibleChatCompletion):
    """SambaNova Chat Completion."""

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        response_format = params.pop("response_format", None)
        if response_format:  # SambaNova NOT support strict=True
            response_format["json_schema"]["strict"] = False
            params["response_format"] = response_format
        return params
