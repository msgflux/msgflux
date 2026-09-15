from typing import Any, Dict

from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.registry import register_model


class _BaseSambaNova:
    """Configurations to use SambaNova models."""

    provider: str = "sambanova"
    display_name: str = "SambaNova"
    api_key_env: str = "SAMBANOVA_API_KEY"
    base_url_env: str = "SAMBANOVA_BASE_URL"
    base_url: str = "https://api.sambanova.ai/v1"


@register_model
class SambaNovaChatCompletion(_BaseSambaNova, OpenAICompatibleChatCompletion):
    """SambaNova Chat Completion."""

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        response_format = params.pop("response_format", None)
        if response_format:  # SambaNova NOT support strict=True
            response_format["json_schema"]["strict"] = False
            params["response_format"] = response_format
        return params
