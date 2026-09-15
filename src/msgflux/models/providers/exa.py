from typing import Any, Dict

from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.registry import register_model


class _BaseExa:
    """Configurations to use Exa models via OpenAI-compatible API."""

    provider: str = "exa"
    display_name: str = "Exa"
    api_key_env: str = "EXA_API_KEY"
    base_url_env: str = "EXA_BASE_URL"
    base_url: str = "https://api.exa.ai"


@register_model
class ExaChatCompletion(_BaseExa, OpenAICompatibleChatCompletion):
    """Exa Chat Completion for Answer endpoint.

    Models available:
        - exa: For the /answer endpoint
        - exa-research: For deep research tasks
        - exa-research-pro: For comprehensive research

    Requires the `EXA_API_KEY` environment variable to be set.
    """

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt parameters for Exa API.

        Exa's OpenAI-compatible API supports extra parameters via extra_body
        such as 'text' for including full text from sources.
        """
        # Exa doesn't use max_tokens, use max_completion_tokens if needed
        if "max_tokens" in params and params["max_tokens"] is not None:
            params["max_completion_tokens"] = params.pop("max_tokens")
        else:
            params.pop("max_tokens", None)

        # Exa doesn't support tool_choice or tools for answer/research
        params.pop("tool_choice", None)
        params.pop("tools", None)

        return params
