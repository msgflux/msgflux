from os import getenv
from typing import Any, Dict

from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.registry import register_model


class _BaseExa:
    """Configurations to use Exa models via OpenAI-compatible API."""

    provider: str = "exa"

    def _get_base_url(self):
        base_url = getenv("EXA_BASE_URL", "https://api.exa.ai")
        if base_url is None:
            raise ValueError("Please set `EXA_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("EXA_API_KEY")
        if not key:
            raise ValueError(
                "The Exa API key is not available. Please set `EXA_API_KEY`"
            )
        return key


@register_model
class ExaChatCompletion(_BaseExa, OpenAIChatCompletion):
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
