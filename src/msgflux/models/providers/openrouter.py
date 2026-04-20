from os import getenv
from typing import Any, Dict

from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.registry import register_model


class _BaseOpenRouter:
    """Configurations to use OpenRouter models."""

    provider: str = "openrouter"

    def _get_base_url(self):
        base_url = getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        if base_url is None:
            raise ValueError("Please set `OPENROUTER_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError(
                "The OpenRouter API key is not available."
                "Please set `OPENROUTER_API_KEY`"
            )
        return key


@register_model
class OpenRouterChatCompletion(_BaseOpenRouter, OpenAIChatCompletion):
    """OpenRouter Chat Completion."""

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        extra_body = params.get("extra_body", {})
        plugins = []

        if params["tool_choice"] is None:
            if params["tools"] is not None:
                params["tool_choice"] = "auto"
            else:
                params["tool_choice"] = "none"

        reasoning_effort = params.pop("reasoning_effort", None)
        reasoning_max_tokens = params.pop("reasoning_max_tokens", None)
        if reasoning_effort is not None and reasoning_max_tokens is not None:
            raise ValueError(
                "`reasoning_max_tokens` cannot be used together with "
                "`reasoning_effort` for OpenRouter."
            )
        if reasoning_effort is not None:
            extra_body["reasoning"] = {"effort": reasoning_effort}
        if reasoning_max_tokens is not None:
            extra_body["reasoning"] = {"max_tokens": reasoning_max_tokens}

        # For non-OpenAI models enable web-search plugin
        web_search_options = params.get("web_search_options", None)
        if web_search_options is not None and "openai" not in params["model"]:
            params.pop("web_search_options")
            web_pluging = {"id": "web"}
            web_pluging.update(web_search_options)
            plugins.append(web_pluging)

        if plugins:
            extra_body["plugins"] = plugins

        params["extra_body"] = extra_body
        params["extra_headers"] = {
            "HTTP-Referer": "msgflux.com",
            "X-Title": "msgflux",
        }
        return params
