from os import getenv
from typing import Any, Dict

import msgspec

from msgflux.models.chat_extensions import ChatRequestContext, ChatSpeedExtension
from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.reasoning import OpenRouterReasoningCodec
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


class OpenRouterSpeedExtension(ChatSpeedExtension):
    """Map speed preferences to OpenRouter fast mode or Nitro routing."""

    def accepts(self, owner: Any, value: Any) -> bool:
        if value == "ultrafast":
            return False
        if value == "nitro":
            return self._can_apply_nitro(owner.model_id)
        if value == "fast":
            return self._is_claude(owner.model_id) or self._can_apply_nitro(
                owner.model_id
            )
        return False

    def prepare_request(self, owner, request, context: ChatRequestContext):
        speed = owner.chat_settings.get("speed")
        if speed is None or context.operation == "token_count":
            return request
        params = dict(request.params)
        extra_body = dict(params.get("extra_body") or {})
        extra_body.pop("speed", None)
        if extra_body:
            params["extra_body"] = extra_body
        else:
            params.pop("extra_body", None)

        if speed == "fast" and self._is_claude(params.get("model", owner.model_id)):
            params["speed"] = "fast"
        else:
            params["model"] = self._with_nitro(params.get("model", owner.model_id))
        return msgspec.structs.replace(request, params=params)

    @staticmethod
    def _is_claude(model_id: str) -> bool:
        return "anthropic/claude" in model_id.lower()

    @staticmethod
    def _can_apply_nitro(model_id: str) -> bool:
        model_name = model_id.rsplit("/", maxsplit=1)[-1]
        return ":" not in model_name or model_name.endswith(":nitro")

    @staticmethod
    def _with_nitro(model_id: str) -> str:
        if model_id.endswith(":nitro"):
            return model_id
        return f"{model_id}:nitro"


@register_model
class OpenRouterChatCompletion(_BaseOpenRouter, OpenAICompatibleChatCompletion):
    """OpenRouter Chat Completion."""

    chat_extensions = (OpenRouterSpeedExtension(),)

    capabilities = OpenAICompatibleChatCompletion.capabilities.replace(
        default_reasoning_codec=OpenRouterReasoningCodec(),
        reasoning_max_tokens=True,
    )

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        extra_body = dict(params.get("extra_body") or {})
        plugins = []

        store = params.pop("store", None)
        if store is not None:
            provider_preferences = dict(extra_body.get("provider") or {})
            provider_preferences["zdr"] = not store
            extra_body["provider"] = provider_preferences

        if params.get("tool_choice") is None:
            if params.get("tools") is not None:
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
