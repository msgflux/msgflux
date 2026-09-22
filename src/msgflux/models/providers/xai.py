from os import getenv
from typing import Any, Dict

from msgflux.models.chat_capabilities import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
)
from msgflux.models.chat_context import OpenAIResponsesContextAdapter
from msgflux.models.openai_compatible import (
    OpenAIChatCompletionsAPI,
    OpenAICompatibleChatCompletion,
    OpenAIResponsesAPI,
)
from msgflux.models.reasoning import (
    OpenAICompatibleReasoningCodec,
    OpenAIResponsesReasoningCodec,
)
from msgflux.models.registry import register_model
from msgflux.models.session import USER_AGENT, merge_session_headers
from msgflux.runtime.context import get_thread_id


class _BaseXAI:
    """Configurations to use xAI Grok models."""

    provider: str = "xai"
    api_key_env: str = "XAI_API_KEY"

    def _get_base_url(self):
        base_url = getenv("XAI_BASE_URL", "https://api.x.ai/v1")
        if base_url is None:
            raise ValueError("Please set `XAI_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv(self.api_key_env)
        if not key:
            raise ValueError(
                f"The xAI API key is not available. Please set `{self.api_key_env}`"
            )
        return key


@register_model
class XAIChatCompletion(_BaseXAI, OpenAICompatibleChatCompletion):
    """xAI Grok chat completion.

    OpenAI-compatible: `POST /v1/chat/completions` (legacy) and
    `POST /v1/responses` (recommended) on `https://api.x.ai/v1` with an
    `XAI_API_KEY` bearer token.
    """

    capabilities = ChatProviderCapabilities(
        default_api_mode="responses",
        api_modes=(
            ChatAPIModeCapabilities(
                name="responses",
                adapter=OpenAIResponsesAPI(),
                reasoning_codec=OpenAIResponsesReasoningCodec(),
                reasoning_summary=True,
                encrypted_reasoning=True,
                request_reasoning_effort=True,
                context_adapter=OpenAIResponsesContextAdapter(),
            ),
            ChatAPIModeCapabilities(
                name="chat_completions",
                adapter=OpenAIChatCompletionsAPI(),
                request_reasoning_effort=True,
            ),
        ),
        default_reasoning_codec=OpenAICompatibleReasoningCodec(),
    )

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return merge_session_headers(params, "x-grok-conv-id")

    def _adapt_responses_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = super()._adapt_responses_params(params)
        thread_id = get_thread_id()
        if thread_id and params.get("prompt_cache_key") is None:
            params["prompt_cache_key"] = thread_id
        extra_headers = dict(params.get("extra_headers") or {})
        extra_headers.setdefault("User-Agent", USER_AGENT)
        params["extra_headers"] = extra_headers
        return params
