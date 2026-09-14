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
from msgflux.models.providers._session import merge_session_headers
from msgflux.models.reasoning import (
    OpenAICompatibleReasoningCodec,
    OpenAIResponsesReasoningCodec,
)
from msgflux.models.registry import register_model


class _BaseOpenCode:
    """Shared configuration for the OpenCode Zen and Go gateways.

    Both gateways accept the same API key; billing (pay-per-use vs.
    subscription) is resolved server-side from the base URL.
    """

    session_header = "x-opencode-session"

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("OPENCODE_API_KEY")
        if not key:
            raise ValueError(
                "The OpenCode API key is not available. Please set `OPENCODE_API_KEY`"
            )
        return key

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return merge_session_headers(params, self.session_header)

    def _adapt_responses_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = super()._adapt_responses_params(params)
        return merge_session_headers(params, self.session_header)


class _BaseOpenCodeZen(_BaseOpenCode):
    """Configurations for OpenCode Zen (pay-per-use gateway)."""

    provider: str = "opencode"

    def _get_base_url(self):
        base_url = getenv("OPENCODE_BASE_URL", "https://opencode.ai/zen/v1")
        if base_url is None:
            raise ValueError("Please set `OPENCODE_BASE_URL`")
        return base_url


class _BaseOpenCodeGo(_BaseOpenCode):
    """Configurations for OpenCode Go (subscription gateway)."""

    provider: str = "opencode-go"

    def _get_base_url(self):
        base_url = getenv("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")
        if base_url is None:
            raise ValueError("Please set `OPENCODE_GO_BASE_URL`")
        return base_url


_OPENCODE_CAPABILITIES = ChatProviderCapabilities(
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


@register_model
class OpenCodeChatCompletion(_BaseOpenCodeZen, OpenAICompatibleChatCompletion):
    """OpenCode Zen chat completion (pay-per-use gateway).

    OpenAI-compatible `POST /v1/responses` and `POST /v1/chat/completions`
    on `https://opencode.ai/zen/v1`. Only models served on those two
    endpoints are supported; `/messages` (Anthropic) and `/models/*`
    (Google) endpoints are out of scope.
    """

    capabilities = _OPENCODE_CAPABILITIES


@register_model
class OpenCodeGoChatCompletion(_BaseOpenCodeGo, OpenAICompatibleChatCompletion):
    """OpenCode Go chat completion (subscription gateway).

    Same OpenAI-compatible protocols as Zen, served from
    `https://opencode.ai/zen/go/v1` with the same API key.
    """

    capabilities = _OPENCODE_CAPABILITIES
