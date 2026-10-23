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


class _BaseMeta:
    """Configurations to use Meta Model API models."""

    provider: str = "meta"
    api_key_env: str = "META_API_KEY"

    def _get_base_url(self):
        base_url = getenv("META_BASE_URL", "https://api.meta.ai/v1")
        if base_url is None:
            raise ValueError("Please set `META_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv(self.api_key_env)
        if not key:
            raise ValueError(
                f"The Meta API key is not available. Please set `{self.api_key_env}`"
            )
        return key


@register_model
class MetaChatCompletion(_BaseMeta, OpenAICompatibleChatCompletion):
    """Meta Model API chat completion (Muse Spark).

    OpenAI-compatible: `POST /v1/chat/completions` and `POST /v1/responses`
    on `https://api.meta.ai/v1` with a `MODEL_API_KEY` bearer token.
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
        # Meta rejects parallel sampling: only `n=1` is allowed.
        n = params.get("n", None)
        if n is not None and n != 1:
            raise ValueError(
                "Meta Model API only supports `n=1`; "
                f"parallel sampling with `n={n}` returns HTTP 400."
            )
        # Meta has no audio output: never send audio modalities.
        if params.get("modalities") is not None:
            raise ValueError(
                "Meta Model API does not support `modalities`; "
                "any value returns HTTP 400."
            )
        if params.get("audio") is not None:
            raise ValueError(
                "Meta Model API has no audio output; "
                "the `audio` parameter returns HTTP 400."
            )
        return params
