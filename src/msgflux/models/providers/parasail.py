from typing import Any, Dict

from msgflux.models.chat_capabilities import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
)
from msgflux.models.openai_compatible import (
    OpenAIChatCompletionsAPI,
    OpenAICompatibleChatCompletion,
    OpenAIResponsesAPI,
)
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.reasoning import OpenAICompatibleReasoningCodec
from msgflux.models.registry import register_model


class _BaseParasail(ProviderEnvBase):
    """Configurations to use Parasail models."""

    provider: str = "parasail"
    display_name: str = "Parasail"
    api_key_env: str = "PARASAIL_API_KEY"
    base_url_env: str = "PARASAIL_BASE_URL"
    base_url: str = "https://api.parasail.io/v1"


@register_model
class ParasailChatCompletion(_BaseParasail, OpenAICompatibleChatCompletion):
    """Parasail chat completion.

    `POST /v1/chat/completions` (default) and `POST /v1/responses` on
    `https://api.parasail.io/v1`. Responses notes from the gateway docs:
    `store` must be `false` on every request (forced here; the server
    errors otherwise) and there is no server-side state
    (`previous_response_id` unsupported, full history is always sent).
    No reasoning codec is declared: responses reasoning shapes are
    unverified. No top-level `reasoning_effort` either: reasoning controls
    are model-specific (`chat_template_kwargs.thinking` for DeepSeek,
    `enable_thinking` for Qwen, `thinking_budget`/`reasoning_effort` for
    GPT-OSS) and belong in `extra_body` per the model notes.
    """

    capabilities = ChatProviderCapabilities(
        default_api_mode="chat_completions",
        api_modes=(
            ChatAPIModeCapabilities(
                name="chat_completions",
                adapter=OpenAIChatCompletionsAPI(),
            ),
            ChatAPIModeCapabilities(
                name="responses",
                adapter=OpenAIResponsesAPI(),
            ),
        ),
        default_reasoning_codec=OpenAICompatibleReasoningCodec(),
    )

    def _adapt_responses_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = super()._adapt_responses_params(params)
        # The gateway rejects omitted or true `store`.
        params["store"] = False
        return params
