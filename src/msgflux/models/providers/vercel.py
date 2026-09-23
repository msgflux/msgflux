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
from msgflux.models.reasoning import (
    OpenAIResponsesReasoningCodec,
    VercelGatewayReasoningCodec,
)
from msgflux.models.registry import register_model


class _BaseVercel(ProviderEnvBase):
    """Configurations for Vercel AI Gateway models."""

    provider: str = "vercel"
    display_name: str = "Vercel AI Gateway"
    api_key_env: str = "AI_GATEWAY_API_KEY"
    base_url_env: str = "AI_GATEWAY_BASE_URL"
    base_url: str = "https://ai-gateway.vercel.sh/v1"


@register_model
class VercelChatCompletion(_BaseVercel, OpenAICompatibleChatCompletion):
    """Vercel AI Gateway chat completion.

    `POST /v1/chat/completions` on `https://ai-gateway.vercel.sh/v1`
    with `provider/model` identifiers (for example
    `vercel/anthropic/claude-opus-5`). Chat completions is the default;
    the gateway Responses surface (`POST /v1/responses`) is also declared
    with OpenAI-style reasoning, since the gateway mirrors the OpenAI
    reasoning contract on both surfaces. Reasoning effort is
    requested with top-level `reasoning_effort` (gateway alias for the
    `reasoning.effort` extension). The gateway normalizes reasoning to a
    `reasoning` text field plus an ordered `reasoning_details` array
    (signatures, encrypted payloads, summaries), replayed back on
    history conversion for multi-turn and tool flows.
    """

    capabilities = ChatProviderCapabilities(
        default_api_mode="chat_completions",
        api_modes=(
            ChatAPIModeCapabilities(
                name="chat_completions",
                adapter=OpenAIChatCompletionsAPI(),
                request_reasoning_effort=True,
            ),
            ChatAPIModeCapabilities(
                name="responses",
                adapter=OpenAIResponsesAPI(),
                reasoning_codec=OpenAIResponsesReasoningCodec(),
                request_reasoning_effort=True,
            ),
        ),
        default_reasoning_codec=VercelGatewayReasoningCodec(),
    )
