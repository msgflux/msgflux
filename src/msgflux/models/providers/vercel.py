from msgflux.models.chat_capabilities import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
)
from msgflux.models.openai_compatible import (
    OpenAIChatCompletionsAPI,
    OpenAICompatibleChatCompletion,
)
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.reasoning import VercelGatewayReasoningCodec
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
    `vercel/anthropic/claude-opus-5`). Only `chat_completions` is
    declared; the gateway Responses surface is left for a follow-up
    until its reasoning wire shape is verified live. Reasoning effort is
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
        ),
        default_reasoning_codec=VercelGatewayReasoningCodec(),
    )
