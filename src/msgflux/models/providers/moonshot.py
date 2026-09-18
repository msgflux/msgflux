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


class _BaseMoonshot(ProviderEnvBase):
    """Configurations for Moonshot Kimi (global) models."""

    provider: str = "moonshot"
    display_name: str = "Moonshot"
    api_key_env: str = "MOONSHOT_API_KEY"
    base_url_env: str = "MOONSHOT_BASE_URL"
    base_url: str = "https://api.moonshot.ai/v1"


@register_model
class MoonshotChatCompletion(_BaseMoonshot, OpenAICompatibleChatCompletion):
    """Moonshot Kimi global pay-as-you-go chat completion.

    `POST /v1/chat/completions` (default) and `POST /v1/responses` on
    `https://api.moonshot.ai/v1`. Keys are region-scoped: a global key
    does not work on the China base (`api.moonshot.cn`) and vice versa.
    No reasoning codec is declared: responses reasoning shapes are
    unverified. Kimi Code subscription gateways and the Anthropic
    endpoint are out of scope.
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
                request_reasoning_effort=True,
            ),
        ),
        default_reasoning_codec=OpenAICompatibleReasoningCodec(),
    )
