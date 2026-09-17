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


class _BaseZAI(ProviderEnvBase):
    """Configurations for Z.AI pay-as-you-go models."""

    provider: str = "zai"
    display_name: str = "Z.AI"
    api_key_env: str = "ZAI_API_KEY"
    base_url_env: str = "ZAI_BASE_URL"
    base_url: str = "https://api.z.ai/api/paas/v4"


class _BaseZAICode(ProviderEnvBase):
    """Configurations for Z.AI coding-plan models."""

    provider: str = "zai-code"
    display_name: str = "Z.AI Code"
    api_key_env: str = "ZAI_CODE_API_KEY"
    base_url_env: str = "ZAI_CODE_BASE_URL"
    base_url: str = "https://api.z.ai/api/v1"


@register_model
class ZAIChatCompletion(_BaseZAI, OpenAICompatibleChatCompletion):
    """Z.AI pay-as-you-go chat completion.

    OpenAI-compatible `POST /chat/completions` on
    `https://api.z.ai/api/paas/v4`. Only `chat_completions` is declared.
    """


@register_model
class ZAICodeChatCompletion(_BaseZAICode, OpenAICompatibleChatCompletion):
    """Z.AI coding-plan chat completion.

    `POST /chat/completions` (default, keeps reasoning visible) and
    `POST /responses` (required by codex-cli) on
    `https://api.z.ai/api/v1`. No reasoning codec is declared: responses
    reasoning shapes are unverified. Coding-plan keys are not
    interchangeable with pay-as-you-go keys.
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
