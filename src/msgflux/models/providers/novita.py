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
    OpenAICompatibleReasoningCodec,
    OpenAIResponsesReasoningCodec,
)
from msgflux.models.registry import register_model


class _BaseNovita(ProviderEnvBase):
    """Configurations to use Novita models."""

    provider: str = "novita"
    display_name: str = "Novita"
    api_key_env: str = "NOVITA_API_KEY"
    base_url_env: str = "NOVITA_BASE_URL"
    # NOTE: the `/v1/` infix is required: chat works with or without it,
    # but `/responses` 404s without it.
    base_url: str = "https://api.novita.ai/openai/v1"


@register_model
class NovitaChatCompletion(_BaseNovita, OpenAICompatibleChatCompletion):
    """Novita chat completion.

    OpenAI-compatible `POST /chat/completions` and `POST /responses` on
    `https://api.novita.ai/openai`. Chat completions is the default:
    reasoning models return clear-text `reasoning_content`. Responses
    reasoning is OpenAI-shaped clear-text `summary` (not encrypted), so
    the OpenAI responses codec extracts and replays it.
    Anthropic `/messages` models are out of scope.
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
        default_reasoning_codec=OpenAICompatibleReasoningCodec(),
    )
