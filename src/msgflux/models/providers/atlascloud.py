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


class _BaseAtlasCloud(ProviderEnvBase):
    """Configurations to use Atlas Cloud models."""

    provider: str = "atlascloud"
    display_name: str = "AtlasCloud"
    api_key_env: str = "ATLASCLOUD_API_KEY"
    base_url_env: str = "ATLASCLOUD_BASE_URL"
    base_url: str = "https://api.atlascloud.ai/v1"


@register_model
class AtlasCloudChatCompletion(_BaseAtlasCloud, OpenAICompatibleChatCompletion):
    """Atlas Cloud chat completion (multi-provider gateway).

    OpenAI-compatible `POST /v1/chat/completions` (default, widest model
    coverage) and `POST /v1/responses` on `https://api.atlascloud.ai/v1`.
    Responses notes from the gateway docs: no server-side state
    (`previous_response_id`/`store` ignored, full history is always sent),
    `reasoning.summary` and `include` are no-ops, so no reasoning codec or
    summary/encrypted flags are declared.
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
