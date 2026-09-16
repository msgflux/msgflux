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


class _BaseTencentCloud(ProviderEnvBase):
    """Configurations to use Tencent Cloud MaaS models."""

    provider: str = "tencentcloud"
    display_name: str = "TencentCloud"
    api_key_env: str = "TENCENTCLOUD_API_KEY"
    base_url_env: str = "TENCENTCLOUD_BASE_URL"
    base_url: str = "https://tokenhub-intl.tencentcloudmaas.com/v1"


@register_model
class TencentCloudChatCompletion(_BaseTencentCloud, OpenAICompatibleChatCompletion):
    """Tencent Cloud MaaS chat completion.

    `POST /v1/chat/completions` (default) and `POST /v1/responses` on the
    TokenHub endpoint. Inference is bound to console service entitlements
    per key (unentitled models answer 401006); reasoning item shapes on
    responses are unverified, so no reasoning codec is declared there.
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
