from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
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
    base_url: str = "https://api.z.ai/api/coding/paas/v4"


@register_model
class ZAIChatCompletion(_BaseZAI, OpenAICompatibleChatCompletion):
    """Z.AI pay-as-you-go chat completion.

    OpenAI-compatible `POST /chat/completions` on
    `https://api.z.ai/api/paas/v4`. Only `chat_completions` is declared.
    """


class _BaseZhipu(ProviderEnvBase):
    """Configurations for ZhipuAI BigModel (China) pay-as-you-go models."""

    provider: str = "zhipu"
    display_name: str = "Zhipu"
    api_key_env: str = "ZHIPU_API_KEY"
    base_url_env: str = "ZHIPU_BASE_URL"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"


class _BaseZhipuCode(ProviderEnvBase):
    """Configurations for ZhipuAI BigModel (China) coding-plan models."""

    provider: str = "zhipu-code"
    display_name: str = "Zhipu Code"
    api_key_env: str = "ZHIPU_CODE_API_KEY"
    base_url_env: str = "ZHIPU_CODE_BASE_URL"
    base_url: str = "https://open.bigmodel.cn/api/coding/paas/v4"


@register_model
class ZhipuChatCompletion(_BaseZhipu, OpenAICompatibleChatCompletion):
    """ZhipuAI BigModel pay-as-you-go chat completion.

    OpenAI-compatible `POST /chat/completions` on
    `https://open.bigmodel.cn/api/paas/v4`. Only `chat_completions` is
    declared. Note the base path ends in `/paas/v4` (no `/v1` infix);
    endpoints append directly. Async/generic Zhipu APIs and the Anthropic
    endpoint are out of scope.
    """


@register_model
class ZhipuCodeChatCompletion(_BaseZhipuCode, OpenAICompatibleChatCompletion):
    """ZhipuAI BigModel coding-plan chat completion.

    OpenAI-compatible `POST /chat/completions` on
    `https://open.bigmodel.cn/api/coding/paas/v4`. Only `chat_completions`
    is declared. Domestic keys/plans are separate from Z.AI global ones.
    """


@register_model
class ZAICodeChatCompletion(_BaseZAICode, OpenAICompatibleChatCompletion):
    """Z.AI coding-plan chat completion.

    OpenAI-compatible `POST /chat/completions` on the coding path.
    Only `chat_completions` is declared: the Responses protocol lives on
    a different base (`https://api.z.ai/api/v1`), and per-protocol bases
    are not modeled. Coding-plan keys are not interchangeable with
    pay-as-you-go keys.
    """
