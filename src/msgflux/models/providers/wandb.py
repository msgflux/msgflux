from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import register_model


class _BaseWandB(ProviderEnvBase):
    """Configurations to use W&B Serverless Inference models."""

    provider: str = "wandb"
    display_name: str = "W&B"
    api_key_env: str = "WANDB_API_KEY"
    base_url_env: str = "WANDB_BASE_URL"
    base_url: str = "https://api.inference.wandb.ai/v1"


@register_model
class WandBChatCompletion(_BaseWandB, OpenAICompatibleChatCompletion):
    """W&B Serverless Inference chat completion.

    OpenAI-compatible `POST /v1/chat/completions` on
    `https://api.inference.wandb.ai/v1`. Only `chat_completions` is
    declared. Reasoning models return clear-text `reasoning` alongside
    content, which the default codec extracts into history.
    """
