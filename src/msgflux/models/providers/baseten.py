from typing import Any, Dict

from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import register_model
from msgflux.models.session import merge_session_headers


class _BaseBaseten(ProviderEnvBase):
    """Configurations to use Baseten models."""

    provider: str = "baseten"
    display_name: str = "Baseten"
    api_key_env: str = "BASETEN_API_KEY"
    base_url_env: str = "BASETEN_BASE_URL"
    base_url: str = "https://inference.baseten.co/v1"


@register_model
class BasetenChatCompletion(_BaseBaseten, OpenAICompatibleChatCompletion):
    """Baseten chat completion.

    OpenAI-compatible `POST /v1/chat/completions` on
    `https://inference.baseten.co/v1`. Only `chat_completions` is declared.
    Reasoning models return clear-text `reasoning_content` (opt-in per
    model via `chat_template_args`, `reasoning_effort` table in docs).
    The active thread id is sent as `x-session-affinity` for sticky
    routing.
    """

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return merge_session_headers(params, "x-session-affinity")
