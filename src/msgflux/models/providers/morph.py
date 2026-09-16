from typing import Any, Dict

from msgflux.models.openai_compatible import OpenAICompatibleChatCompletion
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.registry import register_model
from msgflux.runtime.context import get_thread_id


class _BaseMorph(ProviderEnvBase):
    """Configurations to use Morph models."""

    provider: str = "morph"
    display_name: str = "Morph"
    api_key_env: str = "MORPH_API_KEY"
    base_url_env: str = "MORPH_BASE_URL"
    base_url: str = "https://api.morphllm.com/v1"


@register_model
class MorphChatCompletion(_BaseMorph, OpenAICompatibleChatCompletion):
    """Morph chat completion (open-weight codegen models).

    OpenAI-compatible `POST /v1/chat/completions` on
    `https://api.morphllm.com/v1`. Only `chat_completions` is declared.
    Reasoning models return clear-text `reasoning_content`. Prefix caching
    is automatic; the active thread id is sent as `prompt_cache_key` so
    multi-turn agents keep landing on the worker holding their cache.
    Kimi K3 dynamic tool loading (tools inside system messages) is a
    separate feature, out of scope here.
    """

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if params.get("prompt_cache_key") is None:
            thread_id = get_thread_id()
            if thread_id:
                params["prompt_cache_key"] = thread_id
        return params
