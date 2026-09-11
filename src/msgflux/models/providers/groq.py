from os import getenv
from typing import Any, Dict

from msgflux.models.chat_capabilities import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
)
from msgflux.models.openai_compatible import (
    OpenAIChatCompletionsAPI,
    OpenAICompatibleChatCompletion,
    OpenAIResponsesAPI,
)
from msgflux.models.reasoning import (
    OpenAICompatibleReasoningCodec,
    TextResponsesReasoningCodec,
)
from msgflux.models.registry import register_model


class _BaseGroq:
    """Configurations to use Groq models."""

    provider: str = "groq"

    def _get_base_url(self):
        base_url = getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        if base_url is None:
            raise ValueError("Please set `GROQ_BASE_URL`")
        return base_url

    def _get_api_key(self):
        """Load API keys from environment variable."""
        key = getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "The Grok API key is not available. Please set `GROQ_API_KEY`"
            )
        return key


@register_model
class GroqChatCompletion(_BaseGroq, OpenAICompatibleChatCompletion):
    """Groq Chat Completion."""

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
                reasoning_codec=TextResponsesReasoningCodec(),
                request_reasoning_effort=True,
            ),
        ),
        default_reasoning_codec=OpenAICompatibleReasoningCodec(),
    )

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        extra_body = dict(params.get("extra_body") or {})
        params["max_completion_tokens"] = params.pop("max_tokens")

        tool_choice = params.get("tool_choice")
        if tool_choice is None:
            if params.get("tools") is not None:
                params["tool_choice"] = "auto"
            else:
                params["tool_choice"] = "none"

        if self.sampling_run_params.get("reasoning_effort", None):
            # GPT-OSS models use include_reasoning, Qwen uses reasoning_format
            if "gpt-oss" in self.model_id.lower():
                extra_body["include_reasoning"] = True
            else:
                extra_body["reasoning_format"] = "parsed"

        params["extra_body"] = extra_body
        return params
