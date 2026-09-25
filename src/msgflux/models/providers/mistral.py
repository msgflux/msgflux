from copy import deepcopy
from typing import Any, Dict, Mapping

from msgflux.models.chat_capabilities import (
    ChatAPIModeCapabilities,
    ChatProviderCapabilities,
)
from msgflux.models.openai_compatible import (
    OpenAIChatCompletionsAPI,
    OpenAICompatibleChatCompletion,
)
from msgflux.models.provider_env import ProviderEnvBase
from msgflux.models.reasoning import MistralReasoningCodec
from msgflux.models.registry import register_model


class _BaseMistral(ProviderEnvBase):
    """Configurations for Mistral AI models."""

    provider: str = "mistral"
    display_name: str = "Mistral"
    api_key_env: str = "MISTRAL_API_KEY"
    base_url_env: str = "MISTRAL_BASE_URL"
    base_url: str = "https://api.mistral.ai/v1"


def _field(payload: Any, field: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(field, default)
    return getattr(payload, field, default)


def _set_field(payload: Any, field: str, value: Any) -> None:
    if isinstance(payload, Mapping):
        payload[field] = value
    else:
        setattr(payload, field, value)


@register_model
class MistralChatCompletion(_BaseMistral, OpenAICompatibleChatCompletion):
    """Mistral chat completion.

    `POST /v1/chat/completions` on `https://api.mistral.ai/v1`.
    Mistral does not expose an OpenAI Responses endpoint, so only
    `chat_completions` is declared. Reasoning is requested with top-level
    `reasoning_effort` (`none`, `minimal`, `low`, `medium`, `high`,
    `xhigh`); with reasoning enabled the assistant `content` is a list of
    `thinking`/`text` chunks instead of a plain string. Thinking traces
    are replayed as native `thinking` chunks in multi-turn history.
    """

    capabilities = ChatProviderCapabilities(
        default_api_mode="chat_completions",
        api_modes=(
            ChatAPIModeCapabilities(
                name="chat_completions",
                adapter=OpenAIChatCompletionsAPI(),
                request_reasoning_effort=True,
            ),
        ),
        default_reasoning_codec=MistralReasoningCodec(),
    )

    @staticmethod
    def _thinking_chunk(thinking: str) -> Dict[str, Any]:
        return {"type": "thinking", "thinking": [{"type": "text", "text": thinking}]}

    @classmethod
    def _with_mistral_reasoning(
        cls, message: Dict[str, Any], reasoning: str
    ) -> Dict[str, Any]:
        content = message.get("content")
        answer = content if isinstance(content, str) and content else None
        chunks = [cls._thinking_chunk(reasoning)]
        if answer:
            chunks.append({"type": "text", "text": answer})
        message["content"] = chunks
        return message

    @classmethod
    def _apply_mistral_history_reasoning(
        cls, messages: list[Dict[str, Any]]
    ) -> list[Dict[str, Any]]:
        for message in messages:
            if not isinstance(message, dict):
                continue
            reasoning = message.pop("reasoning_content", None)
            if (
                isinstance(reasoning, str)
                and reasoning
                and message.get("role") == "assistant"
            ):
                cls._with_mistral_reasoning(message, reasoning)
        return messages

    def _build_chat_completions_generation_params(
        self, messages, system_prompt, prefilling, tool_catalog, **kwargs
    ) -> Dict[str, Any]:
        params = super()._build_chat_completions_generation_params(
            messages, system_prompt, prefilling, tool_catalog, **kwargs
        )
        wire_messages = params.get("messages")
        if isinstance(wire_messages, list):
            params["messages"] = self._apply_mistral_history_reasoning(wire_messages)
        return params

    def _normalize_mistral_message(self, message: Any) -> None:
        thinking, text = MistralReasoningCodec.split_content(_field(message, "content"))
        if thinking is None and text is None:
            return
        _set_field(message, "content", text or "")
        if thinking and not _field(message, "reasoning_content"):
            _set_field(message, "reasoning_content", thinking)

    def _process_completion_model_output(
        self, model_output, generation_schema=None, transport_generation_schema=None
    ):
        choices = _field(model_output, "choices", []) or []
        if choices:
            self._normalize_mistral_message(_field(choices[0], "message"))
        return super()._process_completion_model_output(
            model_output, generation_schema, transport_generation_schema
        )

    def _expand_mistral_chunk(self, chunk: Any) -> Any:
        choices = _field(chunk, "choices", []) or []
        if not choices:
            return [chunk]
        delta = _field(choices[0], "delta")
        if delta is None:
            return [chunk]
        content = _field(delta, "content")
        if not isinstance(content, list):
            return [chunk]
        thinking, text = MistralReasoningCodec.split_content(content)
        if thinking is None and text is None:
            return [chunk]
        expanded = []
        if thinking:
            reasoning_chunk = deepcopy(chunk)
            reasoning_delta = _field(_field(reasoning_chunk, "choices")[0], "delta")
            _set_field(reasoning_delta, "content", None)
            _set_field(reasoning_delta, "reasoning_content", thinking)
            expanded.append(reasoning_chunk)
        if text:
            text_chunk = deepcopy(chunk) if thinking else chunk
            text_delta = _field(_field(text_chunk, "choices")[0], "delta")
            _set_field(text_delta, "content", text)
            if thinking:
                expanded.append(text_chunk)
            else:
                return [text_chunk]
        return expanded or [chunk]

    def _execute_model(self, **kwargs):
        output = super()._execute_model(**kwargs)
        if not kwargs.get("stream"):
            return output
        return (item for chunk in output for item in self._expand_mistral_chunk(chunk))

    async def _aexecute_model(self, **kwargs):
        output = await super()._aexecute_model(**kwargs)
        if not kwargs.get("stream"):
            return output

        async def _aiter():
            async for chunk in output:
                for item in self._expand_mistral_chunk(chunk):
                    yield item

        return _aiter()
