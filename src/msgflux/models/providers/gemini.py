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
from msgflux.models.reasoning import GeminiReasoningCodec
from msgflux.models.registry import register_model
from msgflux.version import __version__


class _BaseGemini(ProviderEnvBase):
    """Configurations for Google Gemini models."""

    provider: str = "gemini"
    display_name: str = "Gemini"
    api_key_env: str = "GEMINI_API_KEY"
    base_url_env: str = "GEMINI_BASE_URL"
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"


def _field(payload: Any, field: str, default: Any = None) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(field, default)
    return getattr(payload, field, default)


def _set_field(payload: Any, field: str, value: Any) -> None:
    if isinstance(payload, Mapping):
        payload[field] = value
    else:
        setattr(payload, field, value)


def _tool_call_parts(call: Any) -> tuple[Any, Any, Any]:
    function = _field(call, "function", None)
    return (
        _field(call, "id"),
        _field(function, "name") if function is not None else None,
        _field(function, "arguments") if function is not None else None,
    )


@register_model
class GeminiChatCompletion(_BaseGemini, OpenAICompatibleChatCompletion):
    """Gemini chat completion through the OpenAI-compatible endpoint.

    `POST /v1beta/openai/chat/completions`. Gemini exposes no Responses
    endpoint, so only `chat_completions` is declared. Reasoning effort is
    requested with top-level `reasoning_effort` (`low`/`medium`/`high`,
    plus `none`/`minimal` depending on the model); it cannot be combined
    with a custom ``extra_body.google.thinking_config``. Thought summaries
    are opt-in through that escape hatch and arrive inline inside
    `<thought>...</thought>` tags.

    Multi-turn reasoning hinges on opaque `thought_signature` values: the
    endpoint returns one on every assistant message (and one per tool
    call), and rejects tool continuations without them. Signatures are
    stored as reasoning/provider state and replayed back into
    `extra_content` on history conversion. The Interactions API
    (`previous_interaction_id` server state) is out of scope.
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
        default_reasoning_codec=GeminiReasoningCodec(),
    )

    def _adapt_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params = super()._adapt_params(params)
        extra_headers = dict(params.get("extra_headers") or {})
        extra_headers.setdefault("x-goog-api-client", f"msgflux-oai/{__version__}")
        params["extra_headers"] = extra_headers
        return params

    def _signature_state(self, signature: str) -> Dict[str, Any]:
        return {
            **self.reasoning_codec.state_identity(
                provider=self.provider,
                api_mode=self.api_mode,
            ),
            "data": {"extra_content": {"google": {"thought_signature": signature}}},
        }

    def _process_completion_model_output(
        self, model_output, generation_schema=None, transport_generation_schema=None
    ):
        choices = _field(model_output, "choices", []) or []
        message = _field(choices[0], "message") if choices else None
        if message is not None:
            content = _field(message, "content")
            if isinstance(content, str) and "<thought>" in content:
                summary, answer = GeminiReasoningCodec.split_thought_content(content)
                _set_field(message, "content", answer)
                if summary and not _field(message, "reasoning_content"):
                    _set_field(message, "reasoning_content", summary)
        response = super()._process_completion_model_output(
            model_output, generation_schema, transport_generation_schema
        )
        if message is not None:
            for call in _field(message, "tool_calls", None) or []:
                call_id, name, arguments = _tool_call_parts(call)
                item: Dict[str, Any] = {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments if arguments else "{}",
                }
                signature = GeminiReasoningCodec.thought_signature(call)
                if signature is not None:
                    item["provider_state"] = self._signature_state(signature)
                response.history_items.append(item)
        return response

    def _process_stream_tool_calls(self, delta, stream_response, aggregator):
        if stream_response.response_type is None:
            stream_response.set_response_type("tool_call")
        for tool_call in _field(delta, "tool_calls", None) or []:
            index = _field(tool_call, "index")
            call_id, name, arguments = _tool_call_parts(tool_call)
            aggregator.process(index, call_id, name, arguments)
            signature = GeminiReasoningCodec.thought_signature(tool_call)
            stream_response.chat_accumulator.add_tool_call_delta(
                index,
                call_id=call_id,
                name=name,
                arguments=arguments,
                provider=self.provider,
                api_mode=self.api_mode,
                provider_state=(
                    {"extra_content": {"google": {"thought_signature": signature}}}
                    if signature is not None
                    else None
                ),
            )

    @staticmethod
    def _split_stream_text(
        text: str, *, in_thought: bool
    ) -> tuple[str | None, str | None, bool]:
        """Split one thought-tagged delta into reasoning and answer text."""
        if "</thought>" in text:
            thought_part, _, answer_part = text.partition("</thought>")
            thought_part = thought_part.removeprefix("<thought>")
            return (thought_part or None, answer_part or None, False)
        if in_thought:
            return (text.removeprefix("<thought>") or None, None, True)
        return (None, text.removeprefix("</thought>") or None, False)

    def _split_gemini_chunk(self, chunk: Any, state: Dict[str, bool]) -> Any:
        choices = _field(chunk, "choices", []) or []
        if not choices:
            return [chunk]
        delta = _field(choices[0], "delta")
        if delta is None:
            return [chunk]
        content = _field(delta, "content")
        if not isinstance(content, str) or "<thought>" not in content:
            if not state["in_thought"]:
                return [chunk]
            if "</thought>" not in content:
                reasoning_chunk = deepcopy(chunk)
                reasoning_delta = _field(_field(reasoning_chunk, "choices")[0], "delta")
                _set_field(reasoning_delta, "content", None)
                _set_field(reasoning_delta, "reasoning_content", content or None)
                return [reasoning_chunk] if content else [chunk]
        if GeminiReasoningCodec.is_thought_delta(delta):
            state["in_thought"] = True
        thinking, answer, in_thought = self._split_stream_text(
            content, in_thought=state["in_thought"]
        )
        state["in_thought"] = in_thought
        expanded = []
        if thinking:
            reasoning_chunk = deepcopy(chunk)
            reasoning_delta = _field(_field(reasoning_chunk, "choices")[0], "delta")
            _set_field(reasoning_delta, "content", None)
            _set_field(reasoning_delta, "reasoning_content", thinking)
            expanded.append(reasoning_chunk)
        if answer:
            text_chunk = deepcopy(chunk) if thinking else chunk
            text_delta = _field(_field(text_chunk, "choices")[0], "delta")
            _set_field(text_delta, "content", answer)
            expanded.append(text_chunk)
        return expanded or [chunk]

    def _execute_model(self, **kwargs):
        output = super()._execute_model(**kwargs)
        if not kwargs.get("stream"):
            return output
        return self._expand_gemini_stream(output)

    def _expand_gemini_stream(self, chunks):
        state = {"in_thought": False}
        for chunk in chunks:
            yield from self._split_gemini_chunk(chunk, state)

    async def _aexecute_model(self, **kwargs):
        output = await super()._aexecute_model(**kwargs)
        if not kwargs.get("stream"):
            return output

        async def _aiter():
            state = {"in_thought": False}
            async for chunk in output:
                for item in self._split_gemini_chunk(chunk, state):
                    yield item

        return _aiter()
