"""Provider-owned reasoning extraction and history encoding."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable, Mapping


class ReasoningCodec:
    """Translate provider reasoning payloads at the model boundary.

    ``ChatMessages`` stores provider-neutral reasoning items. A model selects a
    codec to extract reasoning from responses and to reconstruct the wire fields
    accepted by its API when that history is sent back.
    """

    name = "openai_compatible"
    text_fields = ("reasoning_content", "reasoning", "thinking")
    history_text_field = "reasoning_content"
    canonical_text_field = "text"
    state_field: str | None = None

    def canonical_text_field_for_item(self, payload: Any) -> str:
        """Select the canonical text field for one native reasoning item."""
        del payload
        return self.canonical_text_field

    def extract_text(self, payload: Any) -> str | None:
        for field in self.text_fields:
            value = self._get(payload, field)
            if isinstance(value, str) and value:
                return value
        return None

    def extract_state(
        self,
        payload: Any,
        *,
        serialize: Callable[[Any], Any],
    ) -> Any:
        if self.state_field is None:
            return None
        value = self._get(payload, self.state_field)
        return serialize(value) if value else None

    def encode_chat_message(
        self,
        items: Iterable[Mapping[str, Any]],
        *,
        provider: str,
        api_mode: str,
    ) -> dict[str, Any]:
        """Return reasoning fields to merge into a Chat Completions message."""
        # Parsed reasoning is safe to expose in msgFlux, but there is no common
        # Chat Completions contract for sending it back to a provider. Codecs
        # must opt into replay only when their provider documents one.
        del items, provider, api_mode
        return {}

    def encode_responses_item(
        self,
        item: Mapping[str, Any],
        *,
        provider: str,
        api_mode: str,
    ) -> dict[str, Any] | None:
        """Reconstruct a native Responses reasoning item when supported."""
        del item, provider, api_mode
        return None

    def state_identity(self, *, provider: str, api_mode: str) -> dict[str, str]:
        return {
            "provider": provider,
            "api_mode": api_mode,
            "codec": self.name,
        }

    def matches_state(
        self,
        state: Any,
        *,
        provider: str,
        api_mode: str,
    ) -> bool:
        if not isinstance(state, Mapping) or state.get("provider") != provider:
            return False
        stored_api_mode = state.get("api_mode")
        stored_codec = state.get("codec")
        return (stored_api_mode is None or stored_api_mode == api_mode) and (
            stored_codec is None or stored_codec == self.name
        )

    @staticmethod
    def _get(payload: Any, field: str) -> Any:
        if isinstance(payload, Mapping):
            return payload.get(field)
        return getattr(payload, field, None)

    @classmethod
    def _item_text(cls, item: Mapping[str, Any]) -> str | None:
        for field in ("text", "reasoning_content", "reasoning_text", "think"):
            value = item.get(field)
            if isinstance(value, str) and value:
                return value
        summary = item.get("summary")
        return summary if isinstance(summary, str) and summary else None


class OpenAICompatibleReasoningCodec(ReasoningCodec):
    """Common parsed-text convention used by OpenAI-compatible endpoints."""

    # OpenAI-compatible servers agree on how to return parsed reasoning much
    # more often than on how to replay it. Keep the default extract-only.


class OllamaReasoningCodec(OpenAICompatibleReasoningCodec):
    """Ollama's clear-text ``thinking`` field for native chat history."""

    name = "ollama_thinking"
    history_text_field = "thinking"

    def encode_chat_message(
        self,
        items: Iterable[Mapping[str, Any]],
        *,
        provider: str,
        api_mode: str,
    ) -> dict[str, Any]:
        del provider, api_mode
        chunks = [text for item in items if (text := self._item_text(item))]
        return {self.history_text_field: "".join(chunks)} if chunks else {}


class OpenAIReasoningCodec(OpenAICompatibleReasoningCodec):
    """Reasoning convention for OpenAI Chat Completions."""

    name = "openai_chat_completions"
    state_field = None


class OpenAIResponsesReasoningCodec(ReasoningCodec):
    """Reasoning summaries and opaque items from the OpenAI Responses API."""

    name = "openai_responses"
    history_text_field = "reasoning_content"
    canonical_text_field = "summary"

    def extract_text(self, payload: Any) -> str | None:
        summary = self._get(payload, "summary")
        if not isinstance(summary, list):
            return None
        chunks: list[str] = []
        for part in summary:
            text = self._get(part, "text")
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks) or None

    def extract_state(
        self,
        payload: Any,
        *,
        serialize: Callable[[Any], Any],
    ) -> Any:
        if self._get(payload, "type") != "reasoning":
            return None
        state = serialize(payload)
        if isinstance(state, Mapping):
            state = dict(state)
            state.pop("summary", None)
        return state

    def encode_responses_item(
        self,
        item: Mapping[str, Any],
        *,
        provider: str,
        api_mode: str,
    ) -> dict[str, Any] | None:
        state = item.get("provider_state")
        if not self.matches_state(state, provider=provider, api_mode=api_mode):
            return None
        data = state.get("data")
        if not isinstance(data, Mapping):
            return None
        response_item = deepcopy(dict(data))
        summary = item.get("summary")
        if "summary" not in response_item and isinstance(summary, str) and summary:
            response_item["summary"] = [{"type": "summary_text", "text": summary}]
        response_item.setdefault("summary", [])
        return response_item


class TextResponsesReasoningCodec(ReasoningCodec):
    """Clear-text reasoning items used by Responses-compatible providers."""

    name = "responses_reasoning_text"

    def extract_text(self, payload: Any) -> str | None:
        content = self._get(payload, "content")
        if not isinstance(content, list):
            return None
        chunks: list[str] = []
        for part in content:
            if self._get(part, "type") != "reasoning_text":
                continue
            text = self._get(part, "text")
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks) or None

    def extract_state(
        self,
        payload: Any,
        *,
        serialize: Callable[[Any], Any],
    ) -> Any:
        if self._get(payload, "type") != "reasoning":
            return None
        state = serialize(payload)
        if isinstance(state, Mapping):
            state = dict(state)
            state.pop("content", None)
            state.pop("summary", None)
        return state

    def encode_responses_item(
        self,
        item: Mapping[str, Any],
        *,
        provider: str,
        api_mode: str,
    ) -> dict[str, Any] | None:
        state = item.get("provider_state")
        if not self.matches_state(state, provider=provider, api_mode=api_mode):
            return None
        data = state.get("data")
        if not isinstance(data, Mapping):
            return None
        response_item = deepcopy(dict(data))
        text = self._item_text(item)
        if text:
            response_item["content"] = [{"type": "reasoning_text", "text": text}]
        response_item.setdefault("summary", [])
        return response_item


class GeminiInteractionsReasoningCodec(OpenAICompatibleReasoningCodec):
    """Native Gemini Interactions thought steps.

    The Interactions API returns reasoning as ordered ``thought`` steps with
    an optional summary content list and an opaque ``signature``. Signatures
    must be replayed verbatim on later turns, so the complete native step is
    retained as provider state instead of being reconstructed from text.
    Streaming deltas mark the end of a thought step with a ``thought_step``
    payload carrying that same native step.
    """

    name = "gemini_interactions"

    @staticmethod
    def thought_step(payload: Any) -> Mapping[str, Any] | None:
        step = ReasoningCodec._get(payload, "thought_step")
        if isinstance(step, Mapping) and step.get("type") == "thought":
            return step
        if isinstance(payload, Mapping) and payload.get("type") == "thought":
            return payload
        return None

    @classmethod
    def summary_text(cls, step: Any) -> str | None:
        summary = ReasoningCodec._get(step, "summary")
        if isinstance(summary, str):
            return summary or None
        if not isinstance(summary, list):
            return None
        chunks: list[str] = []
        for part in summary:
            if ReasoningCodec._get(part, "type") != "text":
                continue
            text = ReasoningCodec._get(part, "text")
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks) or None

    @staticmethod
    def signature(step: Any) -> str | None:
        signature = ReasoningCodec._get(step, "signature")
        return signature if isinstance(signature, str) and signature else None

    def extract_text(self, payload: Any) -> str | None:
        step = self.thought_step(payload)
        if step is not None and ReasoningCodec._get(payload, "type") == "thought":
            return self.summary_text(step)
        return super().extract_text(payload)

    def extract_state(
        self,
        payload: Any,
        *,
        serialize: Callable[[Any], Any],
    ) -> Any:
        step = self.thought_step(payload)
        if step is None:
            return None
        state = serialize(step)
        return dict(state) if isinstance(state, Mapping) else None


class OpenRouterResponsesReasoningCodec(OpenAIResponsesReasoningCodec):
    """Preserve opaque Responses reasoning and classify clear text by item."""

    name = "openrouter_responses"

    def extract_text(self, payload: Any) -> str | None:
        summary = super().extract_text(payload)
        return summary or TextResponsesReasoningCodec.extract_text(self, payload)

    def canonical_text_field_for_item(self, payload: Any) -> str:
        return "summary" if super().extract_text(payload) else "text"

    def extract_state(
        self,
        payload: Any,
        *,
        serialize: Callable[[Any], Any],
    ) -> Any:
        if self._get(payload, "type") != "reasoning":
            return None
        return serialize(payload)

    def encode_responses_item(
        self,
        item: Mapping[str, Any],
        *,
        provider: str,
        api_mode: str,
    ) -> dict[str, Any] | None:
        state = item.get("provider_state")
        if not self.matches_state(state, provider=provider, api_mode=api_mode):
            return None
        data = state.get("data")
        return deepcopy(dict(data)) if isinstance(data, Mapping) else None


class OpenRouterReasoningCodec(OpenAICompatibleReasoningCodec):
    """OpenRouter reasoning text plus its ordered opaque detail blocks."""

    name = "openrouter_reasoning_details"
    state_field = "reasoning_details"

    def encode_chat_message(
        self,
        items: Iterable[Mapping[str, Any]],
        *,
        provider: str,
        api_mode: str,
    ) -> dict[str, Any]:
        items = list(items)
        chunks = [text for item in items if (text := self._item_text(item))]
        encoded = {self.history_text_field: "".join(chunks)} if chunks else {}
        details: list[Any] = []
        for item in items:
            state = item.get("provider_state")
            if not self.matches_state(state, provider=provider, api_mode=api_mode):
                continue
            data = state.get("data")
            if isinstance(data, list):
                details.extend(deepcopy(data))
            elif data is not None:
                details.append(deepcopy(data))
        if details:
            encoded["reasoning_details"] = details
        return encoded


class AnthropicThinkingCodec(OpenAICompatibleReasoningCodec):
    """Anthropic thinking/redacted blocks in the Messages API.

    History items carry complete blocks as state so multi-turn replay is
    verbatim (text, signatures, and order preserved, as the API requires;
    modified blocks are rejected). `encode_chat_message` rebuilds the
    ordered block list for the wire converter to prepend to the assistant
    content.
    """

    name = "anthropic_thinking"
    thinking_block_types = ("thinking", "redacted_thinking")

    def extract_state(
        self,
        payload: Any,
        *,
        serialize: Callable[[Any], Any],
    ) -> Any:
        del serialize
        block = self._get(payload, "thinking_block")
        if isinstance(block, Mapping) and block.get("type") in (
            self.thinking_block_types
        ):
            return deepcopy(dict(block))
        return None

    def encode_chat_message(
        self,
        items: Iterable[Mapping[str, Any]],
        *,
        provider: str,
        api_mode: str,
    ) -> dict[str, Any]:
        blocks: list[Any] = []
        for item in items:
            state = item.get("provider_state")
            if not self.matches_state(state, provider=provider, api_mode=api_mode):
                continue
            data = state.get("data")
            if isinstance(data, Mapping) and data.get("type") in (
                self.thinking_block_types
            ):
                blocks.append(deepcopy(dict(data)))
        return {"thinking_blocks": blocks} if blocks else {}
