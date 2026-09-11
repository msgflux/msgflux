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
