"""Context operations shared by compatible chat API modes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from msgflux.chat_messages import ChatMessages
from msgflux.models.chat_api import PreparedChatRequest
from msgflux.models.compaction import ContextTokenEstimate, ModelCompaction

if TYPE_CHECKING:
    from msgflux.tools.catalog import ToolCatalogView


def _payload_value(payload: Any, name: str) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(name)
    return getattr(payload, name, None)


class ChatContextAdapter:
    """Translate model context operations to and from one wire protocol."""

    api_mode: str | None = None

    def prepare_token_count(
        self,
        owner: Any,
        messages: ChatMessages | list[Mapping[str, Any]],
        *,
        system_prompt: str | None = None,
        tool_catalog: ToolCatalogView | None = None,
    ) -> PreparedChatRequest:
        raise NotImplementedError

    def decode_token_count(
        self,
        owner: Any,
        payload: Any,
    ) -> ContextTokenEstimate:
        raise NotImplementedError

    def prepare_compaction(
        self,
        owner: Any,
        messages: ChatMessages | list[Mapping[str, Any]],
        *,
        system_prompt: str | None = None,
    ) -> PreparedChatRequest:
        raise NotImplementedError

    def decode_compaction(self, owner: Any, payload: Any) -> ModelCompaction:
        raise NotImplementedError


class OpenAIResponsesContextAdapter(ChatContextAdapter):
    """Context operations exposed by the OpenAI-compatible Responses API."""

    api_mode = "responses"

    @staticmethod
    def _input(owner: Any, messages: Any) -> Any:
        if isinstance(messages, str):
            return messages
        if not isinstance(messages, ChatMessages):
            messages = ChatMessages(messages)
        return messages.to_responses_input(
            provider=owner.provider,
            api_mode=owner.api_mode,
            reasoning_codec=owner.reasoning_codec,
        )

    def prepare_token_count(
        self,
        owner: Any,
        messages: ChatMessages | list[Mapping[str, Any]],
        *,
        system_prompt: str | None = None,
        tool_catalog: ToolCatalogView | None = None,
    ) -> PreparedChatRequest:
        params = {
            "model": owner.model_id,
            "input": self._input(owner, messages),
            "instructions": system_prompt,
        }
        if tool_catalog and owner._catalog_tool_entries(tool_catalog):
            params["tools"] = owner._tools_to_responses(tool_catalog)
        return PreparedChatRequest(
            api=self.api_mode,
            endpoint="/responses/input_tokens",
            params=params,
        )

    def decode_token_count(
        self,
        owner: Any,
        payload: Any,
    ) -> ContextTokenEstimate:
        _ = owner
        return ContextTokenEstimate(
            input_tokens=int(_payload_value(payload, "input_tokens")),
            source="provider",
        )

    def prepare_compaction(
        self,
        owner: Any,
        messages: ChatMessages | list[Mapping[str, Any]],
        *,
        system_prompt: str | None = None,
    ) -> PreparedChatRequest:
        return PreparedChatRequest(
            api=self.api_mode,
            endpoint="/responses/compact",
            params={
                "model": owner.model_id,
                "input": self._input(owner, messages),
                "instructions": system_prompt,
            },
        )

    def decode_compaction(self, owner: Any, payload: Any) -> ModelCompaction:
        items = owner._serialize_openai_value(_payload_value(payload, "output"))
        usage = owner.usage_codec.normalize(_payload_value(payload, "usage"))
        return ModelCompaction(
            format="provider",
            items=items,
            provider=owner.provider,
            api_mode=owner.api_mode,
            model_id=owner.model_id,
            usage=dict(usage) if isinstance(usage, Mapping) else None,
        )


__all__ = ["ChatContextAdapter", "OpenAIResponsesContextAdapter"]
