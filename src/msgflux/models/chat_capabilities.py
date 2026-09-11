"""Declarative capabilities for compatible chat model providers."""

from typing import Any

import msgspec

from msgflux.models.chat_api import ChatAPIAdapter
from msgflux.models.chat_context import ChatContextAdapter
from msgflux.models.reasoning import ReasoningCodec


class ChatAPIModeCapabilities(msgspec.Struct, frozen=True, kw_only=True):
    """Capabilities and strategies associated with one chat API mode."""

    name: str
    adapter: ChatAPIAdapter
    reasoning_codec: ReasoningCodec | None = None
    reasoning_summary: bool = False
    encrypted_reasoning: bool = False
    request_reasoning_effort: bool = False
    context_adapter: ChatContextAdapter | None = None
    hosted_tool_search_model_families: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Chat API mode names cannot be empty")
        if not isinstance(self.adapter, ChatAPIAdapter):
            raise TypeError("`adapter` must be a ChatAPIAdapter instance")
        if self.reasoning_codec is not None and not isinstance(
            self.reasoning_codec, ReasoningCodec
        ):
            raise TypeError("`reasoning_codec` must be a ReasoningCodec instance")
        if self.context_adapter is not None and not isinstance(
            self.context_adapter, ChatContextAdapter
        ):
            raise TypeError("`context_adapter` must be a ChatContextAdapter instance")
        if (
            self.context_adapter is not None
            and self.context_adapter.api_mode is not None
            and self.context_adapter.api_mode != self.name
        ):
            raise ValueError(
                f"Context adapter for {self.context_adapter.api_mode!r} cannot be "
                f"attached to API mode {self.name!r}"
            )


class ChatProviderCapabilities(msgspec.Struct, frozen=True, kw_only=True):
    """Validated chat behavior declared by a concrete model provider."""

    default_api_mode: str
    api_modes: tuple[ChatAPIModeCapabilities, ...]
    default_reasoning_codec: ReasoningCodec
    init_logprobs: bool = False
    prompt_cache_retention: bool = False
    reasoning_max_tokens: bool = False
    uses_max_completion_tokens: bool = False

    def __post_init__(self) -> None:
        if not self.api_modes:
            raise ValueError("A chat provider must declare at least one API mode")
        if not all(
            isinstance(mode, ChatAPIModeCapabilities) for mode in self.api_modes
        ):
            raise TypeError(
                "`api_modes` must contain ChatAPIModeCapabilities instances"
            )
        if not isinstance(self.default_reasoning_codec, ReasoningCodec):
            raise TypeError(
                "`default_reasoning_codec` must be a ReasoningCodec instance"
            )
        names = tuple(mode.name for mode in self.api_modes)
        if len(names) != len(set(names)):
            raise ValueError("Chat provider API mode names must be unique")
        if self.default_api_mode not in names:
            raise ValueError(
                f"Default API mode {self.default_api_mode!r} is not declared; "
                f"available modes: {', '.join(names)}."
            )

    @property
    def supported_api_modes(self) -> tuple[str, ...]:
        return tuple(mode.name for mode in self.api_modes)

    def mode(self, name: str) -> ChatAPIModeCapabilities:
        for mode in self.api_modes:
            if mode.name == name:
                return mode
        raise KeyError(name)

    def replace(self, **changes: Any) -> "ChatProviderCapabilities":
        """Return a copy with explicit field replacements."""
        return msgspec.structs.replace(self, **changes)
