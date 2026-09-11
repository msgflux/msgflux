"""Composable request extensions for chat-model providers."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Literal

import msgspec

from msgflux.models.chat_api import PreparedChatRequest

ChatRequestOperation = Literal["generate", "warmup", "compact", "token_count"]
ChatSpeed = Literal["fast", "ultrafast", "nitro"]


class ChatRequestContext(msgspec.Struct, frozen=True, kw_only=True):
    """Stable information about the operation being prepared for transport."""

    operation: ChatRequestOperation
    api_mode: str


class ChatModelExtension:
    """Stateless provider plugin that transforms a prepared chat request."""

    name: str

    def configure(
        self,
        owner: Any,
        current: Any,
        changes: Mapping[str, Any],
    ) -> Any:
        """Return the extension configuration after applying ``changes``."""
        _ = owner
        if current is None:
            current = {}
        if not isinstance(current, Mapping):
            raise TypeError(
                f"Chat extension `{self.name}` must override `configure()` "
                "because its current configuration is not a mapping"
            )
        return {**current, **changes}

    def accepts(self, owner: Any, value: Any) -> bool:
        """Return whether this extension can represent a configuration value."""
        _ = owner, value
        return False

    def prepare_request(
        self,
        owner: Any,
        request: PreparedChatRequest,
        context: ChatRequestContext,
    ) -> PreparedChatRequest:
        """Return the request to send, optionally transformed."""
        _ = owner, context
        return request


class ChatSpeedExtension(ChatModelExtension):
    """Shared configuration contract for provider-specific speed extensions."""

    name = "speed"

    def configure(self, owner, current, changes: Mapping[str, Any]):
        unexpected = sorted(set(changes) - {"value"})
        if unexpected:
            joined = ", ".join(unexpected)
            raise TypeError(f"Unexpected speed configuration fields: {joined}")
        if "value" not in changes:
            raise TypeError("The speed extension requires a `value` field")
        speed = changes["value"]
        validate_chat_speed(speed)
        if speed is not None and not self.accepts(owner, speed):
            warnings.warn(
                f"Provider `{owner.provider}` does not support `speed={speed!r}` "
                f"for model `{owner.model_id}`; the setting was ignored.",
                UserWarning,
                stacklevel=3,
            )
            return current
        return speed


def validate_chat_speed(speed: str | None) -> None:
    """Validate the provider-independent speed preference."""
    if speed is None:
        return
    if not isinstance(speed, str):
        raise TypeError("`speed` must be a string or None")
    if speed not in {"fast", "ultrafast", "nitro"}:
        raise ValueError("`speed` must be one of: fast, ultrafast, nitro")
