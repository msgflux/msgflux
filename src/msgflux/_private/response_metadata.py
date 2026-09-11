from __future__ import annotations

from typing import Any, Mapping

from msgflux.chat_messages import ChatMessages


def minimal_usage_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, int] | None:
    """Select usage counters that belong in durable chat history."""
    if not isinstance(metadata, Mapping):
        return None
    usage = metadata.get("usage")
    if not isinstance(usage, Mapping):
        return None

    persisted: dict[str, int] = {}
    for source_key, target_key in (
        ("input_tokens", "input_tokens"),
        ("output_tokens", "output_tokens"),
    ):
        value = usage.get(source_key)
        if isinstance(value, int) and not isinstance(value, bool):
            persisted[target_key] = value

    input_details = usage.get("input_tokens_details")
    if isinstance(input_details, Mapping):
        cached_tokens = input_details.get("cached_tokens")
        if isinstance(cached_tokens, int) and not isinstance(cached_tokens, bool):
            persisted["cached_input_tokens"] = cached_tokens
    return persisted or None


def minimal_model_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Select model audit fields that belong in durable chat history."""
    if not isinstance(metadata, Mapping):
        return None
    model = metadata.get("model")
    if not isinstance(model, Mapping):
        return None

    persisted: dict[str, str] = {}
    for key in ("provider", "model_id", "api_mode", "reasoning_effort"):
        value = model.get(key)
        if isinstance(value, str) and value:
            persisted[key] = value
    return persisted or None


def attach_response_metadata(
    messages: ChatMessages | list[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None,
    *,
    after_index: int,
) -> None:
    """Attach durable response metadata to the last generated timeline item."""
    if not isinstance(messages, ChatMessages):
        return
    usage = minimal_usage_metadata(metadata)
    model = minimal_model_metadata(metadata)
    if usage is None and model is None:
        return

    for item in reversed(messages._items[after_index:]):
        if not (
            item.get("role") == "assistant"
            or item.get("type")
            in {
                "reasoning",
                "function_call",
                "tool_search_call",
                "tool_search_output",
            }
        ):
            continue
        item_metadata = item.setdefault("metadata", {})
        if not isinstance(item_metadata, dict):
            item_metadata = {}
            item["metadata"] = item_metadata
        if usage is not None:
            item_metadata["usage"] = usage
        if model is not None:
            item_metadata["model"] = model
        return
