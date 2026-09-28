"""Small model audit records stored alongside checkpointed chat items."""

from typing import Any, Mapping


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


def last_model_metadata(state: Mapping[str, Any] | None) -> dict[str, str] | None:
    """Read the last recorded model response from a checkpoint snapshot."""
    if not isinstance(state, Mapping):
        return None
    messages = state.get("messages")
    if not isinstance(messages, Mapping):
        return None
    items = messages.get("items")
    if not isinstance(items, list):
        return None

    for item in reversed(items):
        if not isinstance(item, Mapping) or not (
            item.get("role") == "assistant"
            or item.get("type")
            in {"reasoning", "function_call", "tool_search_call", "tool_search_output"}
        ):
            continue
        model = minimal_model_metadata(item.get("metadata"))
        if model and model.get("provider") and model.get("model_id"):
            return {
                key: model[key]
                for key in ("provider", "model_id", "api_mode")
                if key in model
            }
        # A newer response without an identity makes an older model unsafe to
        # suggest as the one that produced the latest reply.
        return None
    return None
