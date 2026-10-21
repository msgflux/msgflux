"""Small model audit records stored alongside checkpointed chat items."""

from typing import Any, Mapping

import msgspec

from msgflux.models.configuration import ModelConfiguration


def minimal_model_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Select model audit fields that belong in durable chat history."""
    if not isinstance(metadata, Mapping):
        return None
    model = metadata.get("model")
    if not isinstance(model, Mapping):
        return None

    persisted: dict[str, Any] = {}
    for key in ("provider", "model_id", "api_mode", "reasoning_effort"):
        value = model.get(key)
        if isinstance(value, str) and value:
            persisted[key] = value
    for key in ("request_reasoning_effort", "conversation_reasoning_effort"):
        if key in model:
            value = model[key]
            if value is None or (isinstance(value, str) and value):
                persisted[key] = value
    return persisted or None


def _last_response_model(items) -> dict[str, Any] | None:
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
            return model
        # A newer response without identity makes an older model unsafe to suggest.
        return None
    return None


def model_configuration_from_history(items) -> ModelConfiguration | None:
    """Capture only the last response; never guess from constructor defaults."""
    model = _last_response_model(items)
    if model is None:
        return None
    return msgspec.convert(model, type=ModelConfiguration, strict=True)


def last_model_configuration(
    state: Mapping[str, Any] | None,
) -> ModelConfiguration | None:
    """Read a saved snapshot, or response metadata from an older checkpoint."""
    if not isinstance(state, Mapping):
        return None
    if "model_configuration" in state:
        saved = state["model_configuration"]
        return (
            None
            if saved is None
            else msgspec.convert(saved, type=ModelConfiguration, strict=True)
        )
    messages = state.get("messages")
    if not isinstance(messages, Mapping):
        return None
    return model_configuration_from_history(messages.get("items"))


def last_model_metadata(state: Mapping[str, Any] | None) -> dict[str, str] | None:
    """Read the last recorded model identity from a checkpoint snapshot."""
    if not isinstance(state, Mapping):
        return None
    messages = state.get("messages")
    if not isinstance(messages, Mapping):
        return None
    model = _last_response_model(messages.get("items"))
    if model is None:
        return None
    return {
        key: model[key] for key in ("provider", "model_id", "api_mode") if key in model
    }
