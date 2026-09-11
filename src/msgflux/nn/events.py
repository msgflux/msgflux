"""Execution event adapters for neural modules."""

import contextvars
from typing import Any, Mapping, Optional, Union

from msgflux._private.response_metadata import minimal_usage_metadata
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.events import EventType, emit_event


def _model_response_event_data(
    *,
    response_type: str | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the compact terminal payload exposed by ``model.response``."""
    event_data: dict[str, Any] = {"response_type": response_type}
    usage = minimal_usage_metadata(metadata)
    if usage is not None:
        event_data["usage"] = usage

    if not isinstance(metadata, Mapping):
        return event_data
    timing = metadata.get("timing")
    if not isinstance(timing, Mapping):
        return event_data

    selected_timing: dict[str, str | float] = {}
    source = timing.get("source")
    if isinstance(source, str) and source:
        selected_timing["source"] = source
    for key in ("latency_ms", "ttft_ms"):
        value = timing.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            selected_timing[key] = float(value)
    if selected_timing:
        event_data["timing"] = selected_timing
    return event_data


def emit_model_response_events(
    response: Union[ModelResponse, ModelStreamResponse],
    *,
    scope: Optional[ExecutionScope] = None,
) -> None:
    """Emit a terminal ``model.response`` with compact operational metadata."""
    if isinstance(response, ModelStreamResponse):
        event_context = contextvars.copy_context()

        def emit_terminal_response(final_state) -> None:
            data = _model_response_event_data(
                response_type=final_state.response_type,
                metadata=final_state.metadata,
            )
            event_context.run(
                lambda: emit_event(
                    EventType.MODEL_RESPONSE,
                    data,
                    scope=scope,
                )
            )

        response._add_consumer_finalizer(emit_terminal_response)
        return

    reasoning = getattr(response, "reasoning", None)
    if reasoning:
        emit_event(
            EventType.REASONING_DELTA,
            {"delta": reasoning},
            scope=scope,
        )
    reasoning_summary = getattr(response, "reasoning_summary", None)
    if reasoning_summary:
        emit_event(
            EventType.REASONING_SUMMARY_DELTA,
            {"delta": reasoning_summary},
            scope=scope,
        )
    emit_event(
        EventType.MODEL_RESPONSE,
        _model_response_event_data(
            response_type=response.response_type,
            metadata=response.metadata,
        ),
        scope=scope,
    )
