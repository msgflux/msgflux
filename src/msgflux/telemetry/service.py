"""Trace detached service attempts without carrying frontend ContextVars."""

import asyncio
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Link, SpanContext, SpanKind, StatusCode


def execution_trigger() -> SpanContext | None:
    """Retain only the caller's immutable trace identity, never its context."""
    context = trace.get_current_span().get_span_context()
    return context if context.is_valid else None


@contextmanager
def execution_span(receipt, agent_id: str, trigger: SpanContext | None):
    """Create one independent trace for a claimed execution attempt."""
    with trace.get_tracer(__name__).start_as_current_span(
        "msgflux.service.execute",
        context=Context(),
        kind=SpanKind.CONSUMER,
        links=[Link(trigger)] if trigger is not None else [],
        attributes={
            "msgflux.agent_id": agent_id,
            "msgflux.thread_id": receipt.thread_id,
            "msgflux.run_id": receipt.run_id,
            "msgflux.request_id": receipt.request_id,
            "msgflux.admission.revision": receipt.revision,
        },
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except asyncio.CancelledError:
            finish_execution_span(span, "interrupted", "CancelledError")
            raise
        except BaseException as exc:
            # Exception messages may contain prompts, paths or credentials.
            finish_execution_span(span, "failed", type(exc).__name__)
            raise


def finish_execution_span(span, status: str, error_type: str | None = None) -> None:
    span.set_attribute("msgflux.run.status", status)
    if error_type:
        key = "error.type" if status == "failed" else "msgflux.run.stop_reason"
        span.set_attribute(key, error_type)
    if status == "failed":
        span.set_status(StatusCode.ERROR)
    elif status == "completed":
        span.set_status(StatusCode.OK)
