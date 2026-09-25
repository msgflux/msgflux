"""Activate spans created by msgtrace's context managers."""

from contextlib import asynccontextmanager, contextmanager

from opentelemetry import trace


@contextmanager
def active_span(context):
    with context as span:
        with trace.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ):
            yield span


@asynccontextmanager
async def aactive_span(context):
    async with context as span:
        with trace.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ):
            yield span
