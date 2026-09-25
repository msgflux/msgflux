"""Useful span boundaries for the functional fan-out helpers."""

import asyncio
import threading

import pytest
from msgtrace.sdk.tracer import tracer_manager
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from msgflux.nn import functional as F


@pytest.fixture
def spans(monkeypatch):
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracer_manager, "_tracer", provider.get_tracer("test"))
    yield exporter
    provider.shutdown()


def test_scatter_gather_span_contains_work_and_reports_partial_failure(spans):
    def child():
        with tracer_manager.tracer.start_as_current_span("child"):
            return "done"

    def fail():
        raise ValueError("expected failure")

    with tracer_manager.tracer.start_as_current_span("root"):
        results = F.scatter_gather([child, fail], timeout=1)

    assert results[0] == "done"
    by_name = {span.name: span for span in spans.get_finished_spans()}
    fanout = by_name["scatter_gather"]
    assert fanout.parent.span_id == by_name["root"].context.span_id
    assert by_name["child"].parent.span_id == fanout.context.span_id
    assert fanout.attributes["msgflux.functional.task_count"] == 2
    assert fanout.attributes["msgflux.functional.timeout_seconds"] == 1
    assert fanout.attributes["msgflux.functional.failed_tasks"] == 1


def test_scatter_gather_parents_async_worker_span(spans):
    class AsyncChild:
        def __call__(self):
            raise AssertionError("sync path should not run")

        async def acall(self):
            with tracer_manager.tracer.start_as_current_span("async-child"):
                return "done"

    with tracer_manager.tracer.start_as_current_span("root"):
        assert F.scatter_gather([AsyncChild()]) == ("done",)

    by_name = {span.name: span for span in spans.get_finished_spans()}
    assert (
        by_name["async-child"].parent.span_id
        == by_name["scatter_gather"].context.span_id
    )


def test_wait_and_dispatch_helpers_do_not_add_spans(spans):
    event = threading.Event()
    event.set()
    completed = threading.Event()

    def work():
        with tracer_manager.tracer.start_as_current_span("work"):
            return 1

    with tracer_manager.tracer.start_as_current_span("root"):
        assert F.wait_for(work) == 1
        F.wait_for_event(event)
        F.detached(completed.set)
        assert completed.wait(2)

    by_name = {span.name: span for span in spans.get_finished_spans()}
    assert set(by_name) == {"root", "work"}
    assert by_name["work"].parent.span_id == by_name["root"].context.span_id


@pytest.mark.asyncio
async def test_async_bcast_span_covers_await_and_parents_child(spans):
    started = asyncio.Event()
    release = asyncio.Event()

    async def child():
        with tracer_manager.tracer.start_as_current_span("child"):
            started.set()
            await release.wait()
            return "done"

    with tracer_manager.tracer.start_as_current_span("root"):
        task = asyncio.create_task(F.abcast_gather([child]))
        await started.wait()
        assert not any(
            span.name == "abcast_gather" for span in spans.get_finished_spans()
        )
        release.set()
        assert await task == ("done",)

    by_name = {span.name: span for span in spans.get_finished_spans()}
    fanout = by_name["abcast_gather"]
    assert fanout.parent.span_id == by_name["root"].context.span_id
    assert by_name["child"].parent.span_id == fanout.context.span_id
    assert fanout.attributes["msgflux.functional.task_count"] == 1
    assert fanout.attributes["msgflux.functional.failed_tasks"] == 0
