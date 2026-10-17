"""Detached attempt traces preserve execution ownership and isolation."""

import asyncio
from contextvars import ContextVar
from unittest.mock import AsyncMock

import pytest
from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from tests.test_agent_service import _agent, _response
from tests.test_agent_service_approvals import _fixture


@pytest.fixture
def traces(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer_provider", lambda: provider)
    yield provider.get_tracer("test"), exporter
    provider.shutdown()


def _attempts(exporter):
    return [
        s for s in exporter.get_finished_spans() if s.name == "msgflux.service.execute"
    ]


async def _close(service):
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
async def test_parallel_attempts_link_callers_without_copying_frontend_context(traces):
    tracer, exporter = traces
    private = ContextVar("frontend-private", default=None)
    service = AgentService(store=SQLiteServiceStore())
    entered = 0
    both = asyncio.Event()
    contexts = []

    def factory(_thread):
        agent = _agent()

        async def respond(**_kwargs):
            nonlocal entered
            contexts.append((private.get(), baggage.get_baggage("private")))
            entered += 1
            if entered == 2:
                both.set()
            await asyncio.wait_for(both.wait(), 3)
            with tracer.start_as_current_span("model-child"):
                await asyncio.sleep(0)
            return _response()

        agent.generator.aforward = AsyncMock(side_effect=respond)
        return AgentSession(agent)

    service.register("main", factory)
    parents = {}
    try:
        for name in ("one", "two"):
            await service.open_thread("main", thread_id=name)
            with tracer.start_as_current_span(f"caller-{name}") as parent:
                parents[name] = parent.get_span_context()
                token = private.set("secret")
                bag = context.attach(baggage.set_baggage("private", "secret"))
                try:
                    await service.prompt(name, "private prompt", request_id=name)
                finally:
                    context.detach(bag)
                    private.reset(token)
        await asyncio.wait_for(
            asyncio.gather(service.wait("one", "one"), service.wait("two", "two")), 5
        )
        attempts = _attempts(exporter)
        assert len(attempts) == 2
        assert len({s.context.trace_id for s in attempts}) == 2
        assert contexts == [(None, None), (None, None)]
        for span in attempts:
            thread = span.attributes["msgflux.thread_id"]
            assert span.parent is None
            assert span.kind == SpanKind.CONSUMER
            assert span.links[0].context == parents[thread]
            assert span.context.trace_id != parents[thread].trace_id
            assert span.attributes["msgflux.agent_id"] == "main"
            assert span.attributes["msgflux.request_id"] == thread
            assert span.attributes["msgflux.run.status"] == "completed"
            assert span.status.status_code == StatusCode.OK
            assert "private prompt" not in str(span.attributes)
            child = next(
                s
                for s in exporter.get_finished_spans()
                if s.name == "model-child"
                and s.context.trace_id == span.context.trace_id
            )
            assert child.parent.span_id == span.context.span_id
        assert not trace.get_current_span().get_span_context().is_valid
    finally:
        await _close(service)


@pytest.mark.asyncio
async def test_pause_and_resume_have_distinct_attempts_and_trigger_links(traces):
    tracer, exporter = traces
    service, _agent_obj, _approvals, calls, _options = _fixture()
    thread = await service.open_thread("main")
    try:
        with tracer.start_as_current_span("prompt") as prompt:
            receipt = await service.prompt(thread.thread_id, "lookup", request_id="req")
        paused = await service.wait(thread.thread_id, "req")
        assert paused.status == "paused"
        assert calls == []
        review = (await service.approval_reviews(thread.thread_id, receipt.run_id))[0]
        await service.decide_approval(
            thread.thread_id,
            receipt.run_id,
            review.request_id,
            approved=True,
            expected_revision=review.revision,
        )
        with tracer.start_as_current_span("resume") as resume:
            await service.resume(thread.thread_id, "req")
        assert (await service.wait(thread.thread_id, "req")).status == "completed"
        first, second = _attempts(exporter)
        assert first.links[0].context == prompt.get_span_context()
        assert second.links[0].context == resume.get_span_context()
        assert first.context.trace_id != second.context.trace_id
        assert first.attributes["msgflux.run.status"] == "paused"
        assert first.status.status_code == StatusCode.UNSET
        assert second.attributes["msgflux.run.status"] == "completed"
        assert first.attributes["msgflux.run_id"] == second.attributes["msgflux.run_id"]
        assert (
            first.attributes["msgflux.request_id"]
            == second.attributes["msgflux.request_id"]
        )
        assert (
            first.attributes["msgflux.admission.revision"]
            < second.attributes["msgflux.admission.revision"]
        )
        assert len(calls) == 1
    finally:
        await _close(service)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["complete", "fail", "cancel", "interrupt", "journal-fail"]
)
async def test_attempt_lifecycle_and_duplicate_admission(traces, monkeypatch, mode):
    _tracer, exporter = traces
    service = AgentService(store=SQLiteServiceStore())
    agent = _agent()
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute(_record, _worker):
        started.set()
        await release.wait()
        if mode == "fail":
            raise ValueError("secret exception body")
        return "interrupted" if mode == "interrupt" else "completed"

    monkeypatch.setattr(service, "_execute", execute)
    if mode == "journal-fail":

        def fail_finish(*_args):
            raise OSError("secret database path")

        monkeypatch.setattr(service.store, "finish", fail_finish)
    service.register("main", lambda _: AgentSession(agent))
    thread = await service.open_thread("main")
    try:
        receipt = await service.prompt(thread.thread_id, "private", request_id="req")
        await asyncio.wait_for(started.wait(), 3)
        assert not _attempts(exporter)
        assert await service.prompt(thread.thread_id, "private", request_id="req")
        worker = service._workers[(thread.thread_id, "req")]
        if mode == "cancel":
            worker.task.cancel()
        release.set()
        await asyncio.gather(worker.task, return_exceptions=True)
        (span,) = _attempts(exporter)
        expected = {
            "complete": "completed",
            "fail": "failed",
            "cancel": "interrupted",
            "interrupt": "interrupted",
            "journal-fail": "failed",
        }[mode]
        assert span.attributes["msgflux.run.status"] == expected
        assert span.parent is None
        assert span.links == ()
        assert span.attributes["msgflux.run_id"] == receipt.run_id
        assert span.status.status_code == (
            StatusCode.ERROR
            if expected == "failed"
            else StatusCode.OK
            if expected == "completed"
            else StatusCode.UNSET
        )
        assert not span.events
        assert "secret" not in str(span.attributes)
        assert "secret" not in str(span.status.description)
        assert not service._workers
        assert not trace.get_current_span().get_span_context().is_valid
    finally:
        release.set()
        await _close(service)


@pytest.mark.asyncio
async def test_interrupt_api_settles_span_without_an_error(traces):
    _tracer, exporter = traces
    entered = asyncio.Event()
    release = asyncio.Event()
    agent = _agent()

    async def delayed(**_kwargs):
        entered.set()
        await release.wait()
        return _response()

    agent.generator.aforward = AsyncMock(side_effect=delayed)
    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _: AgentSession(agent))
    thread = await service.open_thread("main")
    try:
        receipt = await service.prompt(thread.thread_id, "hello", request_id="req")
        await asyncio.wait_for(entered.wait(), 3)
        assert await service.interrupt(thread.thread_id, receipt.run_id)
        release.set()
        assert (await service.wait(thread.thread_id, "req")).status == "interrupted"
        (span,) = _attempts(exporter)
        assert span.attributes["msgflux.run.status"] == "interrupted"
        assert span.status.status_code == StatusCode.UNSET
        assert "error.type" not in span.attributes
    finally:
        release.set()
        await _close(service)


@pytest.mark.asyncio
async def test_unclaimed_admission_creates_no_execution_span(traces, monkeypatch):
    _tracer, exporter = traces
    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _: AgentSession(_agent()))
    thread = await service.open_thread("main")
    monkeypatch.setattr(service.store, "claim", lambda *_: False)
    try:
        await service.prompt(thread.thread_id, "hello", request_id="req")
        await asyncio.gather(*service._producer_tasks)
        assert not _attempts(exporter)
        assert not service._workers
    finally:
        await _close(service)
