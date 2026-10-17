"""HTTP traces reuse host providers and do not retain frontend execution context."""

import asyncio

import httpx2
import pytest
from opentelemetry import trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http import ServiceHTTPConfig, create_service_app
from msgflux.runtime.service.http import telemetry


@pytest.fixture
def providers(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(telemetry.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(telemetry.metrics, "get_meter_provider", lambda: meter)
    yield exporter, provider, reader
    provider.shutdown()
    meter.shutdown()


def _app():
    service = AgentService(store=SQLiteServiceStore())
    app = create_service_app(
        service,
        token="private-token",
        instance_id="instance-test",
        http_config=ServiceHTTPConfig(("http://localhost:3000",)),
    )
    return service, app


@pytest.mark.asyncio
async def test_server_spans_cover_auth_errors_routes_and_incoming_trace(
    providers, monkeypatch
):
    exporter, _provider, reader = providers
    # Even when the host opts into header capture, authentication is redacted.
    monkeypatch.setenv("OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST", ".*")
    service, app = _app()
    traceparent = "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app), base_url="http://service"
        ) as client:
            unauthorized = await client.get(
                "/v1/threads/private-thread/snapshot",
                headers={
                    "traceparent": traceparent,
                    "authorization": "Bearer secret-wrong",
                    "cookie": "private-cookie",
                },
            )
            assert unauthorized.status_code == 401
            missing = await client.get(
                "/v1/threads/another-thread/snapshot",
                headers={"Authorization": "Bearer private-token"},
            )
            assert missing.status_code == 404
            unknown = await client.get(
                "/not-a-route/private-value",
                headers={"Authorization": "Bearer private-token"},
            )
            assert unknown.status_code == 404
            preflight = await client.options(
                "/v1/threads",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                },
            )
            assert preflight.status_code == 204
        spans = exporter.get_finished_spans()
        assert len(spans) == 4
        assert all(span.kind == SpanKind.SERVER for span in spans)
        first, second, third, fourth = spans
        assert first.context.trace_id == int("0123456789abcdef0123456789abcdef", 16)
        assert first.parent.span_id == int("0123456789abcdef", 16)
        assert first.name == second.name == "GET /v1/threads/{thread_id}/snapshot"
        assert first.attributes["msgflux.thread_id"] == "private-thread"
        assert first.attributes["msgflux.service.instance_id"] == "instance-test"
        assert first.attributes["http.status_code"] == 401
        assert second.attributes["http.status_code"] == 404
        assert third.name == "GET"
        assert fourth.name == "OPTIONS"
        for span in spans:
            assert "secret-wrong" not in str(span.attributes)
            assert "private-token" not in str(span.attributes)
            assert "private-cookie" not in str(span.attributes)
        data = reader.get_metrics_data()
        metrics = [
            m for r in data.resource_metrics for s in r.scope_metrics for m in s.metrics
        ]
        assert any("duration" in m.name for m in metrics)
        assert any("active_requests" in m.name for m in metrics)
        assert not trace.get_current_span().get_span_context().is_valid
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_internal_error_is_recorded_without_leaking_body(providers, monkeypatch):
    exporter, _provider, _reader = providers
    service, app = _app()

    def fail():
        raise RuntimeError("private exception detail")

    monkeypatch.setattr(service, "threads", fail)
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app), base_url="http://service"
        ) as client:
            response = await client.get(
                "/v1/threads", headers={"Authorization": "Bearer private-token"}
            )
        assert response.status_code == 500
        (span,) = exporter.get_finished_spans()
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["http.status_code"] == 500
        assert "private exception detail" not in str(span.attributes)
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_admission_links_to_independent_worker_trace(
    providers,
):
    from unittest.mock import AsyncMock
    from tests.test_agent_service import _agent, _response

    exporter, _provider, _reader = providers
    service, app = _app()
    agent = _agent()
    seen = []

    async def respond(**_kwargs):
        seen.append(trace.get_current_span().get_span_context())
        return _response("done")

    agent.generator.aforward = AsyncMock(side_effect=respond)
    service.register("main", lambda _thread: AgentSession(agent))
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app),
            base_url="http://service",
            headers={"Authorization": "Bearer private-token"},
        ) as client:
            thread = (
                await client.post("/v1/threads", json={"agent_id": "main"})
            ).json()
            response = await client.post(
                f"/v1/threads/{thread['thread_id']}/prompt",
                json={"prompt": "private prompt", "request_id": "req-one"},
            )
            receipt = response.json()
            await asyncio.wait_for(service.wait(thread["thread_id"], "req-one"), 3)
        prompt_span = next(
            s for s in exporter.get_finished_spans() if s.name.endswith("/prompt")
        )
        for field in ("thread_id", "request_id", "run_id"):
            assert prompt_span.attributes[f"msgflux.{field}"] == receipt[field]
        assert "private prompt" not in str(prompt_span.attributes)
        assert seen and all(
            context.trace_id != prompt_span.context.trace_id for context in seen
        )
        attempt = next(
            s
            for s in exporter.get_finished_spans()
            if s.name == "msgflux.service.execute"
        )
        assert attempt.parent is None
        assert attempt.links[0].context == prompt_span.context
        assert all(context.trace_id == attempt.context.trace_id for context in seen)
        assert not trace.get_current_span().get_span_context().is_valid
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_stream_has_one_server_span_and_disconnect_ends_it(providers):
    from tests.test_agent_service import _agent

    exporter, provider, _reader = providers
    service, app = _app()
    service.register("main", lambda _thread: AgentSession(_agent()))
    thread = await service.open_thread("main")
    snapshot = asyncio.Event()
    disconnect = asyncio.Event()

    async def receive():
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            snapshot.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": f"/v1/threads/{thread.thread_id}/watch",
        "query_string": b"",
        "root_path": "",
        "server": ("service", 80),
        "client": ("test", 1234),
        "headers": [(b"authorization", b"Bearer private-token")],
    }
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(snapshot.wait(), 3)
        assert not exporter.get_finished_spans()
        disconnect.set()
        await asyncio.wait_for(task, 3)
        (span,) = exporter.get_finished_spans()
        assert span.name == "GET /v1/threads/{thread_id}/watch"
        assert span.kind == SpanKind.SERVER
        assert not trace.get_current_span().get_span_context().is_valid
        # Closing this app/service must not shut down the host's shared provider.
        await service.aclose()
        with provider.get_tracer("test").start_as_current_span("still-usable"):
            pass
        assert exporter.get_finished_spans()[-1].name == "still-usable"
    finally:
        disconnect.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await service.aclose()
        service.store.close()
