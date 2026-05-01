import asyncio
from types import SimpleNamespace

import pytest

from msgflux.channels import ChannelRegistry, RateLimitDecision
from msgflux.channels.http import app as app_module
from msgflux.channels.http.app import create_app


class FakeAgent:
    name = "support"

    async def acall(self, **kwargs):
        assert kwargs["stream"] is False
        assert kwargs["vars"] == {"tenant": "acme"}
        return "msgflux server ok"


class BillingAgent(FakeAgent):
    name = "billing"


def test_chat_completions_route_with_msgspec_response():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ChannelRegistry()
    registry.agent(FakeAgent())
    client = TestClient(create_app(registry))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "support",
            "messages": [{"role": "user", "content": "hello"}],
            "run_config": {"vars": {"tenant": "acme"}},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["choices"][0]["message"]["content"] == "msgflux server ok"


def test_health_and_agents_routes():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ChannelRegistry()
    registry.agent(FakeAgent())
    client = TestClient(create_app(registry))

    home_response = client.get("/")
    assert home_response.status_code == 200
    assert home_response.json() == {
        "status": "ok",
        "agents": "/agents",
        "health": "/health",
        "ready": "/ready",
        "chat_completions": "/v1/chat/completions",
    }

    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}

    agents_response = client.get("/agents")
    assert agents_response.status_code == 200
    assert agents_response.json() == {
        "agents": [
            {
                "name": "support",
                "description": None,
                "tags": [],
                "capabilities": {},
            }
        ]
    }


def test_agents_route_returns_metadata_details():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class MetadataAgent(FakeAgent):
        """Support questions about orders."""

        name = "support"

    registry = ChannelRegistry()
    registry.agent(
        MetadataAgent(),
        tags=["support", "orders"],
        capabilities={"streaming": True, "tools": True},
    )
    client = TestClient(create_app(registry))

    response = client.get("/agents")

    assert response.status_code == 200
    assert response.json() == {
        "agents": [
            {
                "name": "support",
                "description": "Support questions about orders.",
                "tags": ["support", "orders"],
                "capabilities": {"streaming": True, "tools": True},
            }
        ]
    }


def test_ready_route_tracks_lifespan_state():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ChannelRegistry()
    app = create_app(registry)

    assert registry.readiness().status == "starting"
    with TestClient(app) as client:
        response = client.get("/ready")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ready",
            "ready": True,
            "error": None,
        }

    assert registry.readiness().status == "stopped"


def test_registry_settings_disable_docs_and_enable_cors():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ChannelRegistry()
    registry.agent(FakeAgent())
    registry.settings(
        enable_docs=False,
        cors=True,
        allowed_origins=["https://app.example.com"],
    )
    client = TestClient(create_app(registry))

    docs_response = client.get("/docs")
    assert docs_response.status_code == 404

    cors_response = client.options(
        "/v1/chat/completions",
        headers={
            "Origin": "https://app.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert cors_response.headers["access-control-allow-origin"] == (
        "https://app.example.com"
    )


def test_max_request_bytes_returns_413():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ChannelRegistry()
    registry.agent(FakeAgent())
    registry.settings(max_request_bytes=8)
    client = TestClient(create_app(registry))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "support", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_auth_and_authorize_are_enforced():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ChannelRegistry()
    registry.agent(FakeAgent())

    @registry.auth
    def auth(http_request):
        if http_request.headers.get("authorization") != "Bearer ok":
            return False
        return {"tenant": "acme"}

    @registry.authorize(agent="support")
    def authorize(request, context):
        principal = context.state["principal"]
        return principal["tenant"] == request.run_config.get("vars", {}).get("tenant")

    client = TestClient(create_app(registry))

    unauthorized = client.post(
        "/v1/chat/completions",
        json={"model": "support", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert unauthorized.status_code == 401
    assert unauthorized.json()["error"]["code"] == "unauthorized"

    unauthorized_stream = client.post(
        "/v1/chat/completions",
        json={
            "model": "support",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert unauthorized_stream.status_code == 401
    assert unauthorized_stream.json()["error"]["code"] == "unauthorized"

    forbidden = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer ok"},
        json={
            "model": "support",
            "messages": [{"role": "user", "content": "hi"}],
            "run_config": {"vars": {"tenant": "other"}},
        },
    )
    assert forbidden.status_code == 403
    assert forbidden.json()["error"]["code"] == "forbidden"

    allowed = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer ok"},
        json={
            "model": "support",
            "messages": [{"role": "user", "content": "hi"}],
            "run_config": {"vars": {"tenant": "acme"}},
        },
    )
    assert allowed.status_code == 200


def test_rate_limit_by_api_key_returns_429():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ChannelRegistry()
    registry.agent(FakeAgent())
    registry.rate_limit(requests=1, window_s=60, by="api_key")
    client = TestClient(create_app(registry))
    payload = {
        "model": "support",
        "messages": [{"role": "user", "content": "hi"}],
        "run_config": {"vars": {"tenant": "acme"}},
    }

    first = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer key-1"},
        json=payload,
    )
    second = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer key-1"},
        json=payload,
    )
    other_key = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer key-2"},
        json=payload,
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(second.headers["retry-after"]) >= 1
    assert other_key.status_code == 200


def test_pluggable_rate_limit_store_controls_http_429():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class DenyStore:
        async def hit(self, bucket):
            assert bucket.name == "http-api-key"
            return RateLimitDecision(allowed=False, retry_after_s=12)

    registry = ChannelRegistry()
    registry.agent(FakeAgent())
    registry.rate_limit_store(DenyStore())
    registry.rate_limit(name="http-api-key", requests=100, window_s=60, by="api_key")
    client = TestClient(create_app(registry))

    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer key-1"},
        json={
            "model": "support",
            "messages": [{"role": "user", "content": "hi"}],
            "run_config": {"vars": {"tenant": "acme"}},
        },
    )

    assert response.status_code == 429
    assert response.headers["retry-after"] == "12"
    assert response.json()["error"]["code"] == "rate_limit_exceeded"


def test_response_headers_include_request_and_correlation_ids():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ChannelRegistry()
    registry.agent(FakeAgent())
    client = TestClient(create_app(registry))

    response = client.post(
        "/v1/chat/completions",
        headers={
            "X-Request-ID": "req-test",
            "X-Correlation-ID": "corr-test",
            "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        },
        json={
            "model": "support",
            "messages": [{"role": "user", "content": "hi"}],
            "run_config": {"vars": {"tenant": "acme"}},
        },
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-test"
    assert response.headers["x-correlation-id"] == "corr-test"
    assert response.headers["traceparent"] == (
        "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"
    )


def test_rate_limit_can_target_agent():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ChannelRegistry()
    registry.agent(FakeAgent())
    registry.agent(BillingAgent())
    registry.rate_limit(agent="support", requests=1, window_s=60, by="api_key")
    client = TestClient(create_app(registry))
    support_payload = {
        "model": "support",
        "messages": [{"role": "user", "content": "hi"}],
        "run_config": {"vars": {"tenant": "acme"}},
    }
    billing_payload = {
        "model": "billing",
        "messages": [{"role": "user", "content": "hi"}],
        "run_config": {"vars": {"tenant": "acme"}},
    }

    first_support = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer key-1"},
        json=support_payload,
    )
    second_support = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer key-1"},
        json=support_payload,
    )
    billing = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer key-1"},
        json=billing_payload,
    )

    assert first_support.status_code == 200
    assert second_support.status_code == 429
    assert billing.status_code == 200


def test_error_handler_maps_exception_to_http_payload():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class BrokenAgent:
        name = "support"

        async def acall(self, **kwargs):
            raise ValueError("provider down")

    registry = ChannelRegistry()
    registry.agent(BrokenAgent())

    @registry.error_handler(ValueError)
    def map_value_error(exc):
        return {
            "message": str(exc),
            "code": "provider_error",
            "type": "agent_error",
            "status_code": 502,
        }

    client = TestClient(create_app(registry))

    response = client.post(
        "/v1/chat/completions",
        headers={"X-Request-ID": "req-error", "X-Correlation-ID": "corr-error"},
        json={"model": "support", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "message": "provider down",
            "type": "agent_error",
            "code": "provider_error",
            "request_id": "req-error",
            "correlation_id": "corr-error",
        }
    }


def test_request_timeout_returns_504():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class SlowAgent:
        name = "support"

        async def acall(self, **kwargs):
            await asyncio.sleep(0.05)
            return "late"

    registry = ChannelRegistry()
    registry.agent(SlowAgent())
    registry.settings(request_timeout_s=0.001)
    client = TestClient(create_app(registry))

    response = client.post(
        "/v1/chat/completions",
        json={"model": "support", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "request_timeout"


def test_lifespan_hooks_run_with_fastapi_lifespan():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ChannelRegistry()
    events = []

    @registry.startup
    async def startup(app):
        events.append(("startup", bool(app)))

    @registry.shutdown
    async def shutdown(app):
        events.append(("shutdown", bool(app)))

    with TestClient(create_app(registry)) as client:
        assert client.get("/health").status_code == 200
        assert events == [("startup", True)]

    assert events == [("startup", True), ("shutdown", True)]


def test_enable_otel_uses_fastapi_instrumentor(monkeypatch):
    pytest.importorskip("fastapi")

    calls = []
    real_import_module = app_module.import_module

    class FakeInstrumentor:
        @staticmethod
        def instrument_app(app, **kwargs):
            calls.append((app, kwargs))

    def fake_import_module(name):
        if name == "opentelemetry.instrumentation.fastapi":
            return SimpleNamespace(FastAPIInstrumentor=FakeInstrumentor)
        return real_import_module(name)

    monkeypatch.setattr(app_module, "import_module", fake_import_module)
    registry = ChannelRegistry()
    registry.settings(enable_otel=True, otel_kwargs={"excluded_urls": "health"})

    app = create_app(registry)

    assert calls == [(app, {"excluded_urls": "health"})]
