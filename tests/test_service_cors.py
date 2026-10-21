"""Browser access uses native CORS without bypassing bearer authentication."""

import asyncio

import httpx2
import msgspec
import pytest

from msgflux.runtime.service import AgentService, SQLiteServiceStore
from msgflux.runtime.service.http import ServiceHTTPConfig, create_service_app
from msgflux.runtime.service.local.manifest import load_service_manifest

ORIGIN = "http://localhost:3000"


@pytest.mark.asyncio
async def test_preflight_is_unauthenticated_but_api_still_requires_bearer():
    service = AgentService(store=SQLiteServiceStore())
    app = create_service_app(
        service, token="secret", http_config=ServiceHTTPConfig((ORIGIN,))
    )
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app), base_url="http://service"
        ) as client:
            preflight = await client.options(
                "/v1/threads",
                headers={
                    "Origin": ORIGIN,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "authorization, content-type, traceparent",
                },
            )
            assert preflight.status_code == 204
            assert preflight.headers["access-control-allow-origin"] == ORIGIN
            assert "authorization" in preflight.headers["access-control-allow-headers"]
            assert "access-control-allow-credentials" not in preflight.headers
            denied = await client.get("/v1/health", headers={"Origin": ORIGIN})
            assert denied.status_code == 401
            assert denied.headers["access-control-allow-origin"] == ORIGIN
            assert denied.headers["www-authenticate"] == "Bearer"
            assert "Origin" in denied.headers["vary"]
            assert "WWW-Authenticate" in denied.headers["access-control-expose-headers"]
            allowed = await client.get(
                "/v1/health",
                headers={"Origin": ORIGIN, "Authorization": "Bearer secret"},
            )
            assert allowed.status_code == 200
            assert allowed.headers["access-control-allow-origin"] == ORIGIN
            assert not service.agents()
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.parametrize(
    "origin,method,headers",
    [
        ("https://untrusted.example", "POST", "authorization"),
        ("http://localhost:3000.evil.example", "POST", "authorization"),
        (ORIGIN, "DELETE", "authorization"),
        (ORIGIN, "POST", "x-unexpected"),
    ],
)
@pytest.mark.asyncio
async def test_preflight_rejects_disallowed_origin_method_or_headers(
    origin, method, headers
):
    service = AgentService(store=SQLiteServiceStore())
    app = create_service_app(
        service, token="secret", http_config=ServiceHTTPConfig((ORIGIN,))
    )
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app), base_url="http://service"
        ) as client:
            response = await client.options(
                "/v1/threads",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": method,
                    "Access-Control-Request-Headers": headers,
                },
            )
            assert response.status_code == 400
            if origin != ORIGIN:
                assert "access-control-allow-origin" not in response.headers
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.asyncio
async def test_origin_denial_and_nonbrowser_calls_do_not_change_authentication(enabled):
    service = AgentService(store=SQLiteServiceStore())
    app = create_service_app(
        service,
        token="secret",
        http_config=ServiceHTTPConfig((ORIGIN,) if enabled else ()),
    )
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app), base_url="http://service"
        ) as client:
            for headers in ({}, {"Origin": "https://untrusted.example"}):
                response = await client.get(
                    "/v1/health", headers={**headers, "Authorization": "Bearer secret"}
                )
                assert response.status_code == 200
                assert "access-control-allow-origin" not in response.headers
            if not enabled:
                response = await client.options(
                    "/v1/health",
                    headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET"},
                )
                assert response.status_code == 401
                assert "access-control-allow-origin" not in response.headers
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_cors_is_present_on_domain_validation_and_internal_errors(monkeypatch):
    service = AgentService(store=SQLiteServiceStore())
    app = create_service_app(
        service, token="secret", http_config=ServiceHTTPConfig((ORIGIN,))
    )
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app),
            base_url="http://service",
            headers={"Origin": ORIGIN, "Authorization": "Bearer secret"},
        ) as client:
            missing = await client.get("/v1/threads/missing/snapshot")
            assert missing.status_code == 404
            invalid = await client.post("/v1/threads", json={"unexpected": True})
            assert invalid.status_code == 422

            def fail():
                raise RuntimeError("private failure")

            monkeypatch.setattr(service, "threads", fail)
            failed = await client.get("/v1/threads")
            assert failed.status_code == 500
            assert "private failure" not in failed.text
            for response in (missing, invalid, failed):
                assert response.headers["access-control-allow-origin"] == ORIGIN
                assert "Origin" in response.headers["vary"]
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_sse_headers_are_sent_without_waiting_for_stream_completion():
    from msgflux.runtime.service import AgentSession
    from tests.test_agent_service import _agent

    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _thread: AgentSession(_agent()))
    thread = await service.open_thread("main")
    app = create_service_app(
        service, token="secret", http_config=ServiceHTTPConfig((ORIGIN,))
    )
    starts = []
    snapshot = asyncio.Event()
    receive_wait = asyncio.Event()

    async def receive():
        await receive_wait.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            starts.append(message)
        elif message["type"] == "http.response.body" and message.get("body"):
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
        "headers": [(b"origin", ORIGIN.encode()), (b"authorization", b"Bearer secret")],
    }
    task = asyncio.create_task(app(scope, receive, send))
    try:
        await asyncio.wait_for(snapshot.wait(), 2)
        assert not task.done()
        headers = dict(starts[0]["headers"])
        assert starts[0]["status"] == 200
        assert headers[b"access-control-allow-origin"] == ORIGIN.encode()
        assert headers[b"content-type"].startswith(b"text/event-stream")
    finally:
        receive_wait.set()
        await asyncio.wait_for(task, 2)
        await service.aclose()
        service.store.close()


@pytest.mark.parametrize(
    "origin",
    [
        "null",
        "file:///tmp",
        "https://example.com/",
        "https://user:pass@example.com",
        "https://example.com?x=y",
        "https://example.com#fragment",
        "https://*.example.com",
        "http://localhost:0",
        "http://localhost:99999",
    ],
)
def test_config_rejects_nonorigin_values(origin):
    with pytest.raises(ValueError, match="CORS origin"):
        ServiceHTTPConfig((origin,))


def test_manifest_http_policy_is_typed_and_unknown_options_are_rejected(tmp_path):
    config = tmp_path / "config.toml"
    prefix = (
        '[defaults]\nagent = "custom"\n[agents.custom]\nfactory = "package:factory"\n'
    )
    config.write_text(
        prefix + '[http]\ncors_allow_origins = ["http://localhost:3000"]\n'
    )
    assert load_service_manifest(config).http == ServiceHTTPConfig((ORIGIN,))
    config.write_text(prefix + "[http]\nallow_anyone = true\n")
    with pytest.raises(ValueError, match="unknown field"):
        load_service_manifest(config)
    assert msgspec.json.decode(
        msgspec.json.encode(ServiceHTTPConfig((ORIGIN,))), type=ServiceHTTPConfig
    ).cors_allow_origins == (ORIGIN,)


def _session_factory(_thread):
    from msgflux.runtime.service import AgentSession
    from tests.test_agent_service import _agent

    return AgentSession(_agent())


@pytest.mark.asyncio
async def test_manifest_policy_reaches_real_tcp_runner(tmp_path):
    import os
    from pathlib import Path

    from msgflux.runtime.service.local.runner import serve_local_service
    from tests.test_local_service_runner import _wait_for_record

    if os.name != "posix":
        pytest.skip("Local runner requires POSIX lifetime locks")
    config = tmp_path / "config.toml"
    config.write_text(
        '[defaults]\nagent = "custom"\n'
        '[agents.custom]\nfactory = "tests.test_service_cors:_session_factory"\n'
        '[http]\ncors_allow_origins = ["http://localhost:3000"]\n'
    )
    runtime = tmp_path / "runtime"
    runner = asyncio.create_task(
        serve_local_service(config=config, runtime_dir=runtime, cwd=Path.cwd())
    )
    try:
        record = await asyncio.wait_for(_wait_for_record(runtime), 5)
        async with httpx2.AsyncClient(base_url=record.url) as client:
            response = await client.options(
                "/v1/threads",
                headers={
                    "Origin": ORIGIN,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "authorization",
                },
            )
            assert response.status_code == 204
            assert response.headers["access-control-allow-origin"] == ORIGIN
            assert (await client.get("/v1/health")).status_code == 401
    finally:
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)


@pytest.mark.parametrize("origin", [ORIGIN, "https://app.example.com", "null"])
@pytest.mark.asyncio
async def test_wildcard_allows_origins_without_bypassing_bearer(origin):
    service = AgentService(store=SQLiteServiceStore())
    app = create_service_app(
        service, token="secret", http_config=ServiceHTTPConfig(("*",))
    )
    try:
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app), base_url="http://service"
        ) as client:
            preflight = await client.options(
                "/v1/threads",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "authorization, content-type",
                },
            )
            denied = await client.get("/v1/health", headers={"Origin": origin})
            allowed = await client.get(
                "/v1/health",
                headers={"Origin": origin, "Authorization": "Bearer secret"},
            )
            assert preflight.status_code == 204
            assert denied.status_code == 401
            assert allowed.status_code == 200
            for response in (preflight, denied, allowed):
                assert response.headers["access-control-allow-origin"] == "*"
                assert "access-control-allow-credentials" not in response.headers
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.parametrize("origins", [("*", ORIGIN), (ORIGIN, "*"), ("*", "*")])
def test_wildcard_must_be_the_only_origin(origins):
    with pytest.raises(ValueError, match="wildcard must be the only"):
        ServiceHTTPConfig(origins)


def test_manifest_accepts_wildcard_and_rejects_mixed_origins(tmp_path):
    config = tmp_path / "config.toml"
    prefix = (
        '[defaults]\nagent = "custom"\n[agents.custom]\nfactory = "package:factory"\n'
    )
    config.write_text(prefix + '[http]\ncors_allow_origins = ["*"]\n')
    assert load_service_manifest(config).http == ServiceHTTPConfig(("*",))
    config.write_text(
        prefix + '[http]\ncors_allow_origins = ["*", "http://localhost:3000"]\n'
    )
    with pytest.raises(ValueError, match="wildcard must be the only"):
        load_service_manifest(config)
