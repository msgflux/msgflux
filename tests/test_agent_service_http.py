"""Focused contracts for the optional AgentService HTTP adapter."""

import pytest
import httpx2

pytest.importorskip("litestar")
from litestar.testing import AsyncTestClient
from unittest.mock import AsyncMock, Mock

from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http.app import create_service_app


def _service(factory=None):
    service = AgentService(store=SQLiteServiceStore())
    calls = []

    def make_session(thread_id):
        calls.append(thread_id)
        return factory(thread_id)

    service.register("agent", make_session if factory is not None else lambda _: None)
    return service, calls


@pytest.mark.asyncio
async def test_auth_runs_before_body_validation_or_agent_factory():
    service, calls = _service()
    app = create_service_app(service, token="secret")

    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/v1/threads",
            content=b"{ malformed",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 401
    assert response.json() == {
        "code": "unauthorized",
        "message": "Bearer token required",
    }
    assert calls == []
    await service.aclose()


@pytest.mark.asyncio
async def test_routes_use_strict_json_and_service_scope_error_mapping():
    service, _ = _service()
    app = create_service_app(service, token="secret")
    headers = {"Authorization": "Bearer secret"}

    async with AsyncTestClient(app=app) as client:
        assert (await client.get("/v1/agents", headers=headers)).json() == {
            "agents": ["agent"]
        }

        malformed = await client.post(
            "/v1/threads",
            headers=headers,
            json={"agent_id": "agent", "extra": True},
        )
        assert malformed.status_code == 422
        assert malformed.json()["code"] == "invalid_request"

        opened = await client.post(
            "/v1/threads", headers=headers, json={"agent_id": "agent"}
        )
        assert opened.status_code == 201
        thread_id = opened.json()["thread_id"]

        missing = await client.get(
            f"/v1/threads/{thread_id}/requests/unknown", headers=headers
        )
        assert missing.status_code == 404
        assert missing.json()["code"] == "not_found"

        replay = await client.get(
            f"/v1/threads/{thread_id}/watch",
            headers={**headers, "Last-Event-ID": "3"},
        )
        assert replay.status_code == 422
        assert replay.json()["code"] == "invalid_request"

    # The app borrows its service unless ownership is explicitly requested.
    assert service.agents() == ("agent",)
    await service.aclose()


@pytest.mark.asyncio
async def test_thread_workspace_binding_is_exposed_and_immutable(tmp_path):
    service, calls = _service()
    app = create_service_app(service, token="secret")
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    headers = {"Authorization": "Bearer secret"}
    transport = httpx2.ASGITransport(app=app)
    try:
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://testserver", headers=headers
        ) as client:
            body = {"agent_id": "agent", "thread_id": "project", "cwd": str(first)}
            opened = await client.post("/v1/threads", json=body)
            assert opened.status_code == 201
            assert opened.json() == body
            reopened = await client.post(
                "/v1/threads", json={"agent_id": "agent", "thread_id": "project"}
            )
            assert reopened.json() == body
            listed = await client.get("/v1/threads")
            assert listed.json() == {"threads": [body]}
            conflict = await client.post(
                "/v1/threads", json={**body, "cwd": str(second)}
            )
            assert conflict.status_code == 409
            assert conflict.json()["code"] == "service_conflict"
            for path in ("relative", str(tmp_path / "missing")):
                invalid = await client.post(
                    "/v1/threads",
                    json={"agent_id": "agent", "thread_id": "invalid", "cwd": path},
                )
                assert invalid.status_code == 422
            assert calls == []
            assert len(service.threads()) == 1
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_runs_endpoint_only_returns_checkpoint_summaries():
    checkpoints = InMemoryCheckpointStore()
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="http-runs", model=model)
    agent.checkpoint_store = checkpoints
    service, calls = _service(lambda _thread: AgentSession(agent))
    thread = await service.open_thread("agent", thread_id="runs-thread")
    checkpoints.save_state(
        agent.get_module_name(),
        thread.thread_id,
        "legacy-run",
        {"status": "completed", "config": {"secret": "host-only"}, "state": "private"},
    )
    app = create_service_app(service, token="secret")
    headers = {"Authorization": "Bearer secret"}
    try:
        async with AsyncTestClient(app=app) as client:
            denied = await client.get(f"/v1/threads/{thread.thread_id}/runs")
            assert denied.status_code == 401
            assert calls == []
            response = await client.get(
                f"/v1/threads/{thread.thread_id}/runs", headers=headers
            )
            assert response.status_code == 200
            assert set(response.json()) == {"runs"}
            assert set(response.json()["runs"][0]) == {"run_id", "status", "updated_at"}
            assert response.json()["runs"][0]["run_id"] == "legacy-run"
            assert calls == [thread]
            assert "host-only" not in response.text and "private" not in response.text
            missing = await client.get("/v1/threads/missing/runs", headers=headers)
            assert missing.status_code == 404
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_resume_body_rejects_network_quiescence_assertion():
    service, _ = _service()
    thread = await service.open_thread("agent", thread_id="resume-thread")
    app = create_service_app(service, token="secret")

    async with AsyncTestClient(app=app) as client:
        response = await client.post(
            f"/v1/threads/{thread.thread_id}/runs/run-1/resume",
            headers={"Authorization": "Bearer secret"},
            json={"worker_stopped": True},
        )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    await service.aclose()


@pytest.mark.asyncio
async def test_approval_routes_auth_validate_payload_and_forbid_unconfigured_reviewer():
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="http-approval-default", model=model)
    service, calls = _service(lambda _thread: AgentSession(agent))
    thread = await service.open_thread("agent", thread_id="approval-thread")
    app = create_service_app(service, token="secret")
    root = f"/v1/threads/{thread.thread_id}/runs/run-1/approvals"
    try:
        async with AsyncTestClient(app=app) as client:
            denied = await client.get(root)
            assert denied.status_code == 401
            assert calls == []
            malformed = await client.post(
                root + "/request/decision",
                headers={"Authorization": "Bearer secret"},
                json={"approved": 1, "expected_revision": 0},
            )
            assert malformed.status_code == 422
            assert malformed.json()["code"] == "invalid_request"
            forbidden = await client.get(
                root, headers={"Authorization": "Bearer secret"}
            )
            assert forbidden.status_code == 403
            assert forbidden.json()["code"] == "forbidden"
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_prompt_deduplicates_request_and_receipt_tracks_settled_run():
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="http-test-agent", model=model)
    answer = ModelResponse()
    answer.set_response_type("text_generation")
    answer.add("ready")
    answer.reasoning = None
    agent.generator.aforward = AsyncMock(return_value=answer)

    service = AgentService(store=SQLiteServiceStore())
    service.register("agent", lambda _thread_id: AgentSession(agent))
    thread = await service.open_thread("agent", thread_id="prompt-thread")
    app = create_service_app(service, token="secret")

    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        headers = {"Authorization": "Bearer secret"}
        body = {"prompt": "hello", "request_id": "stable-id"}
        first = await client.post(
            f"/v1/threads/{thread.thread_id}/prompt", headers=headers, json=body
        )
        duplicate = await client.post(
            f"/v1/threads/{thread.thread_id}/prompt", headers=headers, json=body
        )
        assert first.status_code == duplicate.status_code == 202
        assert first.json()["run_id"] == duplicate.json()["run_id"]

        settled = await service.wait(thread.thread_id, "stable-id")
        receipt = await client.get(
            f"/v1/threads/{thread.thread_id}/requests/stable-id", headers=headers
        )

    assert settled.status == receipt.json()["status"] == "completed"
    assert agent.generator.aforward.await_count == 1
    await service.aclose()


@pytest.mark.asyncio
async def test_invalid_prompt_payloads_do_not_instantiate_agent():
    service, calls = _service()
    thread = await service.open_thread("agent")
    app = create_service_app(service, token="secret")
    headers = {"Authorization": "Bearer secret"}
    try:
        async with AsyncTestClient(app=app) as client:
            for payload in (
                {"prompt": "hello", "request_id": 123},
                {"prompt": ["hello"], "request_id": "one"},
                {
                    "prompt": "hello",
                    "request_id": "one",
                    "workspace": "other-workspace",
                },
                {"prompt": "hello", "request_id": ""},
            ):
                response = await client.post(
                    f"/v1/threads/{thread.thread_id}/prompt",
                    headers=headers,
                    json=payload,
                )
                assert response.status_code == 422, response.text
            malformed = await client.post(
                f"/v1/threads/{thread.thread_id}/prompt",
                headers={**headers, "Content-Type": "application/json"},
                content=b"{ broken",
            )
            assert malformed.status_code == 422
        assert calls == []
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_owned_app_shutdown_closes_session_once_and_borrows_journal():
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="owned-app", model=model)
    close = AsyncMock()
    journal = SQLiteServiceStore()
    service = AgentService(store=journal)
    service.register("agent", lambda _thread: AgentSession(agent, on_close=close))
    thread = await service.open_thread("agent")
    await service.session(thread.thread_id)
    app = create_service_app(service, token="secret", close_service=True)
    try:
        async with AsyncTestClient(app=app):
            pass
        close.assert_awaited_once()
        with pytest.raises(RuntimeError, match="closed"):
            service.threads()
        assert journal.thread(thread.thread_id) == thread
        await service.aclose()
        close.assert_awaited_once()
    finally:
        journal.close()


@pytest.mark.asyncio
async def test_authenticated_health_identifies_instance_without_creating_agent():
    service, calls = _service()
    app = create_service_app(service, token="secret", instance_id="health-instance")
    try:
        async with AsyncTestClient(app=app) as client:
            unauthorized = await client.get("/v1/health")
            assert unauthorized.status_code == 401
            response = await client.get(
                "/v1/health", headers={"Authorization": "Bearer secret"}
            )
            assert response.json() == {"instance_id": "health-instance", "version": 1}
        assert calls == []
    finally:
        await service.aclose()
        service.store.close()
