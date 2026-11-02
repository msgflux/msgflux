"""Thread-level model binding: defaults, overrides, conflicts and roundtrips."""

import sqlite3
from unittest.mock import Mock

import httpx2
import pytest

from msgflux.nn import Agent
from msgflux.runtime.service import AgentService, AgentSession, ServiceConflictError
from msgflux.runtime.service.records import AgentConfiguration, ServiceThread
from msgflux.runtime.service.store import SQLiteServiceStore

pytest.importorskip("litestar")

from msgflux.runtime.service.http.app import create_service_app
from msgflux.runtime.service.http.client import AgentServiceClient


def _overridable(model="provider/default"):
    return AgentConfiguration(
        harness="coding", model=model, supports_model_override=True
    )


def _agent():
    return Agent(name="m", model=Mock(model_type="chat_completion"))


def _service(configuration=None, *, factory=None):
    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "main",
        factory or (lambda _thread: AgentSession(_agent())),
        configuration=configuration,
    )
    return service


@pytest.mark.asyncio
async def test_open_thread_defaults_model_to_configuration_and_keeps_bindings():
    service = _service(_overridable())
    try:
        default = await service.open_thread("main", thread_id="default")
        pinned = await service.open_thread(
            "main", thread_id="pinned", model="openai/gpt-5"
        )
        assert default.model == "provider/default"
        assert pinned.model == "openai/gpt-5"
        # Reopening without a model is idempotent against the stored pin.
        assert (await service.open_thread("main", thread_id="default")) == default
        assert (await service.open_thread("main", thread_id="pinned")) == pinned
        assert service.store.thread("pinned").model == "openai/gpt-5"
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_reopen_with_different_model_conflicts_without_touching_store():
    service = _service(_overridable())
    try:
        thread = await service.open_thread("main", thread_id="reopen", model="openai/a")
        with pytest.raises(ServiceConflictError, match="model"):
            await service.open_thread("main", thread_id="reopen", model="openai/b")
        assert service.store.thread("reopen") == thread
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_custom_factory_without_support_rejects_override_and_stays_unbound():
    service = _service(AgentConfiguration(harness="coding", model="provider/fixed"))
    try:
        assert (await service.open_thread("main", thread_id="plain")).model is None
        with pytest.raises(ValueError, match="does not support model overrides"):
            await service.open_thread("main", thread_id="rejected", model="openai/x")
        with pytest.raises(KeyError):
            service.store.thread("rejected")
        with pytest.raises(ValueError, match="provider/model-id"):
            await service.open_thread("main", thread_id="invalid", model="openai")
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_factory_receives_pinned_model_binding():
    bindings = []
    service = _service(
        _overridable(),
        factory=lambda thread: bindings.append(thread) or AgentSession(_agent()),
    )
    thread = await service.open_thread("main", thread_id="factory", model="openai/pin")
    try:
        lease = await service.acquire_session(thread.thread_id)
        await lease.aclose()
        assert bindings == [thread]
    finally:
        await service.aclose()
        service.store.close()


def test_legacy_rows_without_model_migrate_to_null(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE service_threads (thread_id TEXT PRIMARY KEY, "
            "agent_id TEXT NOT NULL, cwd TEXT)"
        )
        connection.execute(
            "INSERT INTO service_threads VALUES ('legacy', 'main', NULL)"
        )
        connection.commit()
    finally:
        connection.close()

    store = SQLiteServiceStore(path)
    try:
        legacy = ServiceThread("legacy", "main", None, True, None)
        assert store.thread("legacy") == legacy
        pinned = ServiceThread("pinned", "main", None, True, "openai/x")
        assert store.bind_thread(pinned) == pinned
    finally:
        store.close()

    reopened = SQLiteServiceStore(path)
    try:
        assert reopened.thread("legacy").model is None
        assert reopened.thread("pinned").model == "openai/x"
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_http_open_thread_roundtrips_model_through_client_and_server():
    service = _service(_overridable())
    app = create_service_app(service, token="secret")
    transport = httpx2.ASGITransport(app=app)
    try:
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as http:
            client = AgentServiceClient(
                "http://testserver", token="secret", client=http
            )
            pinned = await client.open_thread(
                "main", thread_id="http", model="openai/http"
            )
            default = await client.open_thread("main", thread_id="http-default")
            assert pinned.model == "openai/http"
            assert default.model == "provider/default"
            assert (await client.open_thread("main", thread_id="http")) == pinned
        assert service.store.thread("http").model == "openai/http"
        assert service.store.thread("http-default").model == "provider/default"
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_restart_with_changed_default_preserves_previously_bound_model(tmp_path):
    path = tmp_path / "restart.sqlite"

    def configured(default):
        service = AgentService(store=SQLiteServiceStore(path))
        service.register(
            "main",
            lambda _: AgentSession(_agent()),
            configuration=_overridable(default),
        )
        return service

    first = configured("provider/original")
    try:
        await first.open_thread("main", thread_id="existing")
    finally:
        await first.aclose()
        first.store.close()
    restarted = configured("provider/new-default")
    try:
        existing = await restarted.open_thread("main", thread_id="existing")
        fresh = await restarted.open_thread("main", thread_id="new")
        assert existing.model == "provider/original"
        assert fresh.model == "provider/new-default"
    finally:
        await restarted.aclose()
        restarted.store.close()
