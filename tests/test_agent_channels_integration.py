"""Offline integration coverage for AgentChannel over AgentService."""

import asyncio
import hashlib
from unittest.mock import AsyncMock, Mock

import httpx2
import msgspec
import pytest

from msgflux.channels import (
    AgentChannel,
    ChannelPermissionError,
    ChannelReply,
    ChannelRequest,
)

from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    ServiceConflictError,
    SQLiteServiceStore,
)
from msgflux.runtime.service.http import AgentServiceClient, create_service_app


def _response(content="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    response.reasoning = None
    return response


def _agent(output="answer"):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="channel-test", model=model)
    agent.generator.aforward = AsyncMock(return_value=_response(output))
    return agent


def _request(prompt="hello"):
    return ChannelRequest(agent_id="main", thread_id="channel-thread", prompt=prompt)


def _request_id(name, principal, external_id):
    digest = hashlib.sha256(
        msgspec.json.encode((name, principal, external_id))
    ).hexdigest()
    return f"channel:{digest}"


def _channel(service, name="web", authorize=None):
    return AgentChannel(
        service, name=name, authorize=authorize or (lambda _ctx, _req: True)
    )


@pytest.mark.asyncio
async def test_duplicate_channel_delivery_is_idempotent_and_conflicts_on_changed_prompt():
    agent = _agent()
    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _thread: AgentSession(agent))
    channel = _channel(service)
    try:
        first = await channel.prompt(_request(), principal="alice", request_id="evt")
        duplicate = await channel.prompt(
            _request(), principal="alice", request_id="evt"
        )
        receipt = await service.wait(
            "channel-thread", _request_id("web", "alice", "evt")
        )
        assert first.receipt.run_id == duplicate.receipt.run_id == receipt.run_id
        assert agent.generator.aforward.await_count == 1
        with pytest.raises(ServiceConflictError, match="another input"):
            await channel.prompt(
                _request("changed"), principal="alice", request_id="evt"
            )
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_channels_share_explicit_thread_history_but_namespace_request_ids():
    agent = _agent()
    checkpoints = InMemoryCheckpointStore()
    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "main", lambda _thread: AgentSession(agent, checkpoint_store=checkpoints)
    )
    try:
        a = await _channel(service, "web").prompt(
            _request("first"), principal="alice", request_id="evt"
        )
        await service.wait("channel-thread", a.receipt.request_id)
        b = await _channel(service, "mobile").prompt(
            _request("second"), principal="alice", request_id="evt"
        )
        await service.wait("channel-thread", b.receipt.request_id)
        assert a.receipt.request_id == _request_id("web", "alice", "evt")
        assert b.receipt.request_id == _request_id("mobile", "alice", "evt")
        assert a.receipt.request_id != b.receipt.request_id
        assert a.receipt.thread_id == b.receipt.thread_id == "channel-thread"
        assert agent.generator.aforward.await_count == 2
        state = checkpoints.load_state(
            "channel-test", "channel-thread", b.receipt.run_id
        )
        history = state["messages"]["items"]
        assert any("first" in str(item) for item in history)
        assert any("second" in str(item) for item in history)
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_different_principals_and_source_event_ids_create_independent_admissions():
    agent = _agent()
    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _thread: AgentSession(agent))
    channel = _channel(service)
    try:
        results = []
        for principal, event in (("alice", "evt"), ("bob", "evt"), ("alice", "other")):
            result = await channel.prompt(
                _request(), principal=principal, request_id=event
            )
            results.append(result.receipt)
            await service.wait("channel-thread", result.receipt.request_id)
        assert len({r.request_id for r in results}) == 3
        assert len({r.run_id for r in results}) == 3
        assert agent.generator.aforward.await_count == 3
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_channel_return_leaves_admitted_worker_running():
    entered, release = asyncio.Event(), asyncio.Event()
    agent = _agent()

    async def delayed(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return _response("survived")

    agent.generator.aforward = AsyncMock(side_effect=delayed)
    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _thread: AgentSession(agent))
    channel = _channel(service)
    try:
        await service.open_thread("main", thread_id="channel-thread")
        async with service.watch("channel-thread"):
            admission = await asyncio.wait_for(
                channel.prompt(_request(), principal="alice", request_id="evt"), 2
            )
            await asyncio.wait_for(entered.wait(), 2)
            rid = _request_id("web", "alice", "evt")
            assert admission.receipt.request_id == rid
            assert service.receipt("channel-thread", rid).status == "running"
        # Closing the only watcher leaves the service-owned worker running.
        assert service.receipt("channel-thread", rid).status == "running"
        release.set()
        settled = await asyncio.wait_for(service.wait("channel-thread", rid), 2)
        assert settled.status == "completed"
        assert agent.generator.aforward.await_count == 1
    finally:
        release.set()
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_sqlite_restart_reuses_admission_and_checkpoint_without_second_model_call(
    tmp_path,
):
    journal_path, checkpoint_path = (
        str(tmp_path / "admissions.sqlite"),
        str(tmp_path / "checkpoints.sqlite"),
    )
    journal, checkpoints = (
        SQLiteServiceStore(journal_path),
        SQLiteCheckpointStore(checkpoint_path),
    )
    agent = _agent()
    first = AgentService(store=journal)
    first.register(
        "main", lambda _thread: AgentSession(agent, checkpoint_store=checkpoints)
    )
    request = _request()
    try:
        accepted = await _channel(first).prompt(
            request, principal="alice", request_id="evt"
        )
        settled = await first.wait(request.thread_id, accepted.receipt.request_id)
        assert settled.status == "completed"
    finally:
        await first.aclose()
    restarted_agent = _agent()
    restarted = AgentService(store=SQLiteServiceStore(journal_path))
    restarted.register(
        "main",
        lambda _thread: AgentSession(restarted_agent, checkpoint_store=checkpoints),
    )
    try:
        duplicate = await _channel(restarted).prompt(
            request, principal="alice", request_id="evt"
        )
        recovered = await restarted.wait(
            request.thread_id, duplicate.receipt.request_id
        )
        assert recovered.run_id == settled.run_id
        assert recovered.status == "completed"
        assert restarted_agent.generator.aforward.await_count == 0
    finally:
        await restarted.aclose()
        restarted.store.close()
        checkpoints.close()
        journal.close()


@pytest.mark.asyncio
async def test_final_authorization_denial_prevents_factory_and_thread_creation():
    factories = []
    checked_agents = []
    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "main", lambda _thread: factories.append("main") or AgentSession(_agent())
    )
    service.register(
        "private", lambda _thread: factories.append("private") or AgentSession(_agent())
    )

    def authorize(_ctx, request):
        checked_agents.append(request.agent_id)
        return request.agent_id == "main"

    channel = _channel(service, authorize=authorize)
    channel.register_preprocessor(
        lambda _ctx, request: msgspec.structs.replace(
            request, agent_id="private", thread_id="forbidden-thread"
        )
    )
    try:
        with pytest.raises(ChannelPermissionError):
            await channel.prompt(_request(), principal="alice", request_id="evt")
        assert checked_agents == ["main", "private"]
        assert factories == []
        assert service.threads() == ()
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_commands_and_formatting_are_presentation_only_for_checkpoint_state():
    checkpoints = SQLiteCheckpointStore(":memory:")
    agent = _agent()
    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "main", lambda _thread: AgentSession(agent, checkpoint_store=checkpoints)
    )
    channel = _channel(service)
    channel.register_command("help", lambda _ctx, _req, _args: "command result")
    channel.register_postprocessor(lambda _ctx, text: text.upper())
    try:
        command = await channel.prompt(
            _request("/help"), principal="alice", request_id="cmd-evt"
        )
        assert command.content == "COMMAND RESULT"
        assert service.threads() == ()
        admission = await channel.prompt(
            _request(), principal="alice", request_id="evt"
        )
        await service.wait("channel-thread", admission.receipt.request_id)
        state = checkpoints.load_state(
            "channel-test", "channel-thread", admission.receipt.run_id
        )
        before = msgspec.json.encode(state)
        formatted = await channel.format(admission, "display")
        assert isinstance(formatted, ChannelReply)
        assert formatted.content == "DISPLAY"
        after = checkpoints.load_state(
            "channel-test", "channel-thread", admission.receipt.run_id
        )
        assert msgspec.json.encode(after) == before
        assert agent.generator.aforward.await_count == 1
    finally:
        await service.aclose()
        service.store.close()
        checkpoints.close()


@pytest.mark.asyncio
async def test_channel_uses_native_agent_service_http_client_over_asgi():
    agent = _agent()
    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _thread: AgentSession(agent))
    app = create_service_app(service, token="integration-secret")
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as http:
        client = AgentServiceClient(
            "http://testserver", token="integration-secret", client=http
        )
        try:
            admission = await _channel(client).prompt(
                _request(), principal="alice", request_id="http-event"
            )
            receipt = await service.wait("channel-thread", admission.receipt.request_id)
            assert receipt.status == "completed"
            assert receipt.run_id == admission.receipt.run_id
            assert agent.generator.aforward.await_count == 1
        finally:
            await client.aclose()
            await service.aclose()
            service.store.close()


@pytest.mark.asyncio
async def test_channel_principal_does_not_replace_service_execution_principal():
    observed = []
    agent = _agent()

    async def answer(*_args, **_kwargs):
        observed.append(get_execution_scope().principal)
        return _response("answer")

    agent.generator.aforward = AsyncMock(side_effect=answer)
    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "main",
        lambda _thread: AgentSession(
            agent,
            scope_factory=lambda scope: scope.with_overrides(principal="service-host"),
        ),
    )
    try:
        admission = await _channel(service).prompt(
            _request(), principal="channel-user", request_id="evt"
        )
        settled = await service.wait("channel-thread", admission.receipt.request_id)
        assert settled.status == "completed"
        assert observed == ["service-host"]
    finally:
        await service.aclose()
        service.store.close()
