"""Tests for the remote AgentSessionClient facade."""

import asyncio
from uuid import UUID
from unittest.mock import AsyncMock, Mock

import httpx2
import msgspec
import pytest

from msgflux.runtime.service.http import AgentSessionClient
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    RunSummary,
    SQLiteServiceStore,
    ServiceThread,
)
from msgflux.runtime.service.http import AgentServiceClient, create_service_app

TOKEN = "coding-client-test-token"


def _response(content: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _agent(answer):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="main", model=model, checkpoint_store=InMemoryCheckpointStore())
    agent.generator.aforward = AsyncMock(side_effect=answer)
    return agent


@pytest.mark.asyncio
async def test_open_prompt_receipt_wait_snapshot_and_borrowed_client(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    async def answer(**_kwargs):
        entered.set()
        await release.wait()
        return _response("done")

    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _thread: AgentSession(_agent(answer)))
    app = create_service_app(service, token=TOKEN)
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(
        transport=transport, base_url="http://service"
    ) as http:
        client = AgentServiceClient("http://service", token=TOKEN, client=http)
        session = await AgentSessionClient.open(
            client, thread_id="project-thread", cwd=tmp_path
        )
        assert session.thread_id == "project-thread"
        assert session.agent_id == "main"
        assert session.workspace_root == str(tmp_path)

        receipt = await session.prompt("hello", request_id="stable-request")
        await asyncio.wait_for(entered.wait(), timeout=2)
        duplicate = await session.prompt("hello", request_id="stable-request")
        assert duplicate.run_id == receipt.run_id
        assert (await session.receipt("stable-request")).status == "running"
        snapshot = await session.snapshot()
        assert snapshot.thread_id == session.thread_id

        polled = asyncio.Event()
        receipt_call = client.receipt

        async def track_receipt(thread_id, request_id):
            current = await receipt_call(thread_id, request_id)
            polled.set()
            return current

        client.receipt = track_receipt
        waiter = asyncio.create_task(session.wait("stable-request", poll_interval=0.01))
        await asyncio.wait_for(polled.wait(), timeout=2)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not release.is_set()
        assert (await session.receipt("stable-request")).status == "running"
        assert service.receipt(session.thread_id, "stable-request").status == "running"

        release.set()
        settled = await asyncio.wait_for(session.wait("stable-request"), timeout=2)
        assert settled.status == "completed"
        runs = await session.runs()
        assert isinstance(runs, tuple)
        assert runs[0].run_id == settled.run_id
        assert (await session.latest_run()).status == "completed"
        assert await session.aclose() is None
        # The facade does not close the injected HTTP client.
        assert not http.is_closed
    await service.aclose()


@pytest.mark.asyncio
async def test_wait_validates_poll_interval():
    client = Mock(spec=AgentServiceClient)
    client.receipt = AsyncMock()
    thread = ServiceThread("thread", "main")
    session = AgentSessionClient(client, thread)
    for interval in (0, -1, float("inf"), float("nan"), True, "0.1"):
        with pytest.raises(ValueError, match="finite positive"):
            await session.wait("request", poll_interval=interval)
    client.receipt.assert_not_awaited()


def test_constructor_requires_client_and_valid_thread_record():
    client = Mock(spec=AgentServiceClient)
    with pytest.raises(TypeError, match="AgentServiceClient"):
        AgentSessionClient("http://service", ServiceThread("thread", "main"))
    with pytest.raises(ValueError, match="thread_id"):
        AgentSessionClient(client, ServiceThread(" ", "main"))
    with pytest.raises(ValueError, match="agent_id"):
        AgentSessionClient(client, ServiceThread("thread", " "))
    with pytest.raises(msgspec.ValidationError):
        AgentSessionClient(
            client,
            {"thread_id": "thread", "agent_id": "main", "extra": "rejected"},
        )


@pytest.mark.asyncio
async def test_forwarding_controls_and_run_summaries():
    client = Mock(spec=AgentServiceClient)
    client.interrupt = AsyncMock(return_value=True)
    client.steer = AsyncMock(return_value={"content": "focus"})
    client.resume_checkpoint = AsyncMock(return_value="receipt")
    client.runs = AsyncMock(return_value=(RunSummary("run-1", "paused", 1.0),))
    thread = ServiceThread("thread", "main", "/workspace")
    session = AgentSessionClient(client, thread)

    assert await session.cancel("run-1") is True
    assert await session.steer("run-1", "focus") == {"content": "focus"}
    assert await session.resume("run-1") == "receipt"
    assert await session.runs() == (RunSummary("run-1", "paused", 1.0),)
    assert await session.latest_run() == RunSummary("run-1", "paused", 1.0)
    client.interrupt.assert_awaited_once_with("thread", "run-1")
    client.steer.assert_awaited_once_with("thread", "run-1", "focus")
    client.resume_checkpoint.assert_awaited_once_with("thread", "run-1")
    assert client.runs.await_count == 2


@pytest.mark.asyncio
async def test_prompt_generates_request_id_and_watch_delegates():
    client = Mock(spec=AgentServiceClient)
    client.prompt = AsyncMock(return_value="receipt")
    watcher_context = object()
    client.watch = Mock(return_value=watcher_context)
    session = AgentSessionClient(client, ServiceThread("thread", "main"))

    assert await session.prompt("hello") == "receipt"
    request_id = client.prompt.await_args.kwargs["request_id"]
    UUID(request_id)
    client.prompt.assert_awaited_once_with("thread", "hello", request_id=request_id)
    assert session.watch() is watcher_context
    client.watch.assert_called_once_with("thread")


@pytest.mark.asyncio
async def test_latest_run_is_none_when_no_summaries():
    client = Mock(spec=AgentServiceClient)
    client.runs = AsyncMock(return_value=())
    session = AgentSessionClient(client, ServiceThread("thread", "main"))
    assert await session.latest_run() is None
