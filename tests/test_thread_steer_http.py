"""HTTP thread steering: one route decides queued vs started atomically."""

import asyncio

import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.hooks import Hook
from msgflux.runtime.service import (
    AgentSession,
    ServiceBusyError,
    ThreadSteerResult,
)
from msgflux.runtime.service.http import AgentServiceClient, AgentServiceHTTPError
from tests.test_agent_service_http_integration import (
    TOKEN,
    _agent,
    _response,
    _server,
)


@pytest.mark.asyncio
async def test_idle_thread_steer_starts_a_run_and_returns_its_identities():
    checkpoints = InMemoryCheckpointStore()
    agent = _agent(
        lambda **_kwargs: _response("started answer"), checkpoints=checkpoints
    )
    async with _server(
        lambda _thread: AgentSession(agent, checkpoint_store=checkpoints)
    ) as (service, url):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            thread = await client.open_thread("main")
            result = await client.steer_thread(thread.thread_id, "fresh turn")

            assert isinstance(result, ThreadSteerResult)
            assert result.status == "started"
            assert result.thread_id == thread.thread_id
            assert result.request_id and result.notification_id is None
            settled = await service.wait(thread.thread_id, result.request_id)
            assert settled.status == "completed"
            assert settled.run_id == result.run_id
            assert len(await client.runs(thread.thread_id)) == 1
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_active_thread_steer_queues_into_the_same_run():
    entered, release = asyncio.Event(), asyncio.Event()
    history = []

    async def lookup(query: str):
        entered.set()
        await release.wait()
        return query

    calls = ToolCallAggregator()
    calls.process(0, "lookup-1", "lookup", '{"query":"value"}')
    tool_response = ModelResponse()
    tool_response.set_response_type("tool_call")
    tool_response.add(calls)

    async def answer(**kwargs):
        history.append(kwargs["messages"].to_chatml())
        return tool_response if len(history) == 1 else _response("steered")

    checkpoints = InMemoryCheckpointStore()
    agent = _agent(answer, tools=[lookup], checkpoints=checkpoints)
    async with _server(
        lambda _thread: AgentSession(agent, checkpoint_store=checkpoints)
    ) as (service, url):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            thread = await client.open_thread("main")
            receipt = await client.prompt(
                thread.thread_id, "look up", request_id="thread-steer"
            )
            await asyncio.wait_for(entered.wait(), 3)
            result = await client.steer_thread(
                thread.thread_id, "Use the existing tests"
            )

            assert result.status == "queued"
            assert result.run_id == receipt.run_id
            assert result.notification_id and result.request_id is None

            release.set()
            assert (
                await service.wait(thread.thread_id, receipt.request_id)
            ).status == "completed"
            assert "Use the existing tests" in str(history[-1])
            assert len(await client.runs(thread.thread_id)) == 1
            # The run-scoped route keeps its strict, per-run semantics.
            with pytest.raises(ServiceBusyError) as error:
                await client.steer(thread.thread_id, receipt.run_id, "after settle")
            assert isinstance(error.value.__cause__, AgentServiceHTTPError)
            assert error.value.__cause__.status_code == 409
        finally:
            release.set()
            await client.aclose()


@pytest.mark.asyncio
async def test_sealed_run_lets_thread_steer_retry_after_settlement():
    hook_entered, release_hook = asyncio.Event(), asyncio.Event()

    async def after_run_end(_context):
        hook_entered.set()
        await release_hook.wait()

    checkpoints = InMemoryCheckpointStore()
    agent = _agent(lambda **_kwargs: _response("finished"), checkpoints=checkpoints)
    agent.register_lifecycle_hook(
        "after_run_end", Hook(event="after_run_end", handler=after_run_end)
    )
    async with _server(
        lambda _thread: AgentSession(agent, checkpoint_store=checkpoints)
    ) as (service, url):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            thread = await client.open_thread("main")
            receipt = await client.prompt(
                thread.thread_id, "finish", request_id="sealed-thread-steer"
            )
            await asyncio.wait_for(hook_entered.wait(), 5)

            with pytest.raises(ServiceBusyError) as caught:
                await client.steer_thread(thread.thread_id, "too late")
            assert caught.value.__cause__.status_code == 409
            assert "no longer accepts steering" in str(caught.value)
            assert agent.generator.aforward.await_count == 1

            release_hook.set()
            assert (
                await service.wait(thread.thread_id, receipt.request_id)
            ).status == "completed"
            restarted = await client.steer_thread(thread.thread_id, "next turn")
            assert restarted.status == "started"
            assert restarted.run_id != receipt.run_id
        finally:
            release_hook.set()
            await client.aclose()


@pytest.mark.asyncio
async def test_thread_steer_rejects_blank_content_and_unknown_thread():
    checkpoints = InMemoryCheckpointStore()
    agent = _agent(lambda **_kwargs: _response("unused"), checkpoints=checkpoints)
    async with _server(
        lambda _thread: AgentSession(agent, checkpoint_store=checkpoints)
    ) as (_service, url):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            thread = await client.open_thread("main")
            with pytest.raises(ValueError):
                await client.steer_thread(thread.thread_id, "   ")
            with pytest.raises(AgentServiceHTTPError) as error:
                await client.steer_thread("missing", "hello")
            assert error.value.status_code == 404
            assert agent.generator.aforward.await_count == 0
            assert await client.runs(thread.thread_id) == ()
        finally:
            await client.aclose()
