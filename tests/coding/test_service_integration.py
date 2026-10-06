"""Integration coverage for CodingSession over the embedded AgentService."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.coding import CodingSession
from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime.agent_inbox import AgentInbox
from msgflux.runtime.agent_inbox.providers.in_memory import InMemoryAgentInboxStore
from msgflux.runtime.context import (
    ExecutionScope,
    get_execution_context,
    get_execution_scope,
)
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.tasks import InMemoryTaskStore
from msgflux.tools.builtin import AgentTool


def _response(content: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    response.reasoning = None
    return response


def _agent(name: str, checkpoints, answer):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name=name, model=model, checkpoint_store=checkpoints)
    agent.generator.aforward = AsyncMock(side_effect=answer)
    return agent


def _service(factory):
    service = AgentService(store=SQLiteServiceStore())
    service.register("coding", factory)
    return service


async def _events(watcher):
    return [event async for event in watcher]


async def _events_until_run_end(watcher, run_id):
    events = []
    async for event in watcher:
        events.append(event)
        if (
            event.run_id == run_id
            and event.type == "run.end"
            and len(event.source_path) == 1
        ):
            return events
    return events


@pytest.mark.asyncio
async def test_shared_facades_watch_snapshot_reopen_history_and_borrow_service():
    checkpoints = InMemoryCheckpointStore()
    entered = asyncio.Event()
    release = asyncio.Event()
    observed_history = {}

    async def answer(**kwargs):
        content = kwargs["messages"].to_chatml()[-1]["content"]
        if content == "remember cobalt":
            entered.set()
            await asyncio.wait_for(release.wait(), timeout=2)
            return _response("saved cobalt")
        observed_history["messages"] = kwargs["messages"].to_chatml()
        return _response("cobalt")

    agent = _agent("coding", checkpoints, answer)
    service = _service(
        lambda _thread_id: AgentSession(agent, checkpoint_store=checkpoints)
    )
    thread = await service.open_thread("coding", thread_id="shared-coding-thread")
    first = await CodingSession.from_service(service, thread.thread_id)
    second = await CodingSession.from_service(service, thread.thread_id)

    try:
        receipt = await first.prompt("remember cobalt", request_id="first-turn")
        await asyncio.wait_for(entered.wait(), timeout=2)
        async with second.watch() as watcher:
            assert watcher.snapshot.thread_id == thread.thread_id
            assert watcher.snapshot.active_run.run_id == receipt.run_id
            release.set()
            events = await asyncio.wait_for(
                _events_until_run_end(watcher, receipt.run_id), timeout=2
            )

        assert any(event.type == "run.end" for event in events)
        assert (await asyncio.wait_for(first.wait("first-turn"), timeout=2)).status == (
            "completed"
        )
        snapshot = await second.snapshot()
        assert isinstance(snapshot.messages, ChatMessages)
        assert any(
            item.get("role") == "assistant" and item.get("content") == "saved cobalt"
            for item in snapshot.messages.to_chatml()
        )

        await first.aclose()
        next_receipt = await second.prompt("what word?", request_id="second-turn")
        assert (
            await asyncio.wait_for(
                service.wait(thread.thread_id, "second-turn"), timeout=2
            )
        ).run_id == next_receipt.run_id
        assert any(
            item.get("role") == "user" and "remember cobalt" in item.get("content", "")
            for item in observed_history["messages"]
        )
    finally:
        release.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_facade_observer_cancellation_does_not_cancel_parallel_thread_runs():
    checkpoints = InMemoryCheckpointStore()
    entered = {"thread-one": asyncio.Event(), "thread-two": asyncio.Event()}
    release = asyncio.Event()
    agents = {}

    def factory(thread_id):
        async def answer(**_kwargs):
            entered[thread_id].set()
            await asyncio.wait_for(release.wait(), timeout=2)
            return _response(thread_id)

        agent = _agent(f"coding-{thread_id}", checkpoints, answer)
        agents[thread_id] = agent
        return AgentSession(agent, checkpoint_store=checkpoints)

    service = _service(factory)
    one = await service.open_thread("coding", thread_id="thread-one")
    two = await service.open_thread("coding", thread_id="thread-two")
    first = await CodingSession.from_service(service, one.thread_id)
    second = await CodingSession.from_service(service, two.thread_id)
    try:
        first_receipt = await first.prompt("one", request_id="req-one")
        second_receipt = await second.prompt("two", request_id="req-two")
        await asyncio.wait_for(
            asyncio.gather(entered["thread-one"].wait(), entered["thread-two"].wait()),
            timeout=2,
        )

        async with first.watch() as watcher:
            observer = asyncio.create_task(_events(watcher))
            await asyncio.sleep(0)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer

        assert service.receipt(one.thread_id, "req-one").status == "running"
        release.set()
        settled = await asyncio.wait_for(
            asyncio.gather(
                service.wait(one.thread_id, "req-one"),
                service.wait(two.thread_id, "req-two"),
            ),
            timeout=2,
        )
        assert [item.status for item in settled] == ["completed", "completed"]
        assert agents["thread-one"] is not agents["thread-two"]
    finally:
        release.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_facade_preserves_injected_task_inbox_and_scope_dependencies():
    checkpoints = InMemoryCheckpointStore()
    task_store = InMemoryTaskStore()
    inbox = AgentInbox(store=InMemoryAgentInboxStore(), namespace="coding")
    workspace_scope = ExecutionScope(
        principal="coding-user",
        workspace=None,
    )
    observed = {}

    async def answer(**_kwargs):
        context = get_execution_context()
        observed.update(
            scope=get_execution_scope(),
            task_store=context["task_store"],
            agent_inbox=context["agent_inbox"],
        )
        return _response("ready")

    agent = _agent("coding", checkpoints, answer)
    session = CodingSession(
        agent,
        checkpoint_store=checkpoints,
        task_store=task_store,
        agent_inbox=inbox,
        scope_factory=lambda scope: scope.with_overrides(
            principal=workspace_scope.principal
        ),
    )
    try:
        receipt = await session.prompt("hello", request_id="dependencies-turn")
        settled = await asyncio.wait_for(session.wait("dependencies-turn"), timeout=2)

        assert settled.run_id == receipt.run_id
        assert observed["scope"].principal == "coding-user"
        assert observed["scope"].thread_id == session.thread_id
        assert observed["task_store"] is task_store
        assert observed["agent_inbox"].store is inbox.store
        assert observed["agent_inbox"].thread_id == session.thread_id
        assert observed["agent_inbox"].run_id == receipt.run_id
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_cancelled_embedded_close_keeps_journal_open_until_worker_settles():
    checkpoints = InMemoryCheckpointStore()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def answer(**_kwargs):
        entered.set()
        await asyncio.wait_for(release.wait(), timeout=2)
        return _response("settled")

    session = CodingSession(
        _agent("coding", checkpoints, answer), checkpoint_store=checkpoints
    )
    try:
        receipt = await session.prompt("wait", request_id="close-turn")
        await asyncio.wait_for(entered.wait(), timeout=2)
        closer = asyncio.create_task(session.aclose())
        await asyncio.sleep(0)
        closer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closer

        # The embedded close may still be waiting in the background; its
        # admission journal must remain available until that worker settles.
        assert (
            session.service.receipt(session.thread_id, "close-turn").run_id
            == receipt.run_id
        )
        release.set()
        await asyncio.wait_for(session.service.aclose(), timeout=2)
    finally:
        release.set()
        await session.aclose()


@pytest.mark.asyncio
async def test_stream_retains_child_events_and_waits_for_root_terminal_event():
    checkpoints = InMemoryCheckpointStore()
    root_model = Mock()
    root_model.model_type = "chat_completion"
    child_model = Mock()
    child_model.model_type = "chat_completion"

    child = Agent(name="reviewer", model=child_model)
    child.generator.aforward = AsyncMock(return_value=_response("reviewed"))

    tool_calls = ToolCallAggregator()
    tool_calls.process(
        0,
        "call_reviewer",
        "agent",
        '{"name":"reviewer","message":"Review the patch"}',
    )
    tool_response = ModelResponse()
    tool_response.set_response_type("tool_call")
    tool_response.add(tool_calls)
    root = Agent(
        name="coding",
        model=root_model,
        tools=[AgentTool(), child],
        checkpoint_store=checkpoints,
    )
    root.generator.aforward = AsyncMock(
        side_effect=[tool_response, _response("review complete")]
    )
    session = CodingSession(root, checkpoint_store=checkpoints)
    try:
        events = await asyncio.wait_for(
            _events(session.stream("Delegate review")), timeout=3
        )

        child_end = next(
            event
            for event in events
            if event.type == "run.end" and event.source_path[-1] == "agent:reviewer"
        )
        root_end = next(
            event
            for event in events
            if event.type == "run.end" and len(event.source_path) == 1
        )
        assert child_end.run_id != root_end.run_id
        assert events.index(child_end) < events.index(root_end)
    finally:
        await session.aclose()


@pytest.mark.asyncio
async def test_attaching_facade_does_not_modify_an_active_agents_hooks():
    entered, release = asyncio.Event(), asyncio.Event()

    async def answer(**_kwargs):
        entered.set()
        await release.wait()
        return _response("done")

    agent = _agent("attach-policy", InMemoryCheckpointStore(), answer)
    service = _service(lambda _thread: AgentSession(agent))
    thread = await service.open_thread("coding")
    try:
        receipt = await service.prompt(thread.thread_id, "hello", request_id="one")
        await asyncio.wait_for(entered.wait(), 2)
        assert not agent.has_extension("coding_checkpoints")
        facade = await CodingSession.from_service(service, thread.thread_id)
        assert facade.agent is agent
        assert not agent.has_extension("coding_checkpoints")
        await facade.aclose()
        release.set()
        assert (
            await service.wait(thread.thread_id, receipt.request_id)
        ).status == "completed"
    finally:
        release.set()
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_slow_stream_consumer_drains_published_events_and_terminal_retry():
    agent = _agent(
        "drain", InMemoryCheckpointStore(), lambda **_kwargs: _response("done")
    )
    session = CodingSession(agent)
    events = session.stream("hello", request_id="stable")
    try:
        first = await asyncio.wait_for(anext(events), 2)
        assert first.type == "run.start"
        # Deliberately leave the iterator suspended while its producer finishes.
        assert (await session.wait("stable")).status == "completed"
        remaining = [event async for event in events]
        types = [event.type for event in remaining]
        assert "model.response" in types
        assert "message.end" in types
        assert types[-1] == "run.end"
        duplicate = [
            event async for event in session.stream("hello", request_id="stable")
        ]
        assert duplicate == []
        assert agent.generator.aforward.await_count == 1
    finally:
        await events.aclose()
        await session.aclose()


@pytest.mark.asyncio
async def test_failed_model_settles_stream_and_preserves_input_checkpoint():
    checkpoints = InMemoryCheckpointStore()

    async def fail(**_kwargs):
        raise RuntimeError("provider unavailable")

    session = CodingSession(_agent("failure", checkpoints, fail))

    async def collect():
        return [
            event
            async for event in session.stream("retain this input", request_id="failed")
        ]

    try:
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await asyncio.wait_for(collect(), 2)
        assert session.receipt("failed").status == "failed"
        state = session.saved_state(session.receipt("failed").run_id)
        assert "retain this input" in str(state)
    finally:
        await session.aclose()
