import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.coding import CodingSession
from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.modules.agent import Agent
from msgflux.runtime import AgentApprovals, InMemoryApprovalStore
from msgflux.runtime.context import (
    ExecutionScope,
    get_execution_context,
    get_execution_scope,
)
from msgflux.runtime.service import SQLiteServiceStore


@pytest.mark.asyncio
async def test_session_stream_keeps_thread_and_injects_durable_store():
    store = InMemoryCheckpointStore()
    observed = {}

    async def respond(**kwargs):
        observed["prompt"] = kwargs["messages"].to_chatml()[-1]["content"]
        observed["scope"] = get_execution_scope()
        observed["context"] = get_execution_context()
        return _text_response("reply")

    agent = _coding_agent(store, respond)
    session = CodingSession(agent, checkpoint_store=store)

    events = [event async for event in session.stream("hello")]

    assert observed["prompt"] == "hello"
    assert observed["scope"].thread_id == session.thread_id
    assert observed["scope"].namespace == "coding"
    assert observed["context"]["checkpoint_store"] is store
    assert any(event.type == "run.end" for event in events)
    assert session.thread_id.startswith("thd_")


@pytest.mark.asyncio
async def test_session_scope_factory_can_add_execution_environment():
    initial = ExecutionScope(thread_id="fixed-thread", namespace="coding")
    observed = {}

    async def respond(**_kwargs):
        observed["scope"] = get_execution_scope()
        return _text_response("reply")

    agent = _coding_agent(InMemoryCheckpointStore(), respond)
    session = CodingSession(
        agent,
        thread_id="fixed-thread",
        scope_factory=lambda scope: ExecutionScope(
            thread_id=scope.thread_id,
            namespace=scope.namespace,
            abort_signal=scope.abort_signal,
            principal="ui",
        ),
    )

    [event async for event in session.stream("hello")]

    assert observed["scope"].principal == "ui"
    assert observed["scope"].abort_signal is not None
    assert initial.thread_id == session.thread_id


@pytest.mark.asyncio
async def test_cancel_aborts_current_stream():
    started = asyncio.Event()

    async def respond(**_kwargs):
        started.set()
        scope = get_execution_scope()
        await scope.abort_signal.wait()
        scope.abort_signal.raise_if_aborted()
        return _text_response("unreachable")

    session = CodingSession(_coding_agent(InMemoryCheckpointStore(), respond))
    stream = session.stream("wait")
    first = await anext(stream)

    async def consume_rest():
        return [event async for event in stream]

    task = asyncio.create_task(consume_rest())
    await started.wait()
    await session.cancel(first.run_id)

    with pytest.raises(RuntimeError):
        await task
    await stream.aclose()


def _text_response(text):
    response = Mock(spec=ModelResponse)
    response.response_type = "text_generation"
    response.data = text
    response.reasoning = None
    response.metadata = {"model": "scripted"}
    response.consume.return_value = text
    return response


def _coding_agent(store, scripted_forward):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="coding", model=model, checkpoint_store=store)
    agent.generator.aforward = AsyncMock(side_effect=scripted_forward)
    return agent


@pytest.mark.asyncio
async def test_new_session_resumes_durable_thread_and_snapshot_history():
    store = InMemoryCheckpointStore()
    first_agent = _coding_agent(store, lambda **_kwargs: _text_response("first reply"))
    first = CodingSession(first_agent, checkpoint_store=store)
    thread_id = first.thread_id

    [event async for event in first.stream("remember the word: cobalt")]

    seen_on_second_call = {}

    async def answer_from_history(**kwargs):
        messages = kwargs["messages"]
        assert isinstance(messages, ChatMessages)
        seen_on_second_call["chat"] = messages.to_chatml()
        return _text_response("cobalt")

    second_agent = _coding_agent(store, answer_from_history)
    reconnected = CodingSession(
        second_agent,
        thread_id=thread_id,
        checkpoint_store=store,
    )

    events = [event async for event in reconnected.stream("what word?")]
    snapshot = await reconnected.snapshot()

    assert reconnected.thread_id == thread_id
    assert reconnected.namespace == "coding"
    assert any(
        item.get("role") == "user" and "cobalt" in item.get("content", "")
        for item in seen_on_second_call["chat"]
    )
    assert any(
        item.get("role") == "assistant" and item.get("content") == "first reply"
        for item in seen_on_second_call["chat"]
    )
    assert snapshot.thread_id == thread_id
    assert isinstance(snapshot.messages, ChatMessages)
    assert snapshot.messages.to_chatml()[-1]["content"] == "cobalt"
    assert any(event.type == "run.end" for event in events)


@pytest.mark.asyncio
async def test_prompt_with_stable_request_id_does_not_dispatch_twice():
    store = InMemoryCheckpointStore()
    calls = []

    async def respond(**_kwargs):
        calls.append(1)
        return _text_response("once")

    session = CodingSession(_coding_agent(store, respond), checkpoint_store=store)
    first = await session.prompt("input", request_id="request-1")
    settled = await session.wait("request-1")
    duplicate = await session.prompt("input", request_id="request-1")

    assert settled.status == "completed"
    assert duplicate.run_id == first.run_id
    assert len(calls) == 1
    await session.aclose()


@pytest.mark.asyncio
async def test_embedded_session_borrows_supplied_service_store():
    service_store = SQLiteServiceStore()
    session = CodingSession(
        _coding_agent(
            InMemoryCheckpointStore(), lambda **_kwargs: _text_response("ok")
        ),
        service_store=service_store,
    )

    await session.aclose()

    assert service_store.threads()[0].thread_id == session.thread_id
    service_store.close()


def _tool_response():
    calls = ToolCallAggregator()
    calls.process(0, "call_lookup", "lookup", '{"query":"secret"}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


@pytest.mark.asyncio
async def test_resume_approval_pause_replays_tool_batch_once():
    checkpoint_store = InMemoryCheckpointStore()
    approval_store = InMemoryApprovalStore()
    calls = []

    def lookup(query):
        calls.append(query)
        return "found"

    model = Mock()
    model.model_type = "chat_completion"
    policy = AgentApprovals(approval_store, {"lookup": "v1"}, "policy-v1")
    agent = Agent(
        name="coding",
        model=model,
        tools=[lookup],
        checkpoint_store=checkpoint_store,
        approvals=policy,
    )
    agent.generator.aforward = AsyncMock(
        side_effect=[_tool_response(), _text_response("done")]
    )
    session = CodingSession(
        agent,
        checkpoint_store=checkpoint_store,
        scope_factory=lambda scope: ExecutionScope(
            thread_id=scope.thread_id,
            namespace=scope.namespace,
            abort_signal=scope.abort_signal,
            principal="human",
        ),
    )

    paused_events = []
    with pytest.raises(TaskPauseRequestedError):
        async for event in session.stream("lookup the secret"):
            paused_events.append(event)
    assert any(event.type == "run.paused" for event in paused_events)
    assert calls == []
    paused_run = next(
        event.run_id for event in paused_events if event.type == "run.paused"
    )
    record = approval_store.pending("coding", session.thread_id, paused_run)[0]
    paused_state = checkpoint_store.load_state("coding", session.thread_id, paused_run)
    assert paused_state["status"] == "paused"
    assert paused_state["runtime"]["extensions"]["pending_approvals"]
    await agent.adecide_approval(
        record.request_id,
        approved=True,
        decided_by="human",
    )

    thread_id = session.thread_id
    await session.aclose()
    reopened = CodingSession(
        agent,
        thread_id=thread_id,
        checkpoint_store=checkpoint_store,
        scope_factory=lambda scope: ExecutionScope(
            thread_id=scope.thread_id,
            namespace=scope.namespace,
            abort_signal=scope.abort_signal,
            principal="human",
        ),
    )
    resumed_events = [
        event async for event in reopened.resume(paused_run, worker_stopped=True)
    ]

    assert any(event.type == "run.end" for event in resumed_events)
    assert calls == ["secret"], [
        (event.type, event.run_id, dict(event.data)) for event in resumed_events
    ]
    assert agent.generator.aforward.call_count == 2
    assert (
        checkpoint_store.load_state("coding", session.thread_id, paused_run)["status"]
        == "completed"
    )
    await reopened.aclose()


@pytest.mark.asyncio
async def test_detaching_observer_does_not_cancel_service_owned_run():
    release = asyncio.Event()
    finished = asyncio.Event()

    async def respond(**_kwargs):
        await release.wait()
        finished.set()
        return _text_response("survived observer detach")

    session = CodingSession(_coding_agent(InMemoryCheckpointStore(), respond))
    events = session.stream("run")
    first = await anext(events)
    run_id = first.run_id
    await events.aclose()
    reattached = await CodingSession.from_service(session.service, session.thread_id)
    replayed = []

    async def collect_reattached():
        async with reattached.watch() as watcher:
            async for event in watcher:
                if event.run_id == run_id:
                    replayed.append(event)
                if (
                    event.type == "run.end"
                    and event.run_id == run_id
                    and len(event.source_path) == 1
                ):
                    return

    observer = asyncio.create_task(collect_reattached())
    await asyncio.sleep(0)
    release.set()
    await asyncio.wait_for(finished.wait(), timeout=2)
    await asyncio.wait_for(observer, timeout=2)
    receipt = session.service.receipt_for_run(session.thread_id, run_id)
    settled = await session.service.wait(session.thread_id, receipt.request_id)
    assert settled.status == "completed"
    assert any(event.type == "run.end" for event in replayed)
    assert await reattached.snapshot()
    await session.aclose()


@pytest.mark.asyncio
async def test_cancelled_aclose_wait_keeps_joining_before_store_close():
    started = asyncio.Event()
    shutdown_requested = asyncio.Event()
    release = asyncio.Event()

    async def respond(**_kwargs):
        started.set()
        scope = get_execution_scope()
        await scope.abort_signal.wait()
        shutdown_requested.set()
        await release.wait()
        return _text_response("settled")

    session = CodingSession(_coding_agent(InMemoryCheckpointStore(), respond))
    stream = session.stream("close")
    await anext(stream)
    await started.wait()
    closing = asyncio.create_task(session.aclose())
    await shutdown_requested.wait()
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    # The owned SQLite journal is still live while the shielded shutdown joins
    # the foreground worker.
    assert session._service_store.threads()
    release.set()
    await session.aclose()
    await stream.aclose()
