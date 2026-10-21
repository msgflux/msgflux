"""Integration coverage for Agent-managed per-thread durable resources."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.coding import CodingSession
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_context,
)
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.tools.builtin import AgentTool


def _response(content: str = "saved") -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _agent(name: str, agent_dir, answer):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name=name, model=model, agent_dir=agent_dir)
    agent.generator.aforward = AsyncMock(side_effect=answer)
    return agent


@pytest.mark.asyncio
async def test_agent_dir_lazily_persists_and_reopens_thread_history(tmp_path):
    agent_dir = tmp_path / "assistant-state"
    thread_id = "durable-thread"
    first = _agent("assistant", agent_dir, lambda **_kwargs: _response("remembered"))
    assert not agent_dir.exists()

    await first.acall(
        "Remember the phrase silver fern",
        scope=ExecutionScope(thread_id=thread_id, run_id="run_first"),
    )
    thread_dir = agent_dir / "threads" / thread_id
    assert (thread_dir / "checkpoints.sqlite3").is_file()
    assert (thread_dir / "tasks.sqlite3").is_file()
    assert (thread_dir / "inbox.sqlite3").is_file()
    await first.aclose()

    observed = {}

    async def answer_from_history(**kwargs):
        observed["messages"] = kwargs["messages"].to_chatml()
        return _response("silver fern")

    reopened = _agent("assistant", agent_dir, answer_from_history)
    try:
        await reopened.acall(
            "What phrase did I ask you to remember?",
            scope=ExecutionScope(thread_id=thread_id, run_id="run_second"),
        )
        assert any(
            item.get("role") == "user" and "silver fern" in item.get("content", "")
            for item in observed["messages"]
        )
    finally:
        await reopened.aclose()


@pytest.mark.asyncio
async def test_coding_session_defers_disk_creation_until_first_prompt_and_reopens(
    tmp_path,
):
    agent_dir = tmp_path / "coding-state"
    thread_id = "coding-thread"
    first_agent = _agent("coding", agent_dir, lambda **_kwargs: _response("done"))
    first = CodingSession(first_agent, thread_id=thread_id)
    try:
        assert first.checkpoint_store is None
        snapshot = await first.snapshot()
        assert snapshot.thread_id == thread_id
        assert not agent_dir.exists()

        receipt = await first.prompt("Remember the phrase blue heron")
        settled = await asyncio.wait_for(first.wait(receipt.request_id), timeout=3)
        assert settled.status == "completed"
        assert (agent_dir / "runtime" / "service.sqlite3").is_file()
        assert (agent_dir / "threads" / thread_id / "checkpoints.sqlite3").is_file()
    finally:
        await first.aclose()

    observed = {}

    async def answer_from_history(**kwargs):
        observed["messages"] = kwargs["messages"].to_chatml()
        return _response("blue heron")

    second_agent = _agent("coding", agent_dir, answer_from_history)
    second = CodingSession(second_agent, thread_id=thread_id)
    try:
        assert second.checkpoint_store is not None
        receipt = await second.prompt("What phrase did I ask you to remember?")
        settled = await asyncio.wait_for(second.wait(receipt.request_id), timeout=3)
        assert settled.status == "completed"
        assert any(
            item.get("role") == "user" and "blue heron" in item.get("content", "")
            for item in observed["messages"]
        )
    finally:
        await second.aclose()


@pytest.mark.asyncio
async def test_agent_service_manages_isolated_thread_resources(tmp_path):
    agent_dir = tmp_path / "service-state"
    agents = {}

    def make_session(thread):
        async def answer(**_kwargs):
            context = get_execution_context()
            agents[thread.thread_id + "_stores"] = (
                context["checkpoint_store"],
                context["task_store"],
                context["agent_inbox"].store,
            )
            return _response(thread.thread_id)

        agent = _agent("service-agent", agent_dir, answer)
        agents[thread.thread_id] = agent
        return AgentSession(agent)

    service = AgentService(store=SQLiteServiceStore())
    service.register("service-agent", make_session)
    first = await service.open_thread("service-agent", thread_id="service-one")
    second = await service.open_thread("service-agent", thread_id="service-two")
    try:
        # Observation may construct the session, but it must not provision a new
        # thread's persistent files before work is admitted.
        await service.snapshot(first.thread_id)
        assert not (agent_dir / "threads" / first.thread_id).exists()

        for thread in (first, second):
            receipt = await service.prompt(
                thread.thread_id,
                "hello",
                request_id=f"request-{thread.thread_id}",
            )
            settled = await asyncio.wait_for(
                service.wait(thread.thread_id, receipt.request_id), timeout=3
            )
            assert settled.status == "completed"

        first_stores = agents[first.thread_id + "_stores"]
        second_stores = agents[second.thread_id + "_stores"]
        assert first_stores[0] is not second_stores[0]
        assert first_stores[1] is not second_stores[1]
        assert first_stores[2] is not second_stores[2]
        assert (agent_dir / "threads" / first.thread_id / "approvals.sqlite3").is_file()
        assert (
            agent_dir / "threads" / second.thread_id / "approvals.sqlite3"
        ).is_file()
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_nested_agent_inherits_parent_thread_resources(tmp_path):
    agent_dir = tmp_path / "nested-state"
    observed = {}

    async def child_answer(**_kwargs):
        context = get_execution_context()
        observed["child"] = (
            context["scope"].thread_id,
            context["checkpoint_store"],
            context["task_store"],
            context["agent_inbox"].store,
        )
        return _response("child answer")

    child = _agent("child", None, child_answer)
    calls = ToolCallAggregator()
    calls.process(
        0,
        "call_child",
        "agent",
        '{"name":"child","message":"handle this"}',
    )
    tool_response = ModelResponse()
    tool_response.set_response_type("tool_call")
    tool_response.add(calls)
    parent_final = _response("parent answer")
    parent_observed = {}

    async def parent_answer(**_kwargs):
        context = get_execution_context()
        parent_observed["stores"] = (
            context["checkpoint_store"],
            context["task_store"],
            context["agent_inbox"].store,
        )
        calls = parent_observed.get("calls", 0)
        parent_observed["calls"] = calls + 1
        return tool_response if calls == 0 else parent_final

    parent = _agent("parent", agent_dir, parent_answer)
    parent.tool_library.add(AgentTool())
    parent.tool_library.add(child)
    try:
        await parent.acall(
            "Delegate",
            scope=ExecutionScope(thread_id="nested-thread", run_id="run_parent"),
        )
        assert observed["child"][0] == "nested-thread"
        assert observed["child"][1:] == parent_observed["stores"]
        child_runs = parent_observed["stores"][0].list_runs("child", "nested-thread")
        assert len(child_runs) == 1
        assert child_runs[0]["run_id"] != "run_parent"
        assert list((agent_dir / "threads").iterdir()) == [
            agent_dir / "threads" / "nested-thread"
        ]
    finally:
        await parent.aclose()
        await child.aclose()


def test_managed_agent_storage_rejects_explicit_store_combinations(tmp_path):
    model = Mock()
    model.model_type = "chat_completion"
    with pytest.raises(ValueError, match="agent_dir cannot be combined"):
        Agent(
            name="managed",
            model=model,
            agent_dir=tmp_path / "agent",
            checkpoint_store=InMemoryCheckpointStore(),
        )
    assert not (tmp_path / "agent").exists()


@pytest.mark.asyncio
async def test_concurrent_threads_bind_distinct_stores_and_inbox_namespaces(tmp_path):
    agent_dir = tmp_path / "concurrent"
    barrier = asyncio.Barrier(2)
    observed = {}

    async def answer(**_kwargs):
        await barrier.wait()
        context = get_execution_context()
        thread_id = context["scope"].thread_id
        observed[thread_id] = (
            context["checkpoint_store"],
            context["task_store"],
            context["agent_inbox"],
        )
        return _response(thread_id)

    agent = _agent("concurrent-agent", agent_dir, answer)
    try:

        async def run(thread_id):
            await agent.acall(
                "hello",
                scope=ExecutionScope(thread_id=thread_id, run_id=f"run-{thread_id}"),
            )

        await asyncio.gather(run("thread-one"), run("thread-two"))
        one = observed["thread-one"]
        two = observed["thread-two"]
        assert all(left is not right for left, right in zip(one, two, strict=True))
        assert (one[2].namespace, one[2].thread_id) == (
            "concurrent-agent",
            "thread-one",
        )
        assert (two[2].namespace, two[2].thread_id) == (
            "concurrent-agent",
            "thread-two",
        )
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_generic_service_reopens_managed_threads_under_same_root(tmp_path):
    agent_dir = tmp_path / "service-reopen"
    thread_id = "same-thread"

    def make_service(answer):
        agent = _agent("generic-service-agent", agent_dir, answer)
        service = AgentService(store=SQLiteServiceStore())
        service.register("generic-service-agent", lambda _thread: AgentSession(agent))
        return agent, service

    first_agent, first_service = make_service(lambda **_kwargs: _response("persistent"))
    try:
        thread = await first_service.open_thread(
            "generic-service-agent", thread_id=thread_id
        )
        receipt = await first_service.prompt(
            thread.thread_id, "remember", request_id="request-first"
        )
        assert (
            await first_service.wait(thread.thread_id, receipt.request_id)
        ).status == ("completed")
    finally:
        await first_service.aclose()
        await first_agent.aclose()

    observed = {}

    async def reopened_answer(**kwargs):
        observed["messages"] = kwargs["messages"].to_chatml()
        return _response("reopened")

    second_agent, second_service = make_service(reopened_answer)
    try:
        thread = await second_service.open_thread(
            "generic-service-agent", thread_id=thread_id
        )
        receipt = await second_service.prompt(
            thread.thread_id, "what did I say?", request_id="request-second"
        )
        assert (
            await second_service.wait(thread.thread_id, receipt.request_id)
        ).status == ("completed")
        assert any(
            "remember" in item.get("content", "")
            for item in observed["messages"]
            if item.get("role") == "user"
        )
    finally:
        await second_service.aclose()
        await second_agent.aclose()


@pytest.mark.asyncio
async def test_child_with_different_agent_dir_is_rejected_before_child_files(tmp_path):
    parent_dir = tmp_path / "parent-state"
    child_dir = tmp_path / "child-state"
    child = _agent("child", child_dir, lambda **_kwargs: _response("child"))
    calls = ToolCallAggregator()
    calls.process(0, "call_child", "agent", '{"name":"child","message":"go"}')
    tool_response = ModelResponse()
    tool_response.set_response_type("tool_call")
    tool_response.add(calls)
    parent_calls = 0

    async def parent_answer(**_kwargs):
        nonlocal parent_calls
        parent_calls += 1
        return tool_response if parent_calls == 1 else _response("done")

    parent = _agent("parent", parent_dir, parent_answer)
    parent.tool_library.add(AgentTool())
    parent.tool_library.add(child)
    try:
        await parent.acall(
            "delegate",
            scope=ExecutionScope(thread_id="parent-thread", run_id="run-parent"),
        )
        assert not child_dir.exists()
    finally:
        await parent.aclose()
        await child.aclose()


@pytest.mark.asyncio
async def test_manual_inherited_stores_rejected_before_managed_directory_creation(
    tmp_path,
):
    agent_dir = tmp_path / "conflicting-agent"
    agent = _agent("managed", agent_dir, lambda **_kwargs: _response())
    manual_store = InMemoryCheckpointStore()
    try:
        with execution_context(
            scope=ExecutionScope(thread_id="manual-thread"),
            checkpoint_store=manual_store,
        ):
            with pytest.raises(
                ValueError, match="conflicts with inherited checkpoint_store"
            ):
                await agent.acall("hello")
        assert not agent_dir.exists()
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_empty_coding_session_closes_without_creating_files(tmp_path):
    agent_dir = tmp_path / "empty-session"
    agent = _agent("empty", agent_dir, lambda **_kwargs: _response())
    session = CodingSession(agent, thread_id="never-prompted")
    await session.aclose()
    await agent.aclose()
    assert not agent_dir.exists()


@pytest.mark.asyncio
async def test_coding_session_does_not_close_borrowed_manual_store(tmp_path):
    store = InMemoryCheckpointStore()
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="borrowed", model=model, checkpoint_store=store)
    session = CodingSession(agent, thread_id="borrowed-thread")
    await session.aclose()
    store.save_state("borrowed", "borrowed-thread", "still-open", {"ok": True})
    assert store.load_state("borrowed", "borrowed-thread", "still-open") == {"ok": True}


@pytest.mark.asyncio
async def test_direct_sync_call_reopens_metadata_and_watch_without_execution(tmp_path):
    response = _response("remembered")
    response.set_metadata(
        {"model": {"provider": "openai-codex", "model_id": "gpt-6-luna"}}
    )
    agent_dir = tmp_path / "sync-state"
    first = _agent("sync-agent", agent_dir, lambda **_kwargs: response)
    first.generator.forward = Mock(return_value=response)
    try:
        assert (
            first.forward(
                "remember this",
                scope=ExecutionScope(thread_id="sync-thread", run_id="sync-run"),
            )
            == "remembered"
        )
    finally:
        await first.aclose()

    reopened = _agent("sync-agent", agent_dir, lambda **_kwargs: _response())
    try:
        assert reopened.get_last_model_metadata(
            scope=ExecutionScope(thread_id="sync-thread")
        ) == {"provider": "openai-codex", "model_id": "gpt-6-luna"}
        async with reopened.watch("sync-thread") as observer:
            assert any(
                "remember this" in str(item) for item in observer.snapshot.messages
            )
        # Missing history queries remain inert rather than provisioning a thread.
        assert (
            reopened.get_last_model_metadata(
                scope=ExecutionScope(thread_id="never-started")
            )
            is None
        )
        assert not (agent_dir / "threads" / "never-started").exists()
    finally:
        await reopened.aclose()


def _exit_without_cleanup(agent_dir, result_pipe):
    import os

    agent = _agent("crash-agent", agent_dir, lambda **_kwargs: _response())
    agent.generator.forward = Mock(return_value=_response("saved before exit"))
    agent.forward(
        "Remember the phrase abrupt exit",
        scope=ExecutionScope(thread_id="crash-thread", run_id="crash-run"),
    )
    result_pipe.send("committed")
    os._exit(7)


@pytest.mark.asyncio
async def test_managed_agent_history_survives_process_exit_without_cleanup(tmp_path):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    agent_dir = tmp_path / "crash-state"
    process = context.Process(target=_exit_without_cleanup, args=(agent_dir, send))
    process.start()
    send.close()
    try:
        assert await asyncio.to_thread(receive.poll, 10)
        assert receive.recv() == "committed"
        await asyncio.to_thread(process.join, 10)
        assert not process.is_alive()
        assert process.exitcode == 7
    finally:
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 5)
        receive.close()
        process.close()

    observed = {}

    async def recover(**kwargs):
        observed["messages"] = kwargs["messages"].to_chatml()
        return _response("recovered")

    reopened = _agent("crash-agent", agent_dir, recover)
    try:
        await reopened.acall(
            "What did I ask you to remember?",
            scope=ExecutionScope(thread_id="crash-thread", run_id="recover-run"),
        )
        assert any("abrupt exit" in str(item) for item in observed["messages"])
    finally:
        await reopened.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("streamed", [False, True])
async def test_managed_storage_preserves_thread_from_message_envelope(
    tmp_path, streamed
):
    from msgflux.chat_messages import ChatMessages
    from msgflux.core.message import Message

    model = Mock(model_type="chat_completion")
    agent = Agent(
        name="envelope",
        model=model,
        agent_dir=tmp_path / "state",
        message_fields={"task": "content", "messages": "context.history"},
    )
    agent.generator.aforward = AsyncMock(return_value=_response("saved"))
    message = Message(content="hello")
    message.set("context.history", ChatMessages(thread_id="envelope-thread"))
    try:
        if streamed:
            events = [event async for event in agent.stream_events(message)]
            assert events[0].data["thread_id"] == "envelope-thread"
        else:
            await agent.acall(message)
        assert set(agent._owned_threads) == {"envelope-thread"}
    finally:
        await agent.aclose()
