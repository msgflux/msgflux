"""Service-level cache policy coverage across real session ownership paths."""

from __future__ import annotations

import asyncio
from threading import Event
from unittest.mock import AsyncMock, Mock

import msgflux as mf
import pytest

from msgflux.coding import CodingSession
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentWorkspace
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    ServiceBusyError,
    ServiceRecoveryRequiredError,
    SessionCachePolicy,
    SQLiteServiceStore,
)
from msgflux.tools.builtin import AgentTool


def _response(content="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _tool_call(name, arguments, call_id="call-1"):
    calls = ToolCallAggregator()
    calls.process(0, call_id, name, arguments)
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _managed_agent(name, agent_dir, answer):
    model = Mock(model_type="chat_completion")
    model.close = Mock()
    model.aclose = AsyncMock()
    agent = Agent(name=name, model=model, agent_dir=agent_dir)
    agent.generator.aforward = AsyncMock(side_effect=answer)
    return agent, model


async def _wait_until(predicate, timeout=3):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise TimeoutError("condition did not become true")
        await asyncio.sleep(0.01)


async def _root_terminal(watcher, run_id):
    events = []
    async with asyncio.timeout(3):
        async for event in watcher:
            events.append(event)
            if (
                event.run_id == run_id
                and len(event.source_path) == 1
                and event.type in {"run.end", "run.error", "run.paused"}
            ):
                return events
    raise AssertionError("watcher ended before the root run settled")


@pytest.mark.asyncio
async def test_zero_timeout_releases_after_finalizer_while_watcher_keeps_events(
    tmp_path,
):
    agent_dir = tmp_path / "zero-timeout-agent"
    observed_history = []
    agents = []

    def factory(_thread):
        generation = len(agents)

        async def answer(**kwargs):
            if generation:
                observed_history.extend(kwargs["messages"].to_chatml())
            return _response(f"generation-{generation}")

        agent, _model = _managed_agent("zero-cache", agent_dir, answer)
        agents.append(agent)
        return AgentSession(agent)

    service = AgentService(
        store=SQLiteServiceStore(),
        cache_policy=SessionCachePolicy(max_loaded=4, idle_timeout=0),
    )
    service.register("zero-cache", factory)
    thread = await service.open_thread("zero-cache", thread_id="zero-cache-thread")
    watcher_context = None
    try:
        first = await service.prompt(
            thread.thread_id, "remember jade lake", request_id="one"
        )
        assert (
            await service.wait(thread.thread_id, first.request_id)
        ).status == "completed"
        old_agent = agents[0]
        resources = old_agent._owned_threads[thread.thread_id].resources

        # Automatic close starts only after the producer and its finalizer have
        # settled; observer consumption is deliberately still pending.
        await _wait_until(lambda: resources._closed)
        assert not service._producer_tasks
        agents[0] = None
        del old_agent

        watcher_context = service.watch(thread.thread_id)
        watcher = await watcher_context.__aenter__()
        assert any(
            item.get("role") == "user"
            and "remember jade lake" in item.get("content", "")
            for item in watcher.snapshot.messages
        )

        second = await service.prompt(
            thread.thread_id, "what did I ask you to remember?", request_id="two"
        )
        assert (
            await service.wait(thread.thread_id, second.request_id)
        ).status == "completed"
        await _wait_until(
            lambda: agents[-1]._owned_threads[thread.thread_id].resources._closed
        )
        assert any(
            item.get("role") == "user"
            and "remember jade lake" in item.get("content", "")
            for item in observed_history
        )

        events = await _root_terminal(watcher, second.run_id)
        assert any(
            event.type == "run.end" and event.run_id == second.run_id
            for event in events
        )
    finally:
        if watcher_context is not None:
            await watcher_context.__aexit__(None, None, None)
        cleaner = service._cache._cleaner_task
        await service.aclose()
        assert cleaner is None or cleaner.done()
        for agent in agents:
            if agent is not None:
                await agent.aclose()


@pytest.mark.asyncio
async def test_max_loaded_rejects_new_thread_while_lease_and_foreground_are_active(
    tmp_path,
):
    entered, finish = asyncio.Event(), asyncio.Event()
    agent_dir = tmp_path / "capacity-active-agent"
    agents = []
    factory_threads = []

    def factory(thread):
        factory_threads.append(thread.thread_id)

        async def answer(**_kwargs):
            if thread.thread_id == "capacity-first":
                entered.set()
                await finish.wait()
            return _response(thread.thread_id)

        agent, _model = _managed_agent("capacity-active", agent_dir, answer)
        agents.append(agent)
        return AgentSession(agent)

    service = AgentService(
        store=SQLiteServiceStore(),
        cache_policy=SessionCachePolicy(max_loaded=1, idle_timeout=None),
    )
    service.register("capacity-active", factory)
    first = await service.open_thread("capacity-active", thread_id="capacity-first")
    second = await service.open_thread("capacity-active", thread_id="capacity-second")
    lease = await service.acquire_session(first.thread_id)
    try:
        with pytest.raises(ServiceBusyError):
            await service.acquire_session(second.thread_id)
        assert factory_threads == [first.thread_id]
        await lease.aclose()

        receipt = await service.prompt(first.thread_id, "wait", request_id="active")
        await asyncio.wait_for(entered.wait(), 3)
        with pytest.raises(ServiceBusyError):
            await service.prompt(
                second.thread_id, "must not admit", request_id="blocked"
            )
        assert factory_threads == [first.thread_id]
        with pytest.raises(KeyError):
            service.receipt(second.thread_id, "blocked")
        assert agents[0].generator.aforward.await_count == 1

        finish.set()
        assert (
            await asyncio.wait_for(service.wait(first.thread_id, receipt.request_id), 3)
        ).status == "completed"
        next_receipt = await service.prompt(
            second.thread_id, "admitted now", request_id="after"
        )
        assert (
            await asyncio.wait_for(
                service.wait(second.thread_id, next_receipt.request_id), 3
            )
        ).status == "completed"
        assert factory_threads == [first.thread_id, second.thread_id]
    finally:
        finish.set()
        await lease.aclose()
        await service.aclose()
        for agent in agents:
            await agent.aclose()


@pytest.mark.asyncio
async def test_background_child_keeps_capacity_until_real_work_finishes(tmp_path):
    child_entered = Event()
    finish_child = Event()
    agent_dir = tmp_path / "capacity-child-agent"
    factory_threads = []
    root_agents = []
    child_agents = []

    @mf.tool_config(runtime_inputs=("handle",))
    def wait_for_finish(handle):
        task = handle.get_task()
        child_entered.set()
        while not finish_child.wait(0.01):
            task.raise_if_interrupted()
        return "child finished"

    def factory(thread):
        factory_threads.append(thread.thread_id)
        if thread.thread_id != "background-thread":
            agent, _model = _managed_agent(
                "capacity-other", agent_dir, lambda **_: _response()
            )
            root_agents.append(agent)
            return AgentSession(agent)

        child = Agent(name="capacity-child", model=Mock(model_type="chat_completion"))

        def child_answer(**_kwargs):
            return (
                _tool_call("wait_for_finish", "{}")
                if not finish_child.is_set()
                else _response("finished")
            )

        child.generator.forward = Mock(side_effect=child_answer)
        child.generator.aforward = AsyncMock(side_effect=child_answer)
        child.tool_library.add(wait_for_finish)
        child_agents.append(child)

        root, _model = _managed_agent(
            "capacity-root", agent_dir, lambda **_: _response()
        )
        root.tool_library.add(mf.tool_config(allow_background=True)(AgentTool()))
        root.tool_library.add(child)
        root_calls = 0

        def root_answer(**_kwargs):
            nonlocal root_calls
            root_calls += 1
            if root_calls == 1:
                return _tool_call(
                    "agent",
                    '{"name":"capacity-child","message":"run child",'
                    '"run_in_background":true}',
                )
            return _response("foreground done")

        root.generator.aforward = AsyncMock(side_effect=root_answer)
        root_agents.append(root)
        return AgentSession(root)

    service = AgentService(
        store=SQLiteServiceStore(),
        cache_policy=SessionCachePolicy(max_loaded=1, idle_timeout=None),
    )
    service.register("capacity", factory)
    background = await service.open_thread("capacity", thread_id="background-thread")
    other = await service.open_thread("capacity", thread_id="other-thread")
    try:
        receipt = await service.prompt(
            background.thread_id, "delegate", request_id="background"
        )
        assert (
            await asyncio.wait_for(
                service.wait(background.thread_id, receipt.request_id), 3
            )
        ).status == "completed"
        assert await asyncio.to_thread(child_entered.wait, 3)
        with pytest.raises(ServiceBusyError):
            await service.prompt(other.thread_id, "must not load", request_id="blocked")
        assert factory_threads == [background.thread_id]
        with pytest.raises(KeyError):
            service.receipt(other.thread_id, "blocked")

        finish_child.set()
        await _wait_until(
            lambda: (
                bool(
                    tasks := root_agents[0]
                    ._owned_threads[background.thread_id]
                    .resources.task_store.list()
                )
                and all(
                    task.status in {"completed", "failed", "interrupted"}
                    for task in tasks
                )
            )
        )
        await _wait_until(
            lambda: (
                root_agents[0]._active_background_future_reason(
                    (root_agents[0].tool_library,)
                )
                is None
            )
        )
        next_receipt = await service.prompt(
            other.thread_id, "now load", request_id="after"
        )
        assert (
            await asyncio.wait_for(
                service.wait(other.thread_id, next_receipt.request_id), 3
            )
        ).status == "completed"
        assert factory_threads == [background.thread_id, other.thread_id]
    finally:
        finish_child.set()
        await service.aclose()
        for agent in root_agents:
            await agent.aclose()
        for agent in child_agents:
            await agent.aclose()


@pytest.mark.asyncio
async def test_capacity_lru_reload_keeps_history_and_quarantined_binding_uses_slot(
    tmp_path,
):
    agent_dir = tmp_path / "capacity-lru-agent"
    generations = {"lru-first": 0, "lru-second": 0, "quarantined": 0}
    observed = []
    agents = []

    def factory(thread):
        thread_id = thread.thread_id
        generation = generations[thread_id]
        generations[thread_id] += 1

        async def answer(**kwargs):
            if thread_id == "lru-first" and generation:
                observed.extend(kwargs["messages"].to_chatml())
            return _response(f"{thread_id}-{generation}")

        agent, _model = _managed_agent("cache-lru", agent_dir, answer)
        agents.append(agent)
        if thread_id == "quarantined":

            def fail_close():
                raise RuntimeError("host cleanup failed")

            return AgentSession(agent, on_close=fail_close)
        return AgentSession(agent)

    service = AgentService(
        store=SQLiteServiceStore(),
        cache_policy=SessionCachePolicy(max_loaded=1, idle_timeout=None),
    )
    service.register("cache-lru", factory)
    first = await service.open_thread("cache-lru", thread_id="lru-first")
    second = await service.open_thread("cache-lru", thread_id="lru-second")
    broken = await service.open_thread("cache-lru", thread_id="quarantined")
    try:
        first_receipt = await service.prompt(
            first.thread_id, "remember opal field", request_id="first"
        )
        assert (
            await service.wait(first.thread_id, first_receipt.request_id)
        ).status == "completed"
        second_receipt = await service.prompt(
            second.thread_id, "touch capacity", request_id="second"
        )
        assert (
            await service.wait(second.thread_id, second_receipt.request_id)
        ).status == "completed"
        reloaded = await service.prompt(
            first.thread_id, "continue", request_id="reload"
        )
        assert (
            await service.wait(first.thread_id, reloaded.request_id)
        ).status == "completed"
        assert generations[first.thread_id] == 2
        assert any(
            item.get("role") == "user"
            and "remember opal field" in item.get("content", "")
            for item in observed
        )

        await service.release_session(first.thread_id)
        broken_lease = await service.acquire_session(broken.thread_id)
        await broken_lease.aclose()
        with pytest.raises(ServiceRecoveryRequiredError):
            await service.release_session(broken.thread_id)
        with pytest.raises(ServiceBusyError):
            await service.acquire_session(second.thread_id)
        assert generations[second.thread_id] == 1
        assert generations["quarantined"] == 1
    finally:
        with pytest.raises(ExceptionGroup):
            await service.aclose()
        for agent in agents:
            await agent.aclose()


@pytest.mark.asyncio
async def test_zero_timeout_borrows_model_and_workspace_and_unused_service_is_lazy(
    tmp_path,
):
    workspace_root = tmp_path / "borrowed-workspace"
    workspace_root.mkdir()
    workspace = AgentWorkspace.local(workspace_root)
    workspace.aclose = AsyncMock()
    model = Mock(model_type="chat_completion")
    model.close = Mock()
    model.aclose = AsyncMock()
    agent = Agent(name="borrowed-cache", model=model)
    agent.workspace = workspace
    agent.generator.aforward = AsyncMock(return_value=_response("done"))
    host_closes = 0

    def on_close():
        nonlocal host_closes
        host_closes += 1

    service = AgentService(
        store=SQLiteServiceStore(),
        cache_policy=SessionCachePolicy(max_loaded=2, idle_timeout=0),
    )
    service.register(
        "borrowed-cache",
        lambda _thread: AgentSession(agent, on_close=on_close),
    )
    thread = await service.open_thread("borrowed-cache", thread_id="borrowed-cache")
    unused_dir = tmp_path / "never-bound"
    unused = AgentService(
        store=SQLiteServiceStore(),
        cache_policy=SessionCachePolicy(max_loaded=1, idle_timeout=0.01),
    )
    unused.register(
        "never-bound",
        lambda _thread: AgentSession(
            _managed_agent("never-bound", unused_dir, lambda **_: _response())[0]
        ),
    )
    unused_thread = await unused.open_thread("never-bound", thread_id="never-bound")
    try:
        assert not unused_dir.exists()
        # The cleaner is lazy: an unused cache has no running background task.
        assert unused._cache._cleaner_task is None
        receipt = await service.prompt(thread.thread_id, "hello", request_id="one")
        assert (
            await service.wait(thread.thread_id, receipt.request_id)
        ).status == "completed"
        await _wait_until(lambda: host_closes == 1)
        assert not model.close.called
        assert model.aclose.await_count == 0
        assert workspace.aclose.await_count == 0
        assert not unused_dir.exists()
        assert unused_thread.thread_id == "never-bound"
    finally:
        await service.aclose()
        await unused.aclose()
        assert unused._cache._cleaner_task is None
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_embedded_coding_session_disables_automatic_release(tmp_path):
    agent, _model = _managed_agent(
        "embedded-cache", tmp_path / "embedded-agent", lambda **_: _response()
    )
    session = CodingSession(agent, thread_id="embedded-cache")
    try:
        assert session.service._cache._policy is None
        assert session.service._cache._cleaner_task is None
        receipt = await session.prompt("retain this binding", request_id="one")
        assert (await session.wait(receipt.request_id)).status == "completed"
        resources = agent._owned_threads[session.thread_id].resources
        await asyncio.sleep(0.05)
        assert not resources._closed
        assert session.service._cache.sessions[session.thread_id] is session._binding
    finally:
        await session.aclose()
        await agent.aclose()
