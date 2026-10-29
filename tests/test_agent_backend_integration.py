"""Integration coverage for the host-owned shared SQLite Agent backend."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict
from unittest.mock import AsyncMock, Mock

import msgflux as mf
import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentApprovals, SQLiteAgentBackend
from msgflux.runtime.context import ExecutionScope, get_execution_context
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    SessionCachePolicy,
    SQLiteServiceStore,
)
from msgflux.tools.builtin import AgentTool

CACHE_BYTES = 2 * 1024 * 1024


def _response(content: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _tool_call(name: str, arguments: str) -> ModelResponse:
    calls = ToolCallAggregator()
    calls.process(0, "call-1", name, arguments)
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _agent(name, *, backend=None, agent_dir=None, answer=None):
    model = Mock(model_type="chat_completion")
    agent = Agent(name=name, model=model, backend=backend, agent_dir=agent_dir)
    agent.generator.aforward = AsyncMock(
        side_effect=answer or (lambda **_kwargs: _response("done"))
    )
    return agent


def _scope(thread_id, namespace, run_id, *, principal="user"):
    return ExecutionScope(
        thread_id=thread_id,
        namespace=namespace,
        run_id=run_id,
        principal=principal,
    )


def _history(messages):
    return messages.to_chatml()


async def _wait_until(predicate, timeout=5):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() >= deadline:
            raise TimeoutError("condition did not become true")
        await asyncio.sleep(0.01)


def _background_status(binding):
    tasks = binding.task_store.list()
    return tasks[0].status if tasks else None


def test_sqlite_agent_backend_is_public_runtime_export():
    from msgflux import runtime

    assert runtime.SQLiteAgentBackend is SQLiteAgentBackend


@pytest.mark.asyncio
async def test_backend_is_lazy_and_busy_close_keeps_shared_pool_usable(tmp_path):
    root = tmp_path / "agent"
    backend = SQLiteAgentBackend(root, cache_bytes=CACHE_BYTES)
    assert backend.connection_count == 0
    assert backend.active_bindings == 0
    assert backend.has_thread("thread-a") is False
    assert not root.exists()

    model = Mock(model_type="chat_completion")
    model.close = Mock()
    model.aclose = AsyncMock()
    first = Agent(name="first", model=model, backend=backend)
    first.generator.aforward = AsyncMock(
        side_effect=lambda **_kwargs: _response("first")
    )
    second = _agent("second", backend=backend, answer=lambda **_: _response("second"))
    try:
        assert (
            await first.acall("hello", scope=_scope("thread-a", "first", "run-a"))
            == "first"
        )
        assert backend.connection_count == 4
        assert backend.active_bindings == 1

        assert (
            await second.acall("hello", scope=_scope("thread-b", "second", "run-b"))
            == "second"
        )
        assert backend.connection_count == 4
        assert backend.active_bindings == 2
        assert backend.has_thread("thread-a") is True
        assert backend.has_thread("thread-b") is True

        # Refusing a busy close must not poison the backend or its live leases.
        with pytest.raises(RuntimeError, match="binding leases are active"):
            backend.close()
        assert backend.connection_count == 4
        assert backend.active_bindings == 2
        assert (
            await second.acall("again", scope=_scope("thread-b", "second", "run-b2"))
            == "second"
        )
    finally:
        await first.aclose()
        await second.aclose()

    assert backend.active_bindings == 0
    assert backend.connection_count == 4
    backend.close()
    assert backend.connection_count == 0
    assert backend.active_bindings == 0
    assert root.is_dir()
    # Backend and Agent close only release persistence; the host still owns the model.
    model.close.assert_not_called()
    model.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_agents_keep_exact_thread_history_and_survive_peer_close(
    tmp_path,
):
    backend = SQLiteAgentBackend(tmp_path / "agent", cache_bytes=CACHE_BYTES)
    names = ("alpha", "beta", "gamma")
    threads = {name: f"thread-{name}" for name in names}
    barrier = asyncio.Barrier(len(names))
    seen = defaultdict(list)
    agents = {}

    for name in names:
        thread_id = threads[name]

        async def answer(_name=name, _thread_id=thread_id, **kwargs):
            if not seen[_name]:
                await barrier.wait()
            seen[_name].append(_history(kwargs["messages"]))
            return _response(f"{_name}-{len(seen[_name])}")

        agents[name] = _agent(name, backend=backend, answer=answer)

    async def prompt(name, content, run_prefix):
        return await agents[name].acall(
            content,
            scope=_scope(threads[name], name, f"{run_prefix}-{name}"),
        )

    try:
        first_results = await asyncio.gather(
            *(prompt(name, f"hello-{name}", "first") for name in names)
        )
        assert set(first_results) == {f"{name}-1" for name in names}
        assert backend.connection_count == 4
        assert backend.active_bindings == 3
        for name in names:
            assert seen[name] == [[{"role": "user", "content": f"hello-{name}"}]]

        await agents["alpha"].aclose()
        assert backend.active_bindings == 2
        assert backend.connection_count == 4
        assert backend.has_thread(threads["alpha"]) is True
        assert backend.has_thread("thread-unknown") is False

        second_results = await asyncio.gather(
            *(prompt(name, f"again-{name}", "second") for name in ("beta", "gamma"))
        )
        assert set(second_results) == {"beta-2", "gamma-2"}
        for name in ("beta", "gamma"):
            assert seen[name][-1] == [
                {"role": "user", "content": f"hello-{name}"},
                {"role": "assistant", "content": f"{name}-1"},
                {"role": "user", "content": f"again-{name}"},
            ]
            assert all("alpha" not in str(item) for item in seen[name][-1])
    finally:
        for name in ("beta", "gamma"):
            await agents[name].aclose()
        backend.close()

    assert backend.connection_count == 0


@pytest.mark.asyncio
async def test_fresh_backend_reopens_completed_history_after_close(tmp_path):
    root = tmp_path / "agent"
    first_backend = SQLiteAgentBackend(root, cache_bytes=CACHE_BYTES)
    first = _agent(
        "assistant", backend=first_backend, answer=lambda **_: _response("remembered")
    )
    try:
        assert (
            await first.acall(
                "Remember the phrase silver fern",
                scope=_scope("durable-thread", "assistant", "run-first"),
            )
            == "remembered"
        )
        assert first_backend.has_thread("durable-thread") is True
    finally:
        await first.aclose()
        first_backend.close()

    assert first_backend.connection_count == 0
    assert first_backend.active_bindings == 0

    second_backend = SQLiteAgentBackend(root, cache_bytes=CACHE_BYTES)
    assert second_backend.has_thread("durable-thread") is True
    observed = {}

    async def answer(**kwargs):
        observed["history"] = _history(kwargs["messages"])
        return _response("silver fern")

    second = _agent("assistant", backend=second_backend, answer=answer)
    try:
        assert (
            await second.acall(
                "What phrase did I ask you to remember?",
                scope=_scope("durable-thread", "assistant", "run-second"),
            )
            == "silver fern"
        )
        assert observed["history"] == [
            {"role": "user", "content": "Remember the phrase silver fern"},
            {"role": "assistant", "content": "remembered"},
            {"role": "user", "content": "What phrase did I ask you to remember?"},
        ]
    finally:
        await second.aclose()
        second_backend.close()


@pytest.mark.asyncio
async def test_backend_rejects_conflicts_and_never_mixes_old_layout(tmp_path):
    model = Mock(model_type="chat_completion")
    backend = SQLiteAgentBackend(tmp_path / "shared", cache_bytes=CACHE_BYTES)

    with pytest.raises(ValueError, match="backend cannot be combined"):
        Agent(
            name="conflict",
            model=model,
            backend=backend,
            agent_dir=tmp_path / "other",
        )
    with pytest.raises(ValueError, match="backend cannot be combined"):
        Agent(
            name="conflict",
            model=model,
            backend=backend,
            checkpoint_store=InMemoryCheckpointStore(),
        )

    shared_root = tmp_path / "shared"
    shared = _agent("shared", backend=backend, answer=lambda **_: _response("shared"))
    try:
        await shared.acall("hello", scope=_scope("shared-thread", "shared", "run"))
    finally:
        await shared.aclose()

    legacy_on_shared = _agent(
        "legacy",
        agent_dir=shared_root,
        answer=lambda **_: _response("legacy"),
    )
    try:
        with pytest.raises(ValueError, match="shared SQLite Agent backend layout"):
            await legacy_on_shared.acall(
                "hello", scope=_scope("legacy-thread", "legacy", "run")
            )
        assert not (shared_root / "threads" / "legacy-thread").exists()
    finally:
        await legacy_on_shared.aclose()
    backend.close()

    legacy_root = tmp_path / "legacy"
    legacy = _agent(
        "legacy", agent_dir=legacy_root, answer=lambda **_: _response("legacy")
    )
    try:
        await legacy.acall("hello", scope=_scope("legacy-thread", "legacy", "run"))
        assert (legacy_root / "threads" / "legacy-thread").is_dir()
    finally:
        await legacy.aclose()

    rejected_backend = SQLiteAgentBackend(legacy_root, cache_bytes=CACHE_BYTES)
    rejected = _agent(
        "legacy", backend=rejected_backend, answer=lambda **_: _response()
    )
    try:
        with pytest.raises(ValueError, match="legacy per-thread Agent stores"):
            await rejected.acall(
                "hello", scope=_scope("legacy-thread", "legacy", "run")
            )
        assert not (legacy_root / "runtime" / "backend").exists()
    finally:
        await rejected.aclose()
        rejected_backend.close()


@pytest.mark.asyncio
async def test_service_passive_observation_and_idle_reload_preserve_history(tmp_path):
    root = tmp_path / "agent"
    backend = SQLiteAgentBackend(root, cache_bytes=CACHE_BYTES)
    journal = SQLiteServiceStore(tmp_path / "service.sqlite3")
    generations = defaultdict(int)
    observed = []
    agents = defaultdict(list)

    def factory(thread):
        thread_id = thread.thread_id
        generation = generations[thread_id]
        generations[thread_id] += 1

        async def answer(**kwargs):
            if thread_id == "service-a" and generation:
                observed.extend(_history(kwargs["messages"]))
            return _response(f"{thread_id}:{generation}")

        agent = _agent("service-agent", backend=backend, answer=answer)
        agents[thread_id].append(agent)
        return AgentSession(agent)

    service = AgentService(
        store=journal,
        cache_policy=SessionCachePolicy(max_loaded=1, idle_timeout=0),
    )
    service.register("service-agent", factory)
    first = await service.open_thread("service-agent", thread_id="service-a")
    second = await service.open_thread("service-agent", thread_id="service-b")
    try:
        # Observation may construct the session but must not create durable state.
        snapshot = await service.snapshot(second.thread_id)
        assert snapshot.thread_id == second.thread_id
        assert not root.exists()
        assert backend.has_thread(second.thread_id) is False
        assert backend.active_bindings == 0
        assert agents[second.thread_id][-1]._owned_threads == {}

        receipt = await service.prompt(
            first.thread_id, "remember opal field", request_id="first"
        )
        assert (
            await asyncio.wait_for(service.wait(first.thread_id, receipt.request_id), 3)
        ).status == "completed"
        assert backend.has_thread(first.thread_id) is True
        assert backend.active_bindings == 1
        old_binding = (
            agents[first.thread_id][0]._owned_threads[first.thread_id].resources
        )
        await _wait_until(lambda: backend.active_bindings == 0)
        assert old_binding._closed

        # Loading a different thread evicts the idle first thread under max_loaded=1.
        third = await service.prompt(
            second.thread_id, "touch capacity", request_id="second"
        )
        assert (
            await asyncio.wait_for(service.wait(second.thread_id, third.request_id), 3)
        ).status == "completed"

        resumed = await service.prompt(
            first.thread_id, "what did I remember?", request_id="third"
        )
        assert (
            await asyncio.wait_for(service.wait(first.thread_id, resumed.request_id), 3)
        ).status == "completed"
        assert generations[first.thread_id] >= 2
        assert any(
            item.get("role") == "user"
            and "remember opal field" in item.get("content", "")
            for item in observed
        )
        assert any(
            item.get("role") == "assistant" and item.get("content") == "service-a:0"
            for item in observed
        )
    finally:
        await service.aclose()
        assert backend.active_bindings == 0
        assert backend.connection_count == 4
        backend.close()
        journal.close()


@pytest.mark.asyncio
async def test_approval_pause_and_resume_use_backend_journal(tmp_path):
    backend = SQLiteAgentBackend(tmp_path / "agent", cache_bytes=CACHE_BYTES)
    calls = []

    def lookup(query: str) -> str:
        calls.append(query)
        return "found"

    agent = Agent(
        name="reviewer",
        model=Mock(model_type="chat_completion"),
        tools=[lookup],
        backend=backend,
    )
    thread_id, run_id = "approval-thread", "approval-run"
    scope = _scope(thread_id, "reviewer", run_id)
    binding = agent._bind_resources(thread_id)
    policy = AgentApprovals(binding.approval_store, {"lookup": "v1"}, "policy-v1")
    agent.generator.aforward = AsyncMock(
        side_effect=[
            _tool_call("lookup", '{"query":"secret"}'),
            _response("approved"),
        ]
    )
    try:
        with pytest.raises(TaskPauseRequestedError):
            await agent.acall("lookup", scope=scope, approvals=policy)
        assert calls == []
        pending = binding.approval_store.pending("reviewer", thread_id, run_id)
        assert len(pending) == 1
        assert pending[0].status == "pending"
        assert (
            binding.checkpoint_store.load_state("reviewer", thread_id, run_id)["status"]
            == "paused"
        )

        agent.decide_approval(
            pending[0].request_id, approved=True, decided_by="host", approvals=policy
        )
        assert await agent.acall("ignored", scope=scope, approvals=policy) == "approved"
        assert calls == ["secret"]
        assert binding.approval_store.pending("reviewer", thread_id, run_id) == []
    finally:
        await agent.aclose()
        backend.close()


@pytest.mark.asyncio
async def test_background_children_share_pool_but_not_cancellation_or_results(tmp_path):
    backend = SQLiteAgentBackend(tmp_path / "agent", cache_bytes=CACHE_BYTES)
    entered_a, release_a = threading.Event(), threading.Event()
    entered_b, release_b = threading.Event(), threading.Event()
    thread_a, thread_b = "background-a", "background-b"
    run_a, run_b = "run-background-a", "run-background-b"

    def make_root(name, child_name, entered, release):
        @mf.tool_config(runtime_inputs=("handle",))
        def wait_for_release(handle):
            task = handle.get_task()
            entered.set()
            while not release.wait(0.01):
                task.raise_if_interrupted()
            return "child finished"

        child = Agent(name=child_name, model=Mock(model_type="chat_completion"))
        child.generator.forward = Mock(
            side_effect=[
                _tool_call("wait_for_release", "{}"),
                _response("child done"),
            ]
        )
        child.generator.aforward = AsyncMock(
            side_effect=[
                _tool_call("wait_for_release", "{}"),
                _response("child done"),
            ]
        )
        child.tool_library.add(wait_for_release)

        root = Agent(
            name=name, model=Mock(model_type="chat_completion"), backend=backend
        )
        root.tool_library.add(mf.tool_config(allow_background=True)(AgentTool()))
        root.tool_library.add(child)
        root.generator.aforward = AsyncMock(
            side_effect=[
                _tool_call(
                    "agent",
                    json.dumps(
                        {
                            "name": child_name,
                            "message": "run child",
                            "run_in_background": True,
                        }
                    ),
                ),
                _response("delegated"),
            ]
        )
        return root, child

    root_a, child_a = make_root("root-a", "child-a", entered_a, release_a)
    root_b, child_b = make_root("root-b", "child-b", entered_b, release_b)
    try:
        results = await asyncio.gather(
            root_a.acall("delegate", scope=_scope(thread_a, "root-a", run_a)),
            root_b.acall("delegate", scope=_scope(thread_b, "root-b", run_b)),
        )
        assert results == ["delegated", "delegated"]
        assert await asyncio.to_thread(entered_a.wait, 3)
        assert await asyncio.to_thread(entered_b.wait, 3)

        binding_a = root_a._owned_threads[thread_a].resources
        binding_b = root_b._owned_threads[thread_b].resources
        task_a = binding_a.task_store.list()[0]
        task_b = binding_b.task_store.list()[0]
        assert task_a.status == task_b.status == "running"
        assert task_a.metadata["thread_id"] == thread_a
        assert task_b.metadata["thread_id"] == thread_b
        assert binding_a.task_store.get(task_b.task_id) is None
        assert binding_b.task_store.get(task_a.task_id) is None
        assert backend.active_bindings == 2
        assert backend.connection_count == 4

        binding_a.task_store.request_interrupt(task_a.task_id)
        await _wait_until(lambda: _background_status(binding_a) == "interrupted")
        await _wait_until(lambda: root_a._owned_threads[thread_a].active_calls == 0)
        assert _background_status(binding_b) == "running"
        future_b = root_b.tool_library.get_background_dispatcher().get_task_future(
            task_b.task_id
        )
        assert future_b is not None and not future_b.done()
        assert any(
            notice.source == "task"
            and notice.ref == task_a.task_id
            and notice.status == "interrupted"
            for notice in binding_a.agent_inbox.fork(run_id=run_a).peek()
        )

        # Closing one binding must not read another thread's result payloads or
        # touch its future. A malformed result row proves close stays projected.
        binding_a.task_store._conn.execute(
            "UPDATE tasks SET result='not-json' WHERE task_id=?", (task_a.task_id,)
        )
        binding_a.task_store._conn.commit()
        await root_a.aclose()
        assert backend.active_bindings == 1
        assert backend.connection_count == 4
        assert _background_status(binding_b) == "running"
        assert not future_b.done()

        release_b.set()
        await _wait_until(lambda: _background_status(binding_b) == "completed")
        assert any(
            notice.source == "task"
            and notice.ref == task_b.task_id
            and notice.status == "completed"
            for notice in binding_b.agent_inbox.fork(run_id=run_b).peek()
        )
        assert (
            binding_b.checkpoint_store.load_state("root-b", thread_b, run_b)["status"]
            == "completed"
        )
    finally:
        release_a.set()
        release_b.set()
        await root_a.aclose()
        await root_b.aclose()
        await child_a.aclose()
        await child_b.aclose()
        backend.close()


@pytest.mark.asyncio
async def test_nested_agents_inherit_binding_and_reject_another_backend(tmp_path):
    backend = SQLiteAgentBackend(tmp_path / "parent", cache_bytes=CACHE_BYTES)
    observed = {}

    async def child_answer(**_kwargs):
        context = get_execution_context()
        observed["child"] = (
            context["scope"].thread_id,
            context["checkpoint_store"],
            context["task_store"],
        )
        return _response("child answer")

    child = _agent("child", answer=child_answer)
    parent = _agent(
        "parent",
        backend=backend,
        answer=lambda **_: _response("parent answer"),
    )
    parent.tool_library.add(AgentTool())
    parent.tool_library.add(child)
    parent.generator.aforward = AsyncMock(
        side_effect=[
            _tool_call(
                "agent", json.dumps({"name": "child", "message": "handle this"})
            ),
            _response("parent answer"),
        ]
    )
    try:
        assert (
            await parent.acall(
                "delegate", scope=_scope("nested-thread", "parent", "run-parent")
            )
            == "parent answer"
        )
        binding = parent._owned_threads["nested-thread"].resources
        assert observed["child"][0] == "nested-thread"
        assert observed["child"][1] is binding.checkpoint_store
        assert observed["child"][2] is binding.task_store
        assert child._owned_threads == {}
        assert backend.active_bindings == 1
    finally:
        await parent.aclose()
        await child.aclose()
        backend.close()

    other_backend = SQLiteAgentBackend(tmp_path / "child", cache_bytes=CACHE_BYTES)
    root_backend = SQLiteAgentBackend(tmp_path / "root", cache_bytes=CACHE_BYTES)
    other_child = _agent(
        "other-child", backend=other_backend, answer=lambda **_: _response("other")
    )
    root = _agent("root", backend=root_backend, answer=lambda **_: _response())
    root.tool_library.add(AgentTool())
    root.tool_library.add(other_child)
    seen = []

    async def root_answer(**kwargs):
        seen.append(_history(kwargs["messages"]))
        return (
            _tool_call(
                "agent",
                json.dumps({"name": "other-child", "message": "handle this"}),
            )
            if len(seen) == 1
            else _response("parent done")
        )

    root.generator.aforward = AsyncMock(side_effect=root_answer)
    try:
        assert (
            await root.acall(
                "delegate", scope=_scope("other-thread", "root", "run-other")
            )
            == "parent done"
        )
        assert any(
            item.get("role") == "tool"
            and "Nested Agent cannot replace its persistence backend"
            in item.get("content", "")
            for item in seen[-1]
        )
        assert other_child._owned_threads == {}
        assert other_backend.active_bindings == 0
    finally:
        await root.aclose()
        await other_child.aclose()
        other_backend.close()
        root_backend.close()
