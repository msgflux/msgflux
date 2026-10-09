"""Loaded AgentSession release and reload integration coverage."""

import asyncio
import gc
import weakref
from threading import Event
from unittest.mock import AsyncMock, Mock

import pytest

import msgflux as mf
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.coding import CodingSession
from msgflux.nn import Agent
from msgflux.runtime.context import ExecutionScope, get_execution_context
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    ServiceBusyError,
    ServiceRecoveryRequiredError,
    SQLiteServiceStore,
)
from msgflux.runtime import AgentWorkspace, PermissionSet
from msgflux.tools.builtin import AgentTool


def _response(content="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    response.reasoning = None
    return response


def _tool_call(name, arguments, call_id="call-1"):
    calls = ToolCallAggregator()
    calls.process(0, call_id, name, arguments)
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _agent(name, answer, *, agent_dir=None):
    model = Mock(model_type="chat_completion")
    model.close = Mock()
    model.aclose = AsyncMock()
    agent = Agent(name=name, model=model, agent_dir=agent_dir)
    agent.generator.aforward = AsyncMock(side_effect=answer)
    return agent, model


async def _release_when_idle(service, thread_id, timeout=3):
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            return await service.release_session(thread_id)
        except ServiceBusyError:
            if asyncio.get_running_loop().time() >= deadline:
                raise
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_managed_release_reloads_new_agent_and_preserves_history(tmp_path):
    agent_dir = tmp_path / "agent-state"
    journal = SQLiteServiceStore(tmp_path / "service.sqlite3")
    service = AgentService(store=journal)
    agents = []
    observed_history = []

    def factory(thread):
        index = len(agents)

        async def answer(**kwargs):
            if index:
                observed_history.extend(kwargs["messages"].to_chatml())
            return _response(f"answer-{index}")

        agent, model = _agent("release-agent", answer, agent_dir=agent_dir)
        agents.append((agent, model))
        return AgentSession(agent)

    service.register("release-agent", factory)
    thread = await service.open_thread("release-agent", thread_id="release-history")
    try:
        first_receipt = await service.prompt(
            thread.thread_id, "remember amber fern", request_id="first"
        )
        assert (
            await service.wait(thread.thread_id, first_receipt.request_id)
        ).status == "completed"
        first_agent, first_model = agents[0]
        first_resources = first_agent._owned_threads[thread.thread_id].resources

        assert await service.release_session(thread.thread_id) is True
        assert first_resources._closed
        assert first_model.close.call_count == 0
        assert first_model.aclose.await_count == 0
        assert {item.thread_id for item in service.threads()} == {thread.thread_id}
        assert service.receipt(thread.thread_id, "first").status == "completed"

        lease = await service.acquire_session(thread.thread_id)
        try:
            second_agent = lease.session.agent
            assert second_agent is not first_agent
            second_receipt = await service.prompt(
                thread.thread_id,
                "what phrase did I ask you to remember?",
                request_id="second",
            )
            settled = await service.wait(thread.thread_id, second_receipt.request_id)
            assert settled.status == "completed"
            assert any(
                item.get("role") == "user"
                and "remember amber fern" in item.get("content", "")
                for item in observed_history
            )
        finally:
            await lease.aclose()
        assert await service.release_session(thread.thread_id) is True
        assert await service.release_session(thread.thread_id) is False
    finally:
        await service.aclose()
        journal.close()
        for agent, model in agents:
            await agent.aclose()
            assert model.close.call_count == 0
            assert model.aclose.await_count == 0


@pytest.mark.asyncio
async def test_lease_pins_session_and_release_is_idempotent_after_detach():
    agent, _model = _agent("lease", lambda **_kwargs: _response())
    service = AgentService(store=SQLiteServiceStore())
    service.register("lease", lambda _thread: AgentSession(agent))
    thread = await service.open_thread("lease", thread_id="lease-pin")
    try:
        lease = await service.acquire_session(thread.thread_id)
        assert lease.session.agent is agent
        with pytest.raises(ServiceBusyError):
            await service.release_session(thread.thread_id)

        await lease.aclose()
        await lease.aclose()
        assert await service.release_session(thread.thread_id) is True
        assert await service.release_session(thread.thread_id) is False
    finally:
        await service.aclose()
        await agent.aclose()


@pytest.mark.asyncio
async def test_attached_service_watcher_does_not_pin_released_session():
    agent, _model = _agent("watch-pins", lambda **_kwargs: _response())
    service = AgentService(store=SQLiteServiceStore())
    service.register("watch-pins", lambda _thread: AgentSession(agent))
    thread = await service.open_thread("watch-pins", thread_id="watch-pin")
    try:
        async with service.watch(thread.thread_id) as watcher:
            assert watcher.snapshot.thread_id == thread.thread_id
            assert await service.release_session(thread.thread_id) is True
        assert await service.release_session(thread.thread_id) is False
    finally:
        await service.aclose()
        await agent.aclose()


@pytest.mark.asyncio
async def test_local_watcher_spans_managed_release_and_reload(tmp_path):
    agent_dir = tmp_path / "watch-reload-agent"
    service = AgentService(store=SQLiteServiceStore())
    agents = []
    observed_history = []

    def factory(_thread):
        generation = len(agents)

        async def answer(**kwargs):
            if generation:
                observed_history.extend(kwargs["messages"].to_chatml())
            return _response(f"answer-{generation}")

        agent, _model = _agent("watch-reload", answer, agent_dir=agent_dir)
        agents.append(agent)
        return AgentSession(agent)

    service.register("watch-reload", factory)
    thread = await service.open_thread("watch-reload", thread_id="watch-reload")
    watcher_context = None
    try:
        first = await service.prompt(
            thread.thread_id, "remember cobalt", request_id="one"
        )
        assert (
            await service.wait(thread.thread_id, first.request_id)
        ).status == "completed"
        old_agent = agents[0]
        old_agent_ref = weakref.ref(old_agent)
        old_resources = old_agent._owned_threads[thread.thread_id].resources

        watcher_context = service.watch(thread.thread_id)
        watcher = await watcher_context.__aenter__()
        assert any(
            item.get("role") == "user" and "remember cobalt" in item.get("content", "")
            for item in watcher.snapshot.messages
        )
        assert await service.release_session(thread.thread_id) is True
        assert old_resources._closed
        agents[0] = None
        del old_agent
        gc.collect()
        assert old_agent_ref() is None

        second = await service.prompt(
            thread.thread_id, "what did I ask?", request_id="two"
        )
        assert (
            await service.wait(thread.thread_id, second.request_id)
        ).status == "completed"
        assert await service.release_session(thread.thread_id) is True
        assert any(
            item.get("role") == "user" and "remember cobalt" in item.get("content", "")
            for item in observed_history
        )

        events = []
        async with asyncio.timeout(3):
            async for event in watcher:
                events.append(event)
                if event.run_id == second.run_id and event.type in {
                    "run.end",
                    "run.error",
                }:
                    break
        assert any(event.run_id == second.run_id for event in events)
        assert any(
            event.run_id == second.run_id and event.type == "run.end"
            for event in events
        )
    finally:
        if watcher_context is not None:
            await watcher_context.__aexit__(None, None, None)
        await service.aclose()
        for agent in agents:
            if agent is not None:
                await agent.aclose()


@pytest.mark.asyncio
async def test_release_rejects_worker_finalizer_after_worker_map_is_cleared():
    finalizer_entered = asyncio.Event()
    finish_finalizer = asyncio.Event()
    agent, _model = _agent("finalizer", lambda **_kwargs: _response())
    service = AgentService(store=SQLiteServiceStore())
    service.register("finalizer", lambda _thread: AgentSession(agent))
    thread = await service.open_thread("finalizer", thread_id="finalizer-window")

    async def paused_execution(_record, _worker):
        return "paused"

    async def hold_finalizer(_worker, _key):
        finalizer_entered.set()
        await finish_finalizer.wait()

    service._execute = paused_execution
    service._resume_quiescent_policy = hold_finalizer
    try:
        receipt = await service.prompt(thread.thread_id, "pause", request_id="paused")
        await asyncio.wait_for(finalizer_entered.wait(), timeout=3)
        assert not service._workers
        assert service._producer_tasks
        with pytest.raises(ServiceBusyError):
            await service.release_session(thread.thread_id)

        finish_finalizer.set()
        assert (
            await service.wait(thread.thread_id, receipt.request_id)
        ).status == "paused"
        for _ in range(10):
            if not service._producer_tasks:
                break
            await asyncio.sleep(0)
        assert not service._producer_tasks
        assert await service.release_session(thread.thread_id) is True
    finally:
        finish_finalizer.set()
        await service.aclose()
        await agent.aclose()


@pytest.mark.asyncio
async def test_same_thread_acquire_is_singleflight_without_blocking_other_threads():
    first_factory_entered = asyncio.Event()
    allow_first_factory = asyncio.Event()
    factory_calls = {}
    agents = []

    async def factory(thread):
        factory_calls[thread.thread_id] = factory_calls.get(thread.thread_id, 0) + 1
        if thread.thread_id == "slow-bind":
            first_factory_entered.set()
            await allow_first_factory.wait()
        agent, _model = _agent(thread.thread_id, lambda **_kwargs: _response())
        agents.append(agent)
        return AgentSession(agent)

    service = AgentService(store=SQLiteServiceStore())
    service.register("singleflight", factory)
    slow = await service.open_thread("singleflight", thread_id="slow-bind")
    independent = await service.open_thread(
        "singleflight", thread_id="independent-bind"
    )
    first = asyncio.create_task(service.acquire_session(slow.thread_id))
    duplicate = asyncio.create_task(service.acquire_session(slow.thread_id))
    await asyncio.wait_for(first_factory_entered.wait(), timeout=2)
    independent_acquire = asyncio.create_task(
        service.acquire_session(independent.thread_id)
    )
    try:
        independent_lease = await asyncio.wait_for(independent_acquire, timeout=2)
        assert factory_calls == {"slow-bind": 1, "independent-bind": 1}
        assert not first.done()
        assert not duplicate.done()

        allow_first_factory.set()
        first_lease, duplicate_lease = await asyncio.wait_for(
            asyncio.gather(first, duplicate), timeout=2
        )
        assert first_lease.session is duplicate_lease.session
        assert factory_calls[slow.thread_id] == 1
        await asyncio.gather(
            first_lease.aclose(), duplicate_lease.aclose(), independent_lease.aclose()
        )
        assert await service.release_session(slow.thread_id) is True
        assert await service.release_session(independent.thread_id) is True
    finally:
        allow_first_factory.set()
        await service.aclose()
        for agent in agents:
            await agent.aclose()


@pytest.mark.asyncio
async def test_release_refuses_while_detached_child_task_uses_thread_resources(
    tmp_path,
):
    entered = Event()
    finish_child = Event()
    agent_dir = tmp_path / "background-state"

    @mf.tool_config(runtime_inputs=("handle",))
    def wait_for_release(handle):
        task = handle.get_task()
        entered.set()
        while not finish_child.wait(0.01):
            task.raise_if_interrupted()
        return "child finished"

    child_calls = 0

    def child_factory(_thread):
        nonlocal child_calls
        child_calls += 1

        def child_answer(**_kwargs):
            return (
                _tool_call("wait_for_release", "{}")
                if not finish_child.is_set()
                else _response("done")
            )

        child, _ = _agent("child", child_answer)
        child.generator.forward = Mock(side_effect=child_answer)
        child.tool_library.add(wait_for_release)
        return child

    root_calls = 0
    root_agent, _model = _agent(
        "root", lambda **_kwargs: _response(), agent_dir=agent_dir
    )
    child = child_factory(None)
    root_agent.tool_library.add(mf.tool_config(allow_background=True)(AgentTool()))
    root_agent.tool_library.add(child)

    def root_answer(**_kwargs):
        nonlocal root_calls
        root_calls += 1
        if root_calls == 1:
            return _tool_call(
                "agent",
                '{"name":"child","message":"background","run_in_background":true}',
            )
        return _response("foreground completed")

    root_agent.generator.aforward = AsyncMock(side_effect=root_answer)
    service = AgentService(store=SQLiteServiceStore())
    service.register("root", lambda _thread: AgentSession(root_agent))
    thread = await service.open_thread("root", thread_id="background-release")
    try:
        receipt = await service.prompt(thread.thread_id, "delegate", request_id="root")
        assert (
            await service.wait(thread.thread_id, receipt.request_id)
        ).status == "completed"
        assert await asyncio.to_thread(entered.wait, 3)
        assert not service._workers
        with pytest.raises(ServiceBusyError):
            await service.release_session(thread.thread_id)

        finish_child.set()
        for _ in range(100):
            resources = root_agent._owned_threads[thread.thread_id].resources
            tasks = resources.task_store.list()
            if tasks and all(
                item.status in {"completed", "failed", "interrupted"} for item in tasks
            ):
                break
            await asyncio.sleep(0.01)
        assert tasks and all(
            item.status in {"completed", "failed", "interrupted"} for item in tasks
        )
        assert await _release_when_idle(service, thread.thread_id) is True
    finally:
        finish_child.set()
        await service.aclose()
        await root_agent.aclose()
        await child.aclose()


@pytest.mark.asyncio
async def test_cleanup_failure_quarantines_binding_without_reloading_factory():
    factory_calls = 0
    cleanup_calls = 0
    agent, _model = _agent("cleanup-quarantine", lambda **_kwargs: _response())

    async def factory(_thread):
        nonlocal factory_calls
        factory_calls += 1

        async def fail_close():
            nonlocal cleanup_calls
            cleanup_calls += 1
            raise RuntimeError("close failed")

        return AgentSession(agent, on_close=fail_close)

    service = AgentService(store=SQLiteServiceStore())
    service.register("cleanup-quarantine", factory)
    thread = await service.open_thread(
        "cleanup-quarantine", thread_id="cleanup-quarantine"
    )
    try:
        lease = await service.acquire_session(thread.thread_id)
        session = lease.session
        await lease.aclose()
        with pytest.raises(ServiceRecoveryRequiredError) as close_error:
            await service.release_session(thread.thread_id)
        assert isinstance(close_error.value.__cause__, RuntimeError)
        assert str(close_error.value.__cause__) == "close failed"
        with pytest.raises(ServiceRecoveryRequiredError):
            await service.acquire_session(thread.thread_id)
        assert factory_calls == 1
        assert cleanup_calls == 1
    finally:
        with pytest.raises(ExceptionGroup):
            await service.aclose()
        await agent.aclose()


@pytest.mark.asyncio
async def test_acquired_scope_retains_namespace_and_thread_identity():
    agent, _model = _agent("scoped-agent", lambda **_kwargs: _response())
    service = AgentService(store=SQLiteServiceStore())
    service.register("scoped-agent", lambda _thread: AgentSession(agent))

    thread = await service.open_thread("scoped-agent", thread_id="scope-release")
    try:
        lease = await service.acquire_session(thread.thread_id)
        try:
            scope = lease.session.scope(thread.thread_id, run_id="run-check")
            assert scope.thread_id == thread.thread_id
            assert scope.namespace == "scoped-agent"
            assert scope.run_id == "run-check"
        finally:
            await lease.aclose()
    finally:
        await service.aclose()
        await agent.aclose()


@pytest.mark.asyncio
async def test_service_coding_facade_does_not_pin_agent_binding():
    made = []

    def factory(_thread):
        agent, _model = _agent("facade-agent", lambda **_kwargs: _response())
        made.append(agent)
        return AgentSession(agent)

    service = AgentService(store=SQLiteServiceStore())
    service.register("facade-agent", factory)
    thread = await service.open_thread("facade-agent", thread_id="facade-release")
    try:
        facade = await CodingSession.from_service(service, thread.thread_id)
        with pytest.raises(RuntimeError, match="acquire_session"):
            _ = facade.agent
        with pytest.raises(RuntimeError, match="acquire_session"):
            _ = facade.checkpoint_store
        with pytest.raises(RuntimeError, match="acquire_session"):
            facade.saved_state("missing")

        receipt = await facade.prompt("keep this thread durable", request_id="first")
        assert (await facade.wait(receipt.request_id)).status == "completed"
        first_agent = made[0]
        assert await service.release_session(thread.thread_id) is True

        next_receipt = await facade.prompt(
            "continue after release", request_id="second"
        )
        assert (await facade.wait(next_receipt.request_id)).status == "completed"
        assert made[1] is not first_agent
    finally:
        await service.aclose()
        for agent in made:
            await agent.aclose()


@pytest.mark.asyncio
async def test_release_reloads_persisted_workspace_policy_with_same_namespace(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    agents = []

    def factory(_thread):
        agent, _model = _agent("policy-release", lambda **_kwargs: _response())
        agent.workspace = workspace
        agents.append(agent)

        def scope_factory(scope):
            return scope.with_overrides(workspace=workspace)

        return AgentSession(agent, scope_factory=scope_factory)

    service = AgentService(store=SQLiteServiceStore())
    service.register("policy-release", factory)
    thread = await service.open_thread("policy-release", thread_id="policy-release")
    try:
        updated = await service.update_workspace_policy(
            thread.thread_id,
            permissions=PermissionSet({"filesystem.read", "filesystem.list"}),
            approval_policy="never",
        )
        assert await service.release_session(thread.thread_id) is True

        lease = await service.acquire_session(thread.thread_id)
        try:
            assert lease.session.namespace == "policy-release"
            restored = lease.session.workspace_policy_state.current
            assert restored.permissions == updated.permissions
            assert restored.revision == updated.revision
        finally:
            await lease.aclose()
    finally:
        await service.aclose()
        await workspace.aclose()
        for agent in agents:
            await agent.aclose()


@pytest.mark.asyncio
async def test_acquire_policy_refresh_failure_releases_session_pin(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    agent, _model = _agent("policy-refresh", lambda **_kwargs: _response())
    agent.workspace = workspace

    def scope_factory(scope):
        return scope.with_overrides(workspace=workspace)

    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "policy-refresh",
        lambda _thread: AgentSession(agent, scope_factory=scope_factory),
    )
    thread = await service.open_thread("policy-refresh", thread_id="policy-refresh")
    original_read = service.store.workspace_policy
    read_count = 0

    def fail_refresh(thread_id):
        nonlocal read_count
        read_count += 1
        if read_count == 2:
            raise RuntimeError("policy refresh failed")
        return original_read(thread_id)

    service.store.workspace_policy = fail_refresh
    try:
        with pytest.raises(RuntimeError, match="policy refresh failed"):
            await service.acquire_session(thread.thread_id)
        assert read_count == 2

        service.store.workspace_policy = original_read
        assert await service.release_session(thread.thread_id) is True
    finally:
        service.store.workspace_policy = original_read
        await service.aclose()
        await agent.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_approval_survives_release_http_snapshot_and_resume_without_repeat(
    tmp_path,
):
    pytest.importorskip("litestar")
    httpx = pytest.importorskip("httpx2")
    from litestar.testing import AsyncTestClient

    from msgflux.runtime import AgentApprovals, SQLiteApprovalStore
    from msgflux.data.stores import SQLiteCheckpointStore
    from msgflux.runtime.service.http.app import create_service_app

    approvals = SQLiteApprovalStore(str(tmp_path / "approvals.sqlite3"))
    checkpoints = SQLiteCheckpointStore(str(tmp_path / "checkpoints.sqlite3"))
    invoked = []
    agents = []

    def change(value: str):
        invoked.append(value)
        return value

    tool_call = _tool_call("change", '{"value":"approved"}')
    factory_responses = [tool_call, _response("done")]

    def factory(_thread):
        agent, _model = _agent(
            "release-approval",
            lambda **_kwargs: factory_responses.pop(0),
        )
        agent.checkpoint_store = checkpoints
        agent.approvals = AgentApprovals(approvals, {"change": "v1"}, "release-policy")
        agent.tool_library.add(change)
        agents.append(agent)
        return AgentSession(
            agent,
            checkpoint_store=checkpoints,
            approval_reviewer="host-reviewer",
        )

    journal = SQLiteServiceStore(tmp_path / "service.sqlite3")
    service = AgentService(store=journal)
    service.register("release-approval", factory)
    thread = await service.open_thread("release-approval", thread_id="release-approval")
    app = create_service_app(service, token="release-token")
    try:
        receipt = await service.prompt(
            thread.thread_id, "change", request_id="release-approval"
        )
        assert (
            await service.wait(thread.thread_id, receipt.request_id)
        ).status == "paused"
        assert invoked == []
        first_agent = agents[0]
        assert await service.release_session(thread.thread_id) is True

        async with AsyncTestClient(app=app) as client:
            response = await client.get(
                f"/v1/threads/{thread.thread_id}/snapshot",
                headers={"Authorization": "Bearer release-token"},
            )
        assert response.status_code == 200
        snapshot = response.json()
        assert snapshot["thread_id"] == thread.thread_id
        assert "change" in str(snapshot["messages"])
        assert len(agents) == 2
        assert agents[1] is not first_agent
        assert await service.release_session(thread.thread_id) is True

        reviews = await service.approval_reviews(thread.thread_id, receipt.run_id)
        assert len(reviews) == 1
        assert reviews[0].status == "pending"
        await service.decide_approval(
            thread.thread_id,
            receipt.run_id,
            reviews[0].request_id,
            approved=True,
            expected_revision=reviews[0].revision,
        )
        resumed = await service.resume(thread.thread_id, receipt.request_id)
        assert resumed.run_id == receipt.run_id
        assert (
            await service.wait(thread.thread_id, receipt.request_id)
        ).status == "completed"
        assert invoked == ["approved"]
        assert len(agents) == 3
        assert agents[-1] is not agents[1]
    finally:
        await service.aclose()
        for agent in agents:
            await agent.aclose()
        checkpoints.close()
        approvals.close()
        journal.close()
