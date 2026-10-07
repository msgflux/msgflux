"""Offline integration tests for the embedded Agent execution owner."""

import asyncio
import multiprocessing
import os
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_context,
    get_execution_scope,
)
from msgflux.runtime import AgentApprovals, InMemoryApprovalStore
from msgflux.runtime.service import AgentService, AgentSession
from msgflux.runtime.service.records import (
    ServiceThread,
    ServiceConflictError,
    ServiceBusyError,
    ServiceRecoveryRequiredError,
)
from msgflux.runtime.service.store import SQLiteServiceStore
from msgflux.runtime.workspace.api import AgentWorkspace


def _service_process_agent(checkpoints):
    agent = _agent("process-service")
    agent.checkpoint_store = checkpoints
    return agent


def _process_crash_after_admission(journal_path, checkpoint_path, thread_id):
    store = SQLiteServiceStore(journal_path)
    store.claim = lambda *_args, **_kwargs: os._exit(71)
    checkpoints = SQLiteCheckpointStore(checkpoint_path)
    service = AgentService(store=store)
    service.register(
        "main",
        lambda _thread_id: AgentSession(
            _service_process_agent(checkpoints), checkpoint_store=checkpoints
        ),
    )

    async def run():
        await service.prompt(thread_id, "hello", request_id="admitted")
        await asyncio.sleep(5)

    asyncio.run(run())
    os._exit(72)


def _process_crash_after_checkpoint(journal_path, checkpoint_path, thread_id):
    store = SQLiteServiceStore(journal_path)
    checkpoints = SQLiteCheckpointStore(checkpoint_path)
    service = AgentService(store=store)
    service.register(
        "main",
        lambda _thread_id: AgentSession(
            _service_process_agent(checkpoints), checkpoint_store=checkpoints
        ),
    )
    store.finish = lambda *_args, **_kwargs: os._exit(73)

    async def run():
        await service.prompt(thread_id, "hello", request_id="finished-checkpoint")
        await service.wait(thread_id, "finished-checkpoint")

    asyncio.run(run())
    os._exit(74)


def _response(content="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    response.reasoning = None
    return response


def _agent(name="service-test"):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name=name, model=model)
    agent.generator.aforward = AsyncMock(return_value=_response())
    return agent


def _service(factory, *, store=None):
    service = AgentService(store=store or SQLiteServiceStore())
    service.register("main", factory)
    return service


@pytest.mark.asyncio
async def test_duplicate_request_returns_same_run_and_conflicting_prompt_fails():
    agent = _agent()
    service = _service(lambda _thread_id: AgentSession(agent))
    thread = await service.open_thread("main", thread_id="dedupe")
    try:
        first = await service.prompt(thread.thread_id, "hello", request_id="req-1")
        settled = await service.wait(thread.thread_id, "req-1")
        duplicate = await service.prompt(thread.thread_id, "hello", request_id="req-1")

        assert duplicate.run_id == first.run_id == settled.run_id
        assert duplicate.status == "completed"
        assert agent.generator.aforward.await_count == 1
        with pytest.raises(ServiceConflictError):
            await service.prompt(thread.thread_id, "different", request_id="req-1")
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_runs_projects_checkpoint_metadata_and_supports_legacy_checkpoints():
    checkpoints = InMemoryCheckpointStore()
    agent = _agent("run-summary")
    agent.checkpoint_store = checkpoints
    service = _service(lambda _thread_id: AgentSession(agent))
    thread = await service.open_thread("main", thread_id="run-summaries")
    namespace = agent.get_module_name()
    # Persisted checkpoints can predate service admissions. Their private state
    # and config must never appear in the public summary.
    checkpoints.save_state(
        namespace, thread.thread_id, "older", {"status": "completed", "secret": "x"}
    )
    checkpoints.save_state(
        namespace,
        thread.thread_id,
        "newer",
        {"status": "interrupted", "config": {"token": "secret"}},
    )
    try:
        runs = await service.runs(thread.thread_id)
        assert [run.run_id for run in runs] == ["newer", "older"]
        assert [run.status for run in runs] == ["interrupted", "completed"]
        assert all(
            run.updated_at is None or isinstance(run.updated_at, float) for run in runs
        )
        checkpoints.list_runs = lambda *_args, **_kwargs: [
            {"run_id": "integer-time", "status": "completed", "updated_at": 42}
        ]
        converted = await service.runs(thread.thread_id)
        assert converted[0].updated_at == 42.0
        assert isinstance(converted[0].updated_at, float)
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_runs_without_checkpoint_store_are_empty_and_unknown_thread_errors():
    agent = _agent("no-checkpoints")
    service = _service(lambda _thread_id: AgentSession(agent))
    thread = await service.open_thread("main", thread_id="no-checkpoints-thread")
    try:
        assert await service.runs(thread.thread_id) == ()
        with pytest.raises(KeyError):
            await service.runs("unknown-thread")
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_observers_and_cancelled_waiter_do_not_own_the_run():
    entered = asyncio.Event()
    release = asyncio.Event()
    agent = _agent()

    async def delayed(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return _response("survived")

    agent.generator.aforward = AsyncMock(side_effect=delayed)
    service = _service(lambda _thread_id: AgentSession(agent))
    thread = await service.open_thread("main", thread_id="observers")
    try:
        async with service.watch(thread.thread_id) as first_watcher:
            async with service.watch(thread.thread_id) as second_watcher:
                receipt = await service.prompt(
                    thread.thread_id, "hello", request_id="req-1"
                )
                waiter = asyncio.create_task(service.wait(thread.thread_id, "req-1"))
                await asyncio.wait_for(entered.wait(), timeout=2)
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
                assert first_watcher.snapshot.thread_id == thread.thread_id
                assert second_watcher.snapshot.thread_id == thread.thread_id
            # Closing one observer while the other is still attached is harmless.
        assert (service.receipt(thread.thread_id, "req-1")).status == "running"
        release.set()
        settled = await service.wait(thread.thread_id, "req-1")
        assert settled.run_id == receipt.run_id
        assert settled.status == "completed"
        assert agent.generator.aforward.await_count == 1
    finally:
        release.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_independent_threads_run_in_parallel_with_distinct_agents_and_scopes(
    tmp_path,
):
    entered = {"one": asyncio.Event(), "two": asyncio.Event()}
    release = asyncio.Event()
    made = {}
    observed_principals = {}
    observed_workspaces = {}
    workspaces = {}

    def factory(thread):
        thread_id = thread.thread_id
        agent = _agent()
        (tmp_path / thread_id).mkdir()
        workspace = AgentWorkspace.local(tmp_path / thread_id)
        workspaces[thread_id] = workspace

        async def delayed(*_args, **_kwargs):
            scope = get_execution_scope()
            observed_principals[thread_id] = scope.principal
            observed_workspaces[thread_id] = scope.workspace
            entered[thread_id].set()
            await release.wait()
            return _response(thread_id)

        agent.generator.aforward = AsyncMock(side_effect=delayed)
        made[thread_id] = agent

        def scope_factory(scope):
            return scope.with_overrides(principal=thread_id, workspace=workspace)

        return AgentSession(
            agent, scope_factory=scope_factory, on_close=workspaces[thread_id].aclose
        )

    service = _service(factory)
    one = await service.open_thread("main", thread_id="one")
    two = await service.open_thread("main", thread_id="two")
    try:
        first = await service.prompt(one.thread_id, "a", request_id="a")
        second = await service.prompt(two.thread_id, "b", request_id="b")
        await asyncio.wait_for(
            asyncio.gather(entered["one"].wait(), entered["two"].wait()), timeout=2
        )
        assert made["one"] is not made["two"]
        assert observed_principals == {"one": "one", "two": "two"}
        assert observed_workspaces["one"] is workspaces["one"]
        assert observed_workspaces["two"] is workspaces["two"]
        assert observed_workspaces["one"] is not observed_workspaces["two"]
        release.set()
        assert (await service.wait(one.thread_id, "a")).run_id == first.run_id
        assert (await service.wait(two.thread_id, "b")).run_id == second.run_id
    finally:
        release.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_thread_cwd_is_canonical_immutable_and_reused(tmp_path):
    first_root = tmp_path / "workspace"
    first_root.mkdir()
    alias = tmp_path / "workspace-alias"
    alias.symlink_to(first_root, target_is_directory=True)
    other_root = tmp_path / "other"
    other_root.mkdir()
    service = _service(lambda _thread: AgentSession(_agent()))
    try:
        thread = await service.open_thread("main", thread_id="cwd", cwd=alias)
        assert thread.cwd == str(first_root.resolve())
        assert await service.open_thread("main", thread_id="cwd") == thread
        assert (
            await service.open_thread("main", thread_id="cwd", cwd=first_root) == thread
        )
        with pytest.raises(ServiceConflictError):
            await service.open_thread("main", thread_id="cwd", cwd=other_root)
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_invalid_thread_cwd_fails_before_journal_write_or_factory(tmp_path):
    made = []
    service = _service(lambda _thread: made.append(True) or AgentSession(_agent()))
    file_path = tmp_path / "file"
    file_path.write_text("x")
    try:
        for invalid in ("relative", tmp_path / "missing", file_path):
            with pytest.raises(ValueError):
                await service.open_thread("main", thread_id="invalid-cwd", cwd=invalid)
        with pytest.raises(KeyError):
            service.store.thread("invalid-cwd")
        assert made == []
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_factory_receives_persisted_service_thread_after_restart(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    store = SQLiteServiceStore(tmp_path / "service.sqlite3")
    first_bindings = []
    first_service = _service(
        lambda binding: first_bindings.append(binding) or AgentSession(_agent()),
        store=store,
    )
    thread = await first_service.open_thread("main", thread_id="restart", cwd=root)
    await first_service.session(thread.thread_id)
    assert first_bindings == [thread]
    await first_service.aclose()

    second_bindings = []

    async def async_factory(binding):
        second_bindings.append(binding)
        return AgentSession(_agent())

    reopened = _service(async_factory, store=store)
    try:
        await reopened.session("restart")
        assert second_bindings == [thread]
    finally:
        await reopened.aclose()


@pytest.mark.asyncio
async def test_model_failure_is_settled_as_failed():
    agent = _agent()
    agent.generator.aforward = AsyncMock(side_effect=RuntimeError("offline model"))
    service = _service(lambda _thread_id: AgentSession(agent))
    thread = await service.open_thread("main", thread_id="failure")
    try:
        await service.prompt(thread.thread_id, "hello", request_id="req-1")
        receipt = await service.wait(thread.thread_id, "req-1")
        assert receipt.status == "failed"
        assert "offline model" in receipt.error
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_paused_approval_keeps_thread_busy_then_resumes_same_run():
    checkpoints = InMemoryCheckpointStore()
    approval_store = InMemoryApprovalStore()
    calls = []

    def lookup(query: str) -> str:
        calls.append(query)
        return "found"

    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="approval-service",
        model=model,
        tools=[lookup],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(approval_store, {"lookup": "v1"}, "policy"),
    )
    tool_calls = ToolCallAggregator()
    tool_calls.process(0, "call_1", "lookup", '{"query":"private"}')
    tool_response = ModelResponse()
    tool_response.set_response_type("tool_call")
    tool_response.add(tool_calls)
    agent.generator.aforward = AsyncMock(side_effect=[tool_response, _response("done")])
    service = _service(
        lambda _thread_id: AgentSession(
            agent,
            checkpoint_store=checkpoints,
            scope_factory=lambda scope: scope.with_overrides(principal="host"),
        )
    )
    thread = await service.open_thread("main", thread_id="approval")
    try:
        receipt = await service.prompt(thread.thread_id, "lookup", request_id="req-1")
        paused = await service.wait(thread.thread_id, "req-1")
        assert paused.status == "paused", paused.error
        assert paused.run_id == receipt.run_id
        assert calls == []
        assert agent.generator.aforward.await_count == 1
        with pytest.raises(ServiceRecoveryRequiredError):
            await service.prompt(thread.thread_id, "another", request_id="req-2")

        approval = approval_store.pending(
            "approval-service", thread.thread_id, receipt.run_id
        )[0]
        await agent.adecide_approval(
            approval.request_id, approved=True, decided_by="host"
        )
        resumed = await service.resume(thread.thread_id, "req-1")
        assert resumed.run_id == receipt.run_id
        settled = await service.wait(thread.thread_id, "req-1")
        assert settled.status == "completed"
        assert agent.generator.aforward.await_count == 2
        assert calls == ["private"]
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_steered_message_reaches_model_after_tool_round():
    tool_started = asyncio.Event()
    finish_tool = asyncio.Event()
    observed_model_inputs = []

    async def wait_for_release() -> str:
        tool_started.set()
        await finish_tool.wait()
        return "tool finished"

    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="steer-service", model=model, tools=[wait_for_release])
    tool_calls = ToolCallAggregator()
    tool_calls.process(0, "call_1", "wait_for_release", "{}")
    tool_response = ModelResponse()
    tool_response.set_response_type("tool_call")
    tool_response.add(tool_calls)

    async def generate(*args, **kwargs):
        observed_model_inputs.append((args, kwargs))
        if len(observed_model_inputs) == 1:
            return tool_response
        return _response("done")

    agent.generator.aforward = AsyncMock(side_effect=generate)
    service = _service(lambda _thread_id: AgentSession(agent))
    thread = await service.open_thread("main", thread_id="steering")
    try:
        receipt = await service.prompt(thread.thread_id, "start", request_id="req-1")
        await asyncio.wait_for(tool_started.wait(), timeout=2)
        notification = await service.steer(
            thread.thread_id, receipt.run_id, "Focus on the new constraint"
        )
        assert notification.source == "incoming_user_message"
        finish_tool.set()
        settled = await service.wait(thread.thread_id, "req-1")
        assert settled.status == "completed"
        assert len(observed_model_inputs) == 2
        assert "Focus on the new constraint" in str(observed_model_inputs[1])
    finally:
        finish_tool.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_prompt_does_not_inherit_foreign_contextvars_or_run_lineage():
    host_checkpoints = InMemoryCheckpointStore()
    host_task_store = object()
    agent = _agent("context-isolation")
    observed = {}

    async def inspect_context(*_args, **_kwargs):
        observed.update(get_execution_context())
        return _response()

    agent.generator.aforward = AsyncMock(side_effect=inspect_context)
    service = _service(
        lambda _thread_id: AgentSession(
            agent,
            checkpoint_store=host_checkpoints,
            task_store=host_task_store,
            scope_factory=lambda scope: scope.with_overrides(principal="trusted-host"),
        )
    )
    thread = await service.open_thread("main", thread_id="owned-thread")
    foreign_task = object()
    foreign_inbox = object()
    foreign_checkpoint = object()
    foreign_task_store = object()
    foreign_scope = ExecutionScope(
        thread_id="foreign-thread",
        namespace="foreign-agent",
        run_id="foreign-run",
        parent_run_id="foreign-parent",
        root_run_id="foreign-root",
        principal="foreign-principal",
    )
    try:
        with execution_context(
            scope=foreign_scope,
            checkpoint_store=foreign_checkpoint,
            task_store=foreign_task_store,
            agent_inbox=foreign_inbox,
            task_handle=foreign_task,
        ):
            receipt = await service.prompt(
                thread.thread_id, "hello", request_id="isolated"
            )
            settled = await service.wait(thread.thread_id, "isolated")
        assert settled.status == "completed"
        assert observed["scope"].thread_id == thread.thread_id
        assert observed["scope"].run_id == receipt.run_id
        assert observed["scope"].namespace == "context-isolation"
        assert observed["scope"].principal == "trusted-host"
        assert observed["task_handle"] is None
        assert observed["task_store"] is host_task_store
        assert observed["checkpoint_store"] is host_checkpoints
        assert observed["agent_inbox"] is not foreign_inbox
        assert observed["scope"].parent_run_id is None
        assert observed["scope"].root_run_id == receipt.run_id
    finally:
        await service.aclose()


def test_agent_session_rejects_conflicting_constructor_checkpoint_store():
    configured_store = InMemoryCheckpointStore()
    supplied_store = InMemoryCheckpointStore()
    agent = Agent(
        name="checkpoint-conflict",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=configured_store,
    )
    with pytest.raises(ValueError, match="checkpoint_store"):
        AgentSession(agent, checkpoint_store=supplied_store)


def test_agent_session_scope_factory_cannot_replace_service_run_id():
    agent = _agent("scope-run-validation")
    session = AgentSession(
        agent,
        scope_factory=lambda scope: scope.with_overrides(run_id="foreign-run"),
    )
    with pytest.raises(ValueError, match="service-owned run identity"):
        session.scope("owned-thread", run_id="service-run")


@pytest.mark.asyncio
async def test_interrupt_is_explicit_and_targets_only_the_selected_run():
    entered = asyncio.Event()
    release = asyncio.Event()
    agent = _agent()

    async def delayed(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return _response()

    agent.generator.aforward = AsyncMock(side_effect=delayed)
    service = _service(lambda _thread_id: AgentSession(agent))
    thread = await service.open_thread("main", thread_id="interrupt")
    try:
        receipt = await service.prompt(thread.thread_id, "hello", request_id="req-1")
        await asyncio.wait_for(entered.wait(), timeout=2)
        assert await service.interrupt(thread.thread_id, "unrelated-run") is False
        assert await service.interrupt(thread.thread_id, receipt.run_id) is True
        release.set()
        settled = await service.wait(thread.thread_id, "req-1")
        assert settled.status == "interrupted"
        assert await service.interrupt(thread.thread_id, receipt.run_id) is False
    finally:
        release.set()
        await service.aclose()


@pytest.mark.asyncio
async def test_unknown_started_work_requires_host_recovery_confirmation():
    store = SQLiteServiceStore()
    first_service = _service(lambda _thread_id: AgentSession(_agent()), store=store)
    thread = await first_service.open_thread("main", thread_id="recovery")
    # Simulate a process that accepted and claimed work, then disappeared.
    record = store.admit(thread.thread_id, "req-1", "hello", "service-test")
    store.claim(record, "old-owner")
    await first_service.aclose()

    reopened_service = _service(lambda _thread_id: AgentSession(_agent()), store=store)
    try:
        with pytest.raises(ServiceRecoveryRequiredError):
            await reopened_service.resume(thread.thread_id, "req-1")
        with pytest.raises(ServiceRecoveryRequiredError):
            await reopened_service.resume(
                thread.thread_id, "req-1", worker_stopped=True
            )
    finally:
        await reopened_service.aclose()
        store.close()


@pytest.mark.asyncio
async def test_reopen_resume_of_terminal_receipt_does_not_call_model_again():
    journal = SQLiteServiceStore()
    checkpoints = InMemoryCheckpointStore()
    agent = _agent()
    first_service = _service(
        lambda _thread_id: AgentSession(agent, checkpoint_store=checkpoints),
        store=journal,
    )
    thread = await first_service.open_thread("main", thread_id="terminal")
    try:
        accepted = await first_service.prompt(
            thread.thread_id, "hello", request_id="req-1"
        )
        terminal = await first_service.wait(thread.thread_id, "req-1")
        assert terminal.status == "completed"
        assert (
            checkpoints.load_state("service-test", thread.thread_id, accepted.run_id)[
                "status"
            ]
            == "completed"
        )
    finally:
        await first_service.aclose()

    restarted_agent = _agent()
    restarted = _service(
        lambda _thread_id: AgentSession(restarted_agent, checkpoint_store=checkpoints),
        store=journal,
    )
    try:
        recovered = await restarted.resume(thread.thread_id, "req-1")
        assert recovered.status == "completed"
        assert recovered.run_id == accepted.run_id
        assert restarted_agent.generator.aforward.await_count == 0
    finally:
        await restarted.aclose()
        journal.close()


@pytest.mark.asyncio
async def test_rejected_scope_factory_is_closed_and_explicit_agent_reuse_conflicts():
    closes = []
    agent = _agent()
    store = SQLiteServiceStore()
    service = _service(
        lambda _thread_id: AgentSession(
            agent,
            scope_factory=lambda scope: scope.with_overrides(thread_id="wrong"),
            on_close=lambda: closes.append("closed"),
        ),
        store=store,
    )
    first = await service.open_thread("main", thread_id="first")
    second = await service.open_thread("main", thread_id="second")
    try:
        with pytest.raises(ValueError, match="preserve thread_id"):
            await service.prompt(first.thread_id, "hello", request_id="one")
        assert closes == ["closed"]
        # The first invalid factory result was discarded; the second request is
        # also rejected before the same Agent can be shared between threads.
        with pytest.raises(ValueError, match="preserve thread_id"):
            await service.prompt(second.thread_id, "hello", request_id="two")
    finally:
        await service.aclose()
        assert store.thread(first.thread_id).agent_id == "main"

    reuse_service = _service(lambda _thread_id: AgentSession(agent), store=store)
    try:
        await reuse_service.prompt(first.thread_id, "hello", request_id="reuse-1")
        await reuse_service.wait(first.thread_id, "reuse-1")
        with pytest.raises(ServiceConflictError, match="isolate Agents"):
            await reuse_service.prompt(second.thread_id, "hello", request_id="reuse-2")
    finally:
        await reuse_service.aclose()
        store.close()


@pytest.mark.asyncio
async def test_cancelled_shutdown_waiter_still_closes_once_and_borrows_store():
    entered = asyncio.Event()
    release = asyncio.Event()
    closes = []
    agent = _agent()

    async def delayed(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return _response()

    agent.generator.aforward = AsyncMock(side_effect=delayed)
    store = SQLiteServiceStore()
    service = _service(
        lambda _thread_id: AgentSession(
            agent, on_close=lambda: closes.append("closed")
        ),
        store=store,
    )
    thread = await service.open_thread("main", thread_id="shutdown")
    await service.prompt(thread.thread_id, "hello", request_id="req-1")
    await asyncio.wait_for(entered.wait(), timeout=2)
    shutdown_waiter = asyncio.create_task(service.aclose())
    await asyncio.sleep(0)
    shutdown_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown_waiter
    release.set()
    await service.aclose()
    assert closes == ["closed"]
    assert store.thread(thread.thread_id) == thread
    store.close()


def _run_spawned_crash(target, *args, exitcode):
    process = multiprocessing.get_context("spawn").Process(target=target, args=args)
    try:
        process.start()
        process.join(timeout=20)
        assert process.exitcode == exitcode
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        process.close()


def test_process_crash_after_admission_redelivers_same_pending_request(tmp_path):
    journal_path = str(tmp_path / "service.sqlite3")
    checkpoint_path = str(tmp_path / "checkpoints.sqlite3")
    initial_store = SQLiteServiceStore(journal_path)
    thread = ServiceThread("pending-process", "main")
    initial_store.bind_thread(thread)
    initial_store.close()

    _run_spawned_crash(
        _process_crash_after_admission,
        journal_path,
        checkpoint_path,
        thread.thread_id,
        exitcode=71,
    )

    journal = SQLiteServiceStore(journal_path)
    checkpoints = SQLiteCheckpointStore(checkpoint_path)
    receipt = journal.get(thread.thread_id, "admitted").receipt
    assert receipt.status == "accepted"
    agent = _service_process_agent(checkpoints)
    service = _service(
        lambda _thread_id: AgentSession(agent, checkpoint_store=checkpoints),
        store=journal,
    )

    async def recover():
        duplicate = await service.prompt(
            thread.thread_id, "hello", request_id="admitted"
        )
        assert duplicate.run_id == receipt.run_id
        settled = await service.wait(thread.thread_id, "admitted")
        assert settled.status == "completed"
        assert settled.run_id == receipt.run_id
        assert agent.generator.aforward.await_count == 1

    try:
        asyncio.run(recover())
    finally:
        asyncio.run(service.aclose())
        checkpoints.close()
        journal.close()


def test_process_crash_after_completed_checkpoint_reconciles_without_model_retry(
    tmp_path,
):
    journal_path = str(tmp_path / "service.sqlite3")
    checkpoint_path = str(tmp_path / "checkpoints.sqlite3")
    initial_store = SQLiteServiceStore(journal_path)
    thread = ServiceThread("checkpoint-process", "main")
    initial_store.bind_thread(thread)
    initial_store.close()

    _run_spawned_crash(
        _process_crash_after_checkpoint,
        journal_path,
        checkpoint_path,
        thread.thread_id,
        exitcode=73,
    )

    journal = SQLiteServiceStore(journal_path)
    checkpoints = SQLiteCheckpointStore(checkpoint_path)
    record = journal.get(thread.thread_id, "finished-checkpoint")
    assert record.receipt.status == "running"
    assert (
        checkpoints.load_state(
            "process-service", thread.thread_id, record.receipt.run_id
        )["status"]
        == "completed"
    )
    agent = _service_process_agent(checkpoints)
    service = _service(
        lambda _thread_id: AgentSession(agent, checkpoint_store=checkpoints),
        store=journal,
    )

    async def recover():
        settled = await service.resume(
            thread.thread_id, "finished-checkpoint", worker_stopped=True
        )
        assert settled.status == "completed"
        assert settled.run_id == record.receipt.run_id
        assert agent.generator.aforward.await_count == 0

    try:
        asyncio.run(recover())
    finally:
        asyncio.run(service.aclose())
        checkpoints.close()
        journal.close()


@pytest.mark.asyncio
async def test_checkpoint_import_requires_quiescence_and_reconciles_terminal_run():
    checkpoints = InMemoryCheckpointStore()
    agent = _agent("pre-service")
    agent.checkpoint_store = checkpoints
    scope = ExecutionScope(thread_id="old-thread", namespace="pre-service")
    with execution_context(scope=scope, checkpoint_store=checkpoints):
        events = [event async for event in agent.stream_events("original", scope=scope)]
    run_id = next(event.run_id for event in events if event.type == "run.start")
    service = _service(lambda _thread_id: AgentSession(agent))
    await service.open_thread("main", thread_id=scope.thread_id)
    try:
        assert (await service.session(scope.thread_id)).agent is agent
        with pytest.raises(ServiceRecoveryRequiredError, match="quiescence"):
            await service.resume_checkpoint(scope.thread_id, run_id)
        assert service.store.get_for_run(scope.thread_id, run_id) is None
        receipt = await service.resume_checkpoint(
            scope.thread_id, run_id, worker_stopped=True
        )
        assert receipt.status == "completed"
        assert receipt.run_id == run_id
        assert service.receipt_for_run(scope.thread_id, run_id) == receipt
        assert agent.generator.aforward.await_count == 1
        assert await service.resume_checkpoint(scope.thread_id, run_id) == receipt
        with pytest.raises(ServiceRecoveryRequiredError, match="no checkpoint"):
            await service.resume_checkpoint(
                scope.thread_id, "missing", worker_stopped=True
            )
        assert service.store.get_for_run(scope.thread_id, "missing") is None
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_checkpoint_import_resumes_approved_tool_without_resending_input():
    checkpoints = InMemoryCheckpointStore()
    approvals = InMemoryApprovalStore()
    calls = []

    def lookup(query: str) -> str:
        calls.append(query)
        return "found"

    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="old-approval",
        model=model,
        tools=[lookup],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(approvals, {"lookup": "v1"}, "policy"),
    )
    tool_calls = ToolCallAggregator()
    tool_calls.process(0, "old-call", "lookup", '{"query":"saved"}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(tool_calls)
    agent.generator.aforward = AsyncMock(side_effect=[response, _response()])
    scope = ExecutionScope(
        thread_id="old-approval-thread", namespace="old-approval", principal="host"
    )
    from msgflux.exceptions import TaskPauseRequestedError

    with execution_context(scope=scope, checkpoint_store=checkpoints):
        with pytest.raises(TaskPauseRequestedError):
            _ = [event async for event in agent.stream_events("lookup", scope=scope)]
    state = checkpoints.load_latest_run("old-approval", scope.thread_id)
    run_id = state["scope"]["run_id"]
    approval = approvals.pending("old-approval", scope.thread_id, run_id)[0]
    await agent.adecide_approval(approval.request_id, approved=True, decided_by="host")
    service = _service(
        lambda _thread_id: AgentSession(
            agent, scope_factory=lambda base: base.with_overrides(principal="host")
        )
    )
    await service.open_thread("main", thread_id=scope.thread_id)
    try:
        receipt = await service.resume_checkpoint(
            scope.thread_id, run_id, worker_stopped=True
        )
        assert receipt.run_id == run_id
        settled = await service.wait(scope.thread_id, receipt.request_id)
        assert settled.status == "completed", settled.error
        assert calls == ["saved"]
        assert agent.generator.aforward.await_count == 2
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_configured_agent_workspace_is_inherited_before_execution_context(
    tmp_path,
):
    workspace = AgentWorkspace.local(tmp_path)
    agent = _agent("configured-workspace")
    agent.workspace = workspace
    observed = []

    async def answer(**_kwargs):
        scope = get_execution_scope()
        observed.append(scope)
        assert scope.workspace is workspace
        assert not scope.permissions.missing(("filesystem.read",))
        return _response("workspace ready")

    agent.generator.aforward = AsyncMock(side_effect=answer)
    service = _service(lambda _thread: AgentSession(agent))
    thread = await service.open_thread("main")
    try:
        assert (await service.session(thread.thread_id)).scope(
            thread.thread_id
        ).workspace is workspace
        receipt = await service.prompt(
            thread.thread_id, "hello", request_id="workspace"
        )
        settled = await service.wait(thread.thread_id, receipt.request_id)
        assert settled.status == "completed", settled.error
        assert len(observed) == 1
    finally:
        await service.aclose()
        service.store.close()
        await workspace.aclose()
