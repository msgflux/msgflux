"""Spawned-process recovery checks over durable Agent runtime stores."""

import asyncio
import multiprocessing
import os
import time

import pytest
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore
from msgflux.runtime.context import ExecutionScope, execution_context
from msgflux.runtime.recovery import AgentTaskRecovery
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.runtime.workspace.local import LocalWorkspaceBackend
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.tasks import SQLiteTaskStore


class _FixedModel:
    model_type = "chat_completion"

    def __init__(self, gate=None):
        self.gate = gate
        self.calls = 0

    def __call__(self, **_kwargs):
        self.calls += 1
        if self.gate is not None and not self.gate.wait(timeout=15):
            raise TimeoutError("Parent did not release the model")
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("durable result")
        return response


def _workspace(paths):
    registry = SQLiteWorkspaceRegistry(paths[3])
    backend = LocalWorkspaceBackend(paths[4], registry=registry)
    try:
        record = registry.get_record("project")
    except FileNotFoundError:
        record = None
    if record is None:
        workspace = asyncio.run(AgentWorkspace.open(backend, "project"))
    else:
        workspace = asyncio.run(
            AgentWorkspace.reconnect(backend, "project", record.identity)
        )
    return registry, backend, workspace


def _crash_after_terminal_checkpoint(paths, connection, gate):
    checkpoints = SQLiteCheckpointStore(paths[0])
    tasks = SQLiteTaskStore(paths[1])
    inbox_store = SQLiteAgentInboxStore(paths[2])
    _registry, _backend, workspace = _workspace(paths)

    def exit_after_commit(_context):
        os._exit(73)

    worker = Agent(
        name="worker",
        model=_FixedModel(gate),
        checkpoint_store=checkpoints,
        workspace=workspace,
        hooks=[Hook(event="after_run_end", handler=exit_after_commit)],
    )
    worker.tool_config = {"background": True}
    library = ToolLibrary(name="lib", tools=[worker], task_store=tasks)
    library.set_agent_inbox(AgentInbox(owner="root", store=inbox_store))
    library.get_background_dispatcher().lease_seconds = 0.2

    with execution_context(
        scope=ExecutionScope(
            thread_id="thread", run_id="root", root_run_id="root", workspace=workspace
        )
    ):
        dispatch = library([("start", "worker", {"task": "Start"})])
        task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]
        connection.send(task_id)
        connection.close()
    gate.wait(timeout=15)


def _crash_before_worker_claim(paths, connection, gate):
    from msgflux.tasks.handle import TaskHandle

    checkpoints = SQLiteCheckpointStore(paths[0])
    tasks = SQLiteTaskStore(paths[1])
    inbox_store = SQLiteAgentInboxStore(paths[2])
    _registry, _backend, workspace = _workspace(paths)
    worker = Agent(
        name="worker",
        model=_FixedModel(),
        checkpoint_store=checkpoints,
        workspace=workspace,
    )
    worker.tool_config = {"background": True}
    library = ToolLibrary(name="lib", tools=[worker], task_store=tasks)
    library.set_agent_inbox(AgentInbox(owner="root", store=inbox_store))

    def exit_before_claim(self, **_kwargs):
        if not gate.wait(timeout=15):
            raise TimeoutError("Parent did not release the pre-claim crash gate")
        os._exit(74)

    TaskHandle.start_worker = exit_before_claim
    with execution_context(
        scope=ExecutionScope(
            thread_id="thread", run_id="root", root_run_id="root", workspace=workspace
        )
    ):
        dispatch = library([("start", "worker", {"task": "Start"})])
        task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]
        connection.send(task_id)
        connection.close()
    gate.wait(timeout=15)


def _inspect_in_fresh_process(paths, task_id, output):
    checkpoints = SQLiteCheckpointStore(paths[0])
    tasks = SQLiteTaskStore(paths[1])
    inbox_store = SQLiteAgentInboxStore(paths[2])
    registry, _backend, workspace = _workspace(paths)
    worker = Agent(name="worker", model=_FixedModel(), workspace=workspace)
    worker.tool_config = {"background": True}
    library = ToolLibrary(name="lib", tools=[worker], task_store=tasks)
    library.set_agent_inbox(AgentInbox(owner="root", store=inbox_store))
    with execution_context(
        scope=ExecutionScope(workspace=workspace), checkpoint_store=checkpoints
    ):
        report = AgentTaskRecovery(library).inspect(task_id)
    output.put(
        (report.classification, report.checkpoint_status, report.workspace_status)
    )
    asyncio.run(workspace.aclose())
    registry.close()
    checkpoints.close()
    tasks.close()
    inbox_store.close()


def _recover_in_fresh_process(paths, task_id, ready, start, output):
    checkpoints = SQLiteCheckpointStore(paths[0])
    tasks = SQLiteTaskStore(paths[1])
    inbox_store = SQLiteAgentInboxStore(paths[2])
    registry, _backend, workspace = _workspace(paths)
    model = _FixedModel()
    worker = Agent(name="worker", model=model, workspace=workspace)
    worker.tool_config = {"background": True}
    library = ToolLibrary(name="lib", tools=[worker], task_store=tasks)
    library.set_agent_inbox(AgentInbox(owner="root", store=inbox_store))
    library.get_background_dispatcher().lease_seconds = 2
    with execution_context(
        scope=ExecutionScope(workspace=workspace), checkpoint_store=checkpoints
    ):
        recovery = AgentTaskRecovery(library)
        ready.set()
        if not start.wait(timeout=15):
            output.put(("timeout", model.calls))
            return
        try:
            result = recovery.recover(task_id, "Continue", worker_stopped=True)
            deadline = time.monotonic() + 15
            while tasks.get(task_id).status in {"queued", "running"}:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Recovered task did not reach a terminal state")
                time.sleep(0.01)
            outcome = ("ok", result, model.calls)
        except RuntimeError as exc:
            outcome = ("observed", str(exc), model.calls)
        output.put(outcome)
    asyncio.run(workspace.aclose())
    registry.close()
    checkpoints.close()
    tasks.close()
    inbox_store.close()


def test_fresh_process_inspects_terminal_agent_checkpoint_without_model_call(tmp_path):
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "workspace"
    root.mkdir()
    paths = tuple(
        str(path)
        for path in (
            tmp_path / "checkpoint.sqlite",
            tmp_path / "tasks.sqlite",
            tmp_path / "inbox.sqlite",
            tmp_path / "registry.sqlite",
            root,
        )
    )
    receive, send = context.Pipe(duplex=False)
    gate = context.Event()
    process = context.Process(
        target=_crash_after_terminal_checkpoint, args=(paths, send, gate)
    )
    inspector = None
    try:
        process.start()
        assert receive.poll(20)
        task_id = receive.recv()
        gate.set()
        process.join(timeout=20)
        assert process.exitcode == 73

        checkpoints = SQLiteCheckpointStore(paths[0])
        tasks = SQLiteTaskStore(paths[1])
        try:
            checkpoint = checkpoints.load_state("worker", "thread", task_id)
            assert checkpoint["status"] == "completed"
            assert checkpoint["task_result"] == {"value": "durable result"}
            assert tasks.get(task_id).status == "running"
            deadline = time.monotonic() + 5
            while tasks.get_worker_lease(task_id).expires_at > time.time():
                assert time.monotonic() < deadline
                time.sleep(0.01)
        finally:
            checkpoints.close()
            tasks.close()

        output = context.Queue()
        inspector = context.Process(
            target=_inspect_in_fresh_process, args=(paths, task_id, output)
        )
        inspector.start()
        assert output.get(timeout=20) == ("terminal_result", "completed", "compatible")
        inspector.join(timeout=20)
        assert inspector.exitcode == 0
    finally:
        receive.close()
        send.close()
        for child in (process, inspector):
            if child is not None:
                if child.is_alive():
                    child.terminate()
                child.join(timeout=5)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=5)
                child.close()


def test_two_fresh_recovery_processes_reconcile_one_terminal_result(tmp_path):
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "workspace"
    root.mkdir()
    paths = tuple(
        str(path)
        for path in (
            tmp_path / "checkpoint.sqlite",
            tmp_path / "tasks.sqlite",
            tmp_path / "inbox.sqlite",
            tmp_path / "registry.sqlite",
            root,
        )
    )
    receive, send = context.Pipe(duplex=False)
    gate = context.Event()
    owner = context.Process(
        target=_crash_after_terminal_checkpoint, args=(paths, send, gate)
    )
    contenders = []
    output = context.Queue()
    try:
        owner.start()
        assert receive.poll(20)
        task_id = receive.recv()
        gate.set()
        owner.join(timeout=20)
        assert owner.exitcode == 73

        checkpoints = SQLiteCheckpointStore(paths[0])
        tasks = SQLiteTaskStore(paths[1])
        try:
            deadline = time.monotonic() + 5
            while tasks.get_worker_lease(task_id).expires_at > time.time():
                assert time.monotonic() < deadline
                time.sleep(0.01)
        finally:
            checkpoints.close()
            tasks.close()

        ready = [context.Event(), context.Event()]
        start = context.Event()
        contenders = [
            context.Process(
                target=_recover_in_fresh_process,
                args=(paths, task_id, ready[index], start, output),
            )
            for index in range(2)
        ]
        for contender in contenders:
            contender.start()
        assert all(event.wait(timeout=20) for event in ready)
        start.set()
        outcomes = [output.get(timeout=30) for _ in contenders]
        for contender in contenders:
            contender.join(timeout=20)
            assert contender.exitcode == 0
        assert all(outcome[2] == 0 for outcome in outcomes)

        final_tasks = SQLiteTaskStore(paths[1])
        try:
            task = final_tasks.get(task_id)
            assert task.status == "completed"
            assert task.result == "durable result"
        finally:
            final_tasks.close()
    finally:
        receive.close()
        send.close()
        for child in [owner, *contenders]:
            if child is not None:
                if child.is_alive():
                    child.terminate()
                child.join(timeout=5)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=5)
                child.close()


def test_two_recovery_processes_contend_for_queued_task_before_worker_claim(tmp_path):
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "workspace"
    root.mkdir()
    paths = tuple(
        str(path)
        for path in (
            tmp_path / "checkpoint.sqlite",
            tmp_path / "tasks.sqlite",
            tmp_path / "inbox.sqlite",
            tmp_path / "registry.sqlite",
            root,
        )
    )
    receive, send = context.Pipe(duplex=False)
    gate = context.Event()
    owner = context.Process(target=_crash_before_worker_claim, args=(paths, send, gate))
    contenders = []
    output = context.Queue()
    start = context.Event()
    try:
        owner.start()
        assert receive.poll(20)
        task_id = receive.recv()
        gate.set()
        owner.join(timeout=10)
        assert owner.exitcode == 74

        tasks = SQLiteTaskStore(paths[1])
        checkpoints = SQLiteCheckpointStore(paths[0])
        try:
            task = tasks.get(task_id)
            assert task.status == "queued"
            assert task.metadata["initial_call_params"] == {"task": "Start"}
            assert tasks.get_worker_lease(task_id) is None
            assert checkpoints.load_state("worker", "thread", task_id) is None
        finally:
            tasks.close()
            checkpoints.close()

        ready = [context.Event(), context.Event()]
        contenders = [
            context.Process(
                target=_recover_in_fresh_process,
                args=(paths, task_id, ready[index], start, output),
            )
            for index in range(2)
        ]
        for contender in contenders:
            contender.start()
        assert all(event.wait(timeout=20) for event in ready)
        start.set()
        outcomes = [output.get(timeout=30) for _ in contenders]
        for contender in contenders:
            contender.join(timeout=20)
            assert contender.exitcode == 0

        assert sum(outcome[2] for outcome in outcomes) == 1
        assert any(outcome[0] == "ok" for outcome in outcomes)
        assert any(outcome[0] == "observed" for outcome in outcomes)
        final_tasks = SQLiteTaskStore(paths[1])
        try:
            task = final_tasks.get(task_id)
            assert task.status == "completed"
            assert task.result == "durable result"
        finally:
            final_tasks.close()
    finally:
        receive.close()
        send.close()
        for child in [owner, *contenders]:
            if child is not None:
                if child.is_alive():
                    child.terminate()
                child.join(timeout=5)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=5)
                child.close()


def test_expired_worker_cannot_publish_after_recovery_owner_claims_lease(tmp_path):
    tasks = SQLiteTaskStore(str(tmp_path / "tasks.sqlite"))
    try:
        tasks.create(task_id="task", tool_name="worker", metadata={})
        old_lease = tasks.claim_worker("task", "old-owner", lease_seconds=0.01)
        assert old_lease is not None
        deadline = time.monotonic() + 2
        while tasks.get_worker_lease("task").expires_at > time.time():
            assert time.monotonic() < deadline
            time.sleep(0.005)

        new_lease = tasks.claim_worker(
            "task", "recovery-owner", lease_seconds=5, recover_expired=True
        )
        assert new_lease is not None
        assert tasks.complete("task", "stale result", owner_id="old-owner") is None
        assert tasks.get("task").status == "running"
        assert tasks.get("task").result is None
        assert tasks.get_worker_lease("task").owner_id == "recovery-owner"

        assert tasks.complete("task", "recovered result", owner_id="recovery-owner")
        assert tasks.get("task").result == "recovered result"
    finally:
        tasks.close()


def test_uncertain_approval_blocks_recovery_even_after_lease_expires(tmp_path):
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "workspace"
    root.mkdir()
    paths = tuple(
        str(path)
        for path in (
            tmp_path / "checkpoint.sqlite",
            tmp_path / "tasks.sqlite",
            tmp_path / "inbox.sqlite",
            tmp_path / "registry.sqlite",
            root,
        )
    )
    receive, send = context.Pipe(duplex=False)
    gate = context.Event()
    owner = context.Process(target=_crash_before_worker_claim, args=(paths, send, gate))
    workspace = None
    registry = None
    checkpoints = tasks = inbox_store = None
    try:
        owner.start()
        assert receive.poll(20)
        task_id = receive.recv()
        gate.set()
        owner.join(timeout=10)
        assert owner.exitcode == 74

        checkpoints = SQLiteCheckpointStore(paths[0])
        tasks = SQLiteTaskStore(paths[1])
        inbox_store = SQLiteAgentInboxStore(paths[2])
        registry, _backend, workspace = _workspace(paths)
        task = tasks.get(task_id)
        assert task.status == "queued"
        lease = tasks.claim_worker(task_id, "expired-owner", lease_seconds=0.01)
        assert lease is not None
        deadline = time.monotonic() + 2
        while tasks.get_worker_lease(task_id).expires_at > time.time():
            assert time.monotonic() < deadline
            time.sleep(0.005)

        worker = Agent(
            name="worker",
            model=_FixedModel(),
            checkpoint_store=checkpoints,
            workspace=workspace,
        )
        worker.tool_config = {"background": True}
        library = ToolLibrary(name="lib", tools=[worker], task_store=tasks)
        library.set_agent_inbox(AgentInbox(owner="root", store=inbox_store))
        recovery = AgentTaskRecovery(library)
        with execution_context(
            scope=ExecutionScope(workspace=workspace), checkpoint_store=checkpoints
        ):
            lease_only = recovery.inspect(task_id)
            assert lease_only.classification == "recoverable"
            checkpoints.save_state(
                "worker",
                task.metadata["checkpoint_thread_id"],
                task_id,
                {
                    "status": "running",
                    "runtime": {
                        "extensions": {
                            "pending_approvals": {
                                "schema_version": 1,
                                "phase": "executing",
                            }
                        }
                    },
                },
            )
            uncertain = recovery.inspect(task_id)
            assert uncertain.classification == "uncertain"
            assert any(
                "approval execution outcome" in reason for reason in uncertain.reasons
            )
            with pytest.raises(RuntimeError, match="execution is uncertain"):
                recovery.recover(task_id, "Continue", worker_stopped=True)
    finally:
        receive.close()
        send.close()
        if owner.is_alive():
            owner.terminate()
        owner.join(timeout=5)
        owner.close()
        if workspace is not None:
            asyncio.run(workspace.aclose())
        if registry is not None:
            registry.close()
        for store in (checkpoints, tasks, inbox_store):
            if store is not None:
                store.close()
