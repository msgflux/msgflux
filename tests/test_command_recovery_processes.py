"""Abrupt-controller recovery tests for canonical background command receipts."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
from pathlib import Path
import shlex
import signal
import sys
import time

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime import (
    AgentTaskRecovery,
    AgentWorkspace,
    PermissionSet,
    ResourcePermission,
)
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore
from msgflux.runtime.context import ExecutionScope, execution_context
from msgflux.runtime.isolation import SandboxRequirements
from msgflux.runtime.workspace.local import LocalWorkspaceBackend
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.runtime.workspace.receipts import CommandExecution
from msgflux.tasks import SQLiteTaskStore
from msgflux.tools.builtin import BashTool


WORKSPACE_ID = "command-recovery-local"


def _permissions():
    return PermissionSet(
        {
            "filesystem.read",
            "filesystem.write",
            "process.execute",
            "process.workspace",
        },
        {
            ResourcePermission(f"workspace:{WORKSPACE_ID}:/", "process.workspace"),
        },
    )


def _command_for(state, root):
    if state == "launched":
        marker = root / "started.txt"
        stop = root / "stop.txt"
        code = (
            "import pathlib,time; "
            f"pathlib.Path({str(marker)!r}).write_text('running'); "
            f"stop=pathlib.Path({str(stop)!r}); "
            "exec('while not stop.exists(): time.sleep(0.03)')"
        )
        return f"exec {shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    return "printf 'receipt-complete\\n' > completed.txt"


class _CommandModel:
    model_type = "chat_completion"

    def __init__(self, gate, command):
        self.gate = gate
        self.command = command
        self.calls = 0

    def __call__(self, **_kwargs):
        self.calls += 1
        if not self.gate.wait(timeout=20):
            raise TimeoutError("Parent did not release the deterministic model")
        response = ModelResponse()
        response.set_response_type("tool_call")
        calls = ToolCallAggregator()
        calls.process(0, "command-call", "bash", json.dumps({"command": self.command}))
        response.add(calls)
        response.reasoning = None
        response.metadata = {}
        return response

    async def acall(self, **kwargs):
        return self(**kwargs)


def _crash_controller_during_command(paths, root, state, gate, connection):
    """Spawn target: persist real dispatcher metadata, then die at a receipt edge."""
    from msgflux.runtime.workspace.receipts import CommandExecution

    root = Path(root)
    checkpoints = SQLiteCheckpointStore(paths[0])
    tasks = SQLiteTaskStore(paths[1])
    inbox_store = SQLiteAgentInboxStore(paths[2])
    registry = SQLiteWorkspaceRegistry(paths[3])
    backend = LocalWorkspaceBackend(root, registry=registry, allow_processes=True)
    workspace = asyncio.run(
        AgentWorkspace.open(
            backend,
            WORKSPACE_ID,
            permissions=_permissions(),
            requirements=SandboxRequirements(),
            write_guarantee="cooperative_compare",
        )
    )
    model = _CommandModel(gate, _command_for(state, root))
    worker = Agent(
        name="worker",
        model=model,
        checkpoint_store=checkpoints,
        workspace=workspace,
        tools=[BashTool()],
    )
    worker.tool_config = {"background": True}
    library = ToolLibrary(name="command-host", tools=[worker], task_store=tasks)
    library.set_agent_inbox(AgentInbox(owner="host", store=inbox_store))
    library.get_background_dispatcher().lease_seconds = 0.4

    original_update = CommandExecution.update
    if state in {"intent", "completed"}:

        async def exit_after_persist(execution, next_state, **kwargs):
            receipt = await original_update(execution, next_state, **kwargs)
            if next_state == state:
                os._exit(73 if state == "intent" else 74)
            return receipt

        CommandExecution.update = exit_after_persist

    scope = ExecutionScope(
        thread_id="command-thread",
        run_id="root-run",
        root_run_id="root-run",
        workspace=workspace,
    )
    with execution_context(scope=scope, checkpoint_store=checkpoints):
        dispatch = library([("start", "worker", {"task": "Run one command"})])
        task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]
        connection.send(task_id)
        connection.close()
        if state == "launched":
            gate.wait(timeout=20)
            time.sleep(20)
        else:
            gate.wait(timeout=20)
            time.sleep(20)


def _reconnect_local(paths, root):
    registry = SQLiteWorkspaceRegistry(paths[3])
    backend = LocalWorkspaceBackend(root, registry=registry, allow_processes=True)
    record = registry.get_record(WORKSPACE_ID)
    workspace = asyncio.run(
        AgentWorkspace.reconnect(
            backend,
            WORKSPACE_ID,
            record.identity,
            permissions=_permissions(),
            requirements=SandboxRequirements(),
            write_guarantee="cooperative_compare",
        )
    )
    checkpoints = SQLiteCheckpointStore(paths[0])
    tasks = SQLiteTaskStore(paths[1])
    inbox_store = SQLiteAgentInboxStore(paths[2])
    return registry, workspace, checkpoints, tasks, inbox_store


def _receipts(task_store, task_id):
    return [
        activity.metadata["receipt"]
        for activity in task_store.list_activity(task_id)
        if activity.kind == "command_receipt"
        and isinstance(activity.metadata.get("receipt"), dict)
    ]


def _wait_for(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("Timed out waiting for durable command evidence")


@pytest.mark.parametrize("crash_state", ["intent", "launched", "completed"])
def test_abrupt_controller_death_preserves_receipt_without_retry(tmp_path, crash_state):  # noqa: C901
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "workspace"
    root.mkdir()
    paths = tuple(
        str(item)
        for item in (
            tmp_path / "checkpoints.sqlite",
            tmp_path / "tasks.sqlite",
            tmp_path / "inbox.sqlite",
            tmp_path / "registry.sqlite",
        )
    )
    parent, child = context.Pipe(duplex=False)
    gate = context.Event()
    controller = context.Process(
        target=_crash_controller_during_command,
        args=(paths, str(root), crash_state, gate, child),
    )
    registry = workspace = checkpoints = tasks = inbox_store = None
    stop_marker = root / "stop.txt"
    task_id = None
    try:
        controller.start()
        assert parent.poll(20), "controller did not persist its background task"
        task_id = parent.recv()
        gate.set()

        if crash_state == "launched":
            _wait_for((root / "started.txt").exists)
            controller.kill()
        controller.join(timeout=20)
        expected = {"intent": 73, "completed": 74}.get(crash_state, -signal.SIGKILL)
        assert controller.exitcode == expected

        registry, workspace, checkpoints, tasks, inbox_store = _reconnect_local(
            paths, root
        )
        lease = tasks.get_worker_lease(task_id)
        if lease is not None:
            _wait_for(lambda: lease.expires_at <= time.time(), timeout=5)
        receipt = _wait_for(lambda: _receipts(tasks, task_id))[-1]
        assert receipt["state"] == crash_state
        assert receipt["task_id"] == task_id
        assert receipt["workspace_reference"]["workspace_id"] == WORKSPACE_ID

        recovery_model = _CommandModel(context.Event(), "should not run")
        worker = Agent(
            name="worker",
            model=recovery_model,
            checkpoint_store=checkpoints,
            tools=[BashTool()],
        )
        worker.tool_config = {"background": True}
        library = ToolLibrary(name="command-host", tools=[worker], task_store=tasks)
        library.set_agent_inbox(AgentInbox(owner="host", store=inbox_store))
        coordinator = AgentTaskRecovery(library, workspace=workspace)
        report = coordinator.inspect(task_id)
        assert report.classification == "uncertain", report
        assert any("command" in reason.lower() for reason in report.reasons)

        command_report = asyncio.run(workspace.ainspect_command(receipt))
        if crash_state == "launched":
            assert command_report.classification == "running", command_report
            stopped = asyncio.run(workspace.aterminate_command(receipt))
            assert stopped.classification in {"unknown", "blocked"}, stopped
            if stopped.classification == "blocked":
                assert stopped.resource_status == "unavailable"
                assert "pidfd" in " ".join(stopped.reasons)
        elif crash_state == "intent":
            assert command_report.classification == "unknown", command_report
            assert command_report.resource_status == "unchecked"
            assert not (root / "started.txt").exists()
        else:
            assert command_report.classification == "completed", command_report
            assert command_report.returncode == 0
            assert (root / "completed.txt").read_text() == "receipt-complete\n"

        with pytest.raises(RuntimeError, match="uncertain"):
            coordinator.recover(
                task_id, "Do not repeat this command", worker_stopped=True
            )
        assert recovery_model.calls == 0
        assert tasks.get(task_id).status == "running"
    finally:
        if not stop_marker.exists():
            stop_marker.write_text("stop")
        if controller.is_alive():
            controller.kill()
        controller.join(timeout=5)
        controller.close()
        parent.close()
        child.close()
        if workspace is not None:
            asyncio.run(workspace.aclose())
        for store in (inbox_store, tasks, checkpoints, registry):
            if store is not None:
                store.close()


def _crash_foreground_after_launch(paths, root, gate, connection):
    """Spawn target proving foreground receipts land in the checkpoint itself."""
    from msgflux.runtime.workspace.receipts import CommandExecution

    root = Path(root)
    checkpoints = SQLiteCheckpointStore(paths[0])
    registry = SQLiteWorkspaceRegistry(paths[3])
    backend = LocalWorkspaceBackend(root, registry=registry, allow_processes=True)
    workspace = asyncio.run(
        AgentWorkspace.open(
            backend,
            WORKSPACE_ID,
            permissions=_permissions(),
            requirements=SandboxRequirements(),
            write_guarantee="cooperative_compare",
        )
    )
    worker = Agent(
        name="foreground-worker",
        model=_CommandModel(gate, _command_for("launched", root)),
        checkpoint_store=checkpoints,
        workspace=workspace,
        tools=[BashTool()],
    )
    original_update = CommandExecution.update

    async def exit_after_launch(execution, state, **kwargs):
        receipt = await original_update(execution, state, **kwargs)
        if state == "launched":
            os._exit(79)
        return receipt

    CommandExecution.update = exit_after_launch
    connection.send("ready")
    connection.close()
    worker(
        "Run the process.",
        scope=ExecutionScope(
            namespace="foreground",
            thread_id="foreground-thread",
            run_id="foreground-run",
            root_run_id="foreground-run",
            workspace=workspace,
        ),
    )


def test_foreground_checkpoint_contains_canonical_command_receipt(tmp_path):
    context = multiprocessing.get_context("spawn")
    root = tmp_path / "workspace"
    root.mkdir()
    paths = tuple(
        str(tmp_path / name)
        for name in (
            "checkpoints.sqlite",
            "tasks.sqlite",
            "inbox.sqlite",
            "registry.sqlite",
        )
    )
    parent, child = context.Pipe(duplex=False)
    gate = context.Event()
    controller = context.Process(
        target=_crash_foreground_after_launch,
        args=(paths, str(root), gate, child),
    )
    checkpoints = registry = workspace = None
    stop = root / "stop.txt"
    try:
        controller.start()
        assert parent.poll(20) and parent.recv() == "ready"
        gate.set()
        _wait_for((root / "started.txt").exists)
        controller.join(timeout=20)
        assert controller.exitcode == 79

        checkpoints = SQLiteCheckpointStore(paths[0])
        state = checkpoints.load_state(
            "foreground-worker", "foreground-thread", "foreground-run"
        )
        assert state is not None
        receipts = state["runtime"]["extensions"]["command_receipts"]
        assert len(receipts) == 1
        receipt = receipts[0]
        assert receipt["state"] == "launched"
        assert receipt["workspace_reference"]["workspace_id"] == WORKSPACE_ID

        registry = SQLiteWorkspaceRegistry(paths[3])
        backend = LocalWorkspaceBackend(root, registry=registry, allow_processes=True)
        record = registry.get_record(WORKSPACE_ID)
        workspace = asyncio.run(
            AgentWorkspace.reconnect(
                backend,
                WORKSPACE_ID,
                record.identity,
                permissions=_permissions(),
                requirements=SandboxRequirements(),
                write_guarantee="cooperative_compare",
            )
        )
        inspected = asyncio.run(workspace.ainspect_command(receipt))
        assert inspected.classification == "running", inspected
        stopped = asyncio.run(workspace.aterminate_command(receipt))
        assert stopped.classification in {"unknown", "blocked"}, stopped
    finally:
        if not stop.exists():
            stop.write_text("stop")
        if controller.is_alive():
            controller.kill()
        controller.join(timeout=5)
        parent.close()
        child.close()
        if workspace is not None:
            asyncio.run(workspace.aclose())
        for store in (checkpoints, registry):
            if store is not None:
                store.close()
