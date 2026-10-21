"""Managed Agent recovery blocks replay after a command-complete crash.

The process exits after Bash and its command receipt complete, but before the
background child publishes its result to the Agent checkpoint.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import threading
from unittest.mock import AsyncMock, Mock

import msgflux as mf
import pytest

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentTaskRecovery, AgentWorkspace, PermissionSet
from msgflux.runtime.context import ExecutionScope, execution_context
from msgflux.runtime.isolation import SandboxRequirements
from msgflux.runtime.workspace.local import LocalWorkspaceBackend
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.tools.builtin import AgentTool, BashTool

_WORKSPACE_ID = "managed-background-recovery"
_THREAD_ID = "managed-recovery-thread"


def _response(content: str | None = None, *, tool: tuple[str, dict] | None = None):
    response = ModelResponse()
    if tool is None:
        response.set_response_type("text_generation")
        response.add(content or "done")
    else:
        name, arguments = tool
        calls = ToolCallAggregator()
        calls.process(0, "call-1", name, json.dumps(arguments))
        response.set_response_type("tool_call")
        response.add(calls)
    response.reasoning = None
    return response


def _permissions():
    return PermissionSet(
        {
            "filesystem.read",
            "filesystem.write",
            "filesystem.delete",
            "filesystem.mkdir",
            "process.execute",
        }
    )


def _crash_managed_background_child(
    agent_dir, workspace_root, registry_path, effect_sender, release_receiver
):
    """Crash after command completion but before publishing Agent output."""
    from msgflux.runtime.workspace.receipts import CommandExecution

    registry = SQLiteWorkspaceRegistry(registry_path)
    backend = LocalWorkspaceBackend(
        workspace_root, registry=registry, allow_processes=True
    )
    workspace = asyncio.run(
        AgentWorkspace.open(
            backend,
            _WORKSPACE_ID,
            permissions=_permissions(),
            requirements=SandboxRequirements(),
            write_guarantee="cooperative_compare",
        )
    )

    child_model = Mock(model_type="chat_completion")
    child = Agent(name="worker", model=child_model, tools=[BashTool()])
    child.generator.aforward = AsyncMock(
        return_value=_response(
            tool=("bash", {"command": "printf 'one\\n' >> effects.txt"})
        )
    )

    root_model = Mock(model_type="chat_completion")
    root = Agent(
        name="root",
        model=root_model,
        agent_dir=agent_dir,
        workspace=workspace,
    )
    root.generator.aforward = AsyncMock(
        side_effect=[
            _response(
                tool=(
                    "agent",
                    {
                        "name": "worker",
                        "message": "Run the command once",
                        "run_in_background": True,
                    },
                )
            ),
            _response("delegated"),
        ]
    )
    root.tool_library.add(mf.tool_config(allow_background=True)(AgentTool()))
    root.tool_library.add(child)

    original_update = CommandExecution.update

    async def exit_after_command_commit(execution, next_state, **kwargs):
        receipt = await original_update(execution, next_state, **kwargs)
        if next_state == "completed":
            effect_sender.send("completed")
            if not release_receiver.poll(timeout=20):
                raise TimeoutError("Parent did not release managed crash barrier")
            release_receiver.recv()
            os._exit(73)
        return receipt

    CommandExecution.update = exit_after_command_commit
    asyncio.run(
        root.acall(
            "Delegate command",
            scope=ExecutionScope(
                thread_id=_THREAD_ID,
                run_id="root-run",
                root_run_id="root-run",
                workspace=workspace,
            ),
        )
    )
    # Keep the process alive if its background task is blocked in the patched
    # update. This event is local to the process and has no shared IPC lock.
    if not threading.Event().wait(timeout=25):
        raise TimeoutError("Background command did not reach its crash barrier")


def _managed_agents(agent_dir, workspace):
    child_model = Mock(model_type="chat_completion")
    child = Agent(name="worker", model=child_model, tools=[BashTool()])
    child.generator.aforward = AsyncMock(
        side_effect=AssertionError("uncertain child command must not be replayed")
    )
    root_model = Mock(model_type="chat_completion")
    root = Agent(
        name="root",
        model=root_model,
        agent_dir=agent_dir,
        workspace=workspace,
    )
    root.generator.aforward = AsyncMock(
        side_effect=AssertionError("recovery inspection must not call the model")
    )
    root.tool_library.add(mf.tool_config(allow_background=True)(AgentTool()))
    root.tool_library.add(child)
    return root, child


def test_managed_background_command_crash_is_durable_and_refuses_uncertain_replay(
    tmp_path, monkeypatch
):
    context = multiprocessing.get_context("spawn")
    agent_dir = str(tmp_path / "managed-agent")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    registry_path = str(tmp_path / "workspace-registry.sqlite")
    effect_receiver, effect_sender = context.Pipe(duplex=False)
    release_receiver, release_sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_crash_managed_background_child,
        args=(
            agent_dir,
            str(workspace_root),
            registry_path,
            effect_sender,
            release_receiver,
        ),
    )
    registry = workspace = root_agent = child_agent = None
    release_sent = False
    try:
        process.start()
        effect_sender.close()
        release_receiver.close()
        assert effect_receiver.poll(timeout=25), "command did not reach crash barrier"
        assert effect_receiver.recv() == "completed"
        assert (workspace_root / "effects.txt").read_text() == "one\n"
        release_sender.send("exit")
        release_sent = True
        process.join(timeout=15)
        assert not process.is_alive(), "crashed worker process did not exit"
        assert process.exitcode == 73

        # Process exit establishes old-worker quiescence. Until then, the lease
        # takes precedence over command-receipt uncertainty.
        registry = SQLiteWorkspaceRegistry(registry_path)
        backend = LocalWorkspaceBackend(
            workspace_root, registry=registry, allow_processes=True
        )
        workspace_record = registry.get_record(_WORKSPACE_ID)
        workspace = asyncio.run(
            AgentWorkspace.reconnect(
                backend,
                _WORKSPACE_ID,
                workspace_record.identity,
                permissions=_permissions(),
                requirements=SandboxRequirements(),
                write_guarantee="cooperative_compare",
            )
        )
        root_agent, child_agent = _managed_agents(agent_dir, workspace)
        bundle = root_agent._bind_resources(_THREAD_ID)
        tasks = bundle.task_store
        (task,) = tasks.list()
        assert task.status == "running"
        assert task.metadata["task_kind"] == "agent"
        assert task.metadata["checkpoint_namespace"] == "worker"

        coordinator = AgentTaskRecovery(root_agent.tool_library, workspace=workspace)
        scope = ExecutionScope(
            namespace="root",
            thread_id=_THREAD_ID,
            workspace=workspace,
        )
        with execution_context(
            scope=scope,
            checkpoint_store=bundle.checkpoint_store,
            task_store=tasks,
            agent_inbox=bundle.agent_inbox,
        ):
            active = coordinator.inspect(task.task_id)
            assert active.classification == "active", active
            lease = tasks.get_worker_lease(task.task_id)
            assert lease is not None

            # Use a controlled clock only after join has proved the old process
            # is gone. Never treat elapsed lease time as proof of quiescence.
            expired_now = lease.expires_at + 1
            monkeypatch.setattr(tasks, "_clock", lambda: expired_now)
            monkeypatch.setattr(
                "msgflux.runtime.recovery.time.time", lambda: expired_now
            )

            report = coordinator.inspect(task.task_id)
            assert report.classification == "uncertain", report
            assert any(
                "command execution outcome" in reason for reason in report.reasons
            )
            with pytest.raises(RuntimeError, match="command execution outcome"):
                coordinator.recover(
                    task.task_id,
                    "Do not repeat this uncertain command",
                    worker_stopped=True,
                )

        assert (workspace_root / "effects.txt").read_text() == "one\n"
        assert child_agent.generator.aforward.await_count == 0
        assert root_agent.generator.aforward.await_count == 0
        assert tasks.get(task.task_id).status == "running"
    finally:
        if process.is_alive():
            if not release_sent:
                try:
                    release_sender.send("cleanup")
                except (BrokenPipeError, OSError):
                    pass
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()
        if child_agent is not None:
            asyncio.run(child_agent.aclose())
        if root_agent is not None:
            asyncio.run(root_agent.aclose())
        if workspace is not None:
            asyncio.run(workspace.aclose())
        if registry is not None:
            registry.close()
        effect_receiver.close()
        release_sender.close()
