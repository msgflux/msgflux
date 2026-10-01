"""Opt-in end-to-end recovery of a queued Agent into a reconnected Docker workspace."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from functools import partial
import json
import os
import shutil
import subprocess
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
    DockerWorkspaceBackend,
    PermissionSet,
    ResourcePermission,
)
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.tasks import SQLiteTaskStore
from msgflux.tools.builtin import BashTool


WORKSPACE_ID = "agent-task-recovery-docker"
IMAGE = "python:3.12-slim"


class _ToolThenFinalModel:
    model_type = "chat_completion"

    def __init__(self):
        self.calls = 0

    def __call__(self, **_kwargs):
        self.calls += 1
        response = ModelResponse()
        if self.calls == 1:
            response.set_response_type("tool_call")
            calls = ToolCallAggregator()
            calls.process(
                0,
                "recovery-bash-call",
                "bash",
                json.dumps(
                    {
                        "command": (
                            "printf 'recovered-in-docker\\n' > recovery.txt && "
                            "python -c 'import sys; "
                            'open("python-minor.txt", "w").write('
                            "str(sys.version_info.minor))'"
                        )
                    }
                ),
            )
            response.add(calls)
        else:
            response.set_response_type("text_generation")
            response.add("The recovered Docker command completed.")
        response.reasoning = None
        response.metadata = {}
        return response

    async def acall(self, **kwargs):
        return self(**kwargs)


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


def _backend(root, registry_path, image_id):
    return DockerWorkspaceBackend(
        root,
        image=image_id,
        registry=SQLiteWorkspaceRegistry(registry_path),
    )


def _worker(checkpoint_store, model, *, workspace=None):
    agent = Agent(
        name="recovery_worker",
        model=model,
        checkpoint_store=checkpoint_store,
        workspace=workspace,
        tools=[BashTool()],
    )
    agent.tool_config = {"background": True}
    return agent


def _wait_for_terminal(task_store, task_id, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = task_store.get(task_id)
        if task is not None and task.status in {"completed", "failed", "interrupted"}:
            return task
        time.sleep(0.05)
    raise AssertionError(f"Agent task {task_id} did not reach a terminal state")


@pytest.mark.asyncio
async def test_queued_agent_recovers_in_reconnected_docker_workspace(  # noqa: C901
    tmp_path, monkeypatch
):
    if os.environ.get("MSGFLUX_TEST_DOCKER") != "1":
        pytest.skip("Set MSGFLUX_TEST_DOCKER=1 for real Docker recovery tests")
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is not installed")
    try:
        image_id = subprocess.check_output(  # noqa: S603 -- fixed Docker CLI lookup
            [docker, "image", "inspect", IMAGE, "--format", "{{.Id}}"],
            text=True,
            timeout=10,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"{IMAGE} is not present locally; no pull attempted: {exc}")
    if not image_id.startswith("sha256:"):
        pytest.skip("Docker did not return an immutable image ID")

    root = tmp_path / "workspace"
    root.mkdir()
    state = tmp_path / "host-state"
    state.mkdir()
    registry_path = state / "workspace.sqlite"
    checkpoint_path = state / "checkpoints.sqlite"
    task_path = state / "tasks.sqlite"
    inbox_path = state / "inbox.sqlite"

    first_workspace = None
    first_registry = None
    first_checkpoints = None
    first_tasks = None
    first_inbox_store = None
    resumed_workspace = None
    resumed_registry = None
    resumed_checkpoints = None
    resumed_tasks = None
    resumed_inbox_store = None
    task_id = None
    held_futures = []
    try:
        first_registry = SQLiteWorkspaceRegistry(registry_path)
        first_backend = DockerWorkspaceBackend(
            root,
            image=image_id,
            registry=first_registry,
        )
        first_workspace = await AgentWorkspace.open(
            first_backend,
            WORKSPACE_ID,
            permissions=_permissions(),
            write_guarantee="cooperative_compare",
        )
        identity = first_workspace.identity

        first_checkpoints = SQLiteCheckpointStore(checkpoint_path)
        first_tasks = SQLiteTaskStore(task_path)
        first_inbox_store = SQLiteAgentInboxStore(inbox_path)
        first_library = ToolLibrary(
            name="recovery_host",
            tools=[
                _worker(
                    first_checkpoints,
                    _ToolThenFinalModel(),
                    workspace=first_workspace,
                )
            ],
            task_store=first_tasks,
        )
        first_library.set_agent_inbox(AgentInbox(owner="host", store=first_inbox_store))

        # Let the real dispatcher persist its complete task metadata, but hold
        # submission so the simulated first controller exits before claiming.
        from msgflux._private.executor import Executor

        original_submit = Executor.submit

        def hold_submission(executor, callable_, *args, **kwargs):
            target = callable_.func if isinstance(callable_, partial) else callable_
            if getattr(target, "__name__", None) == "run_tool":
                future = Future()
                held_futures.append(future)
                return future
            return original_submit(executor, callable_, *args, **kwargs)

        monkeypatch.setattr(Executor, "submit", hold_submission)
        dispatch = first_library([("start", "recovery_worker", {"task": "Run"})])
        task_id = dispatch.tool_calls[0].result.split("task_id='")[1].split("'")[0]
        queued = first_tasks.get(task_id)
        assert queued is not None and queued.status == "queued"
        assert queued.metadata["workspace_reference"]["identity"] == {
            "backend": identity.backend,
            "resource_id": identity.resource_id,
            "generation": identity.generation,
            "config_revision": identity.config_revision,
        }
        assert first_tasks.get_worker_lease(task_id) is None
        monkeypatch.undo()

        # Close every original binding/store before recreating them to model a
        # separate host process with no original dispatcher or in-memory state.
        await first_workspace.aclose()
        first_workspace = None
        first_registry.close()
        first_registry = None
        first_checkpoints.close()
        first_checkpoints = None
        first_tasks.close()
        first_tasks = None
        first_inbox_store.close()
        first_inbox_store = None

        resumed_registry = SQLiteWorkspaceRegistry(registry_path)
        resumed_backend = DockerWorkspaceBackend(
            root,
            image=image_id,
            registry=resumed_registry,
        )
        resumed_workspace = await AgentWorkspace.reconnect(
            resumed_backend,
            WORKSPACE_ID,
            identity,
            permissions=_permissions(),
            write_guarantee="cooperative_compare",
        )
        resumed_checkpoints = SQLiteCheckpointStore(checkpoint_path)
        resumed_tasks = SQLiteTaskStore(task_path)
        resumed_inbox_store = SQLiteAgentInboxStore(inbox_path)
        model = _ToolThenFinalModel()
        recovered_library = ToolLibrary(
            name="recovery_host",
            tools=[_worker(resumed_checkpoints, model)],
            task_store=resumed_tasks,
        )
        recovered_library.set_agent_inbox(
            AgentInbox(owner="host", store=resumed_inbox_store)
        )

        coordinator = AgentTaskRecovery(
            recovered_library,
            workspace=resumed_workspace,
        )
        report = coordinator.inspect(task_id)
        assert report.classification == "recoverable", report
        assert report.task_status == "queued"
        assert report.workspace_status == "compatible"
        assert report.inbox_status == "compatible"
        assert model.calls == 0

        scheduled = coordinator.recover(
            task_id,
            "Continue the saved task.",
            worker_stopped=True,
        )
        assert "recovered" in scheduled.lower()
        completed = _wait_for_terminal(resumed_tasks, task_id)
        assert completed.status == "completed", completed.error
        assert resumed_tasks.get_worker_lease(task_id) is None
        assert model.calls == 2
        assert resumed_workspace.read_text("/recovery.txt") == "recovered-in-docker\n"
        assert resumed_workspace.read_text("/python-minor.txt") == "12"
        checkpoint = resumed_checkpoints.load_state(
            queued.metadata["checkpoint_namespace"],
            queued.metadata["checkpoint_thread_id"],
            queued.metadata["checkpoint_run_id"],
        )
        assert checkpoint["status"] == "completed"
        assert checkpoint["task_result"]["value"] == completed.result
    finally:
        monkeypatch.undo()
        for workspace in (resumed_workspace, first_workspace):
            if workspace is not None:
                await workspace.aclose()
        for store in (
            resumed_inbox_store,
            resumed_tasks,
            resumed_checkpoints,
            resumed_registry,
            first_inbox_store,
            first_tasks,
            first_checkpoints,
            first_registry,
        ):
            if store is not None:
                store.close()
        for future in held_futures:
            future.cancel()
