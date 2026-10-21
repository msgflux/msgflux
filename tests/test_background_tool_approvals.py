"""Approvals protect Bash execution before foreground effects or task admission."""

import asyncio
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from msgflux.exceptions import TaskPauseRequestedError
from msgflux.coding.config import coding_permission_set
from msgflux.nn.hooks import Hook
from msgflux.runtime.context import ExecutionScope, execution_context
from msgflux.runtime.isolation import SandboxRequirements
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.runtime.workspace.policy import WorkspacePolicyState
from msgflux.runtime.workspace.local import LocalWorkspaceBackend
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.tools.builtin import BashTool
from tests.test_workspace_approvals import _agent, _policy, _response, _tool_call


def _setup_bash(tmp_path, background, *, always_background=False, workspace=None):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    workspace = workspace or AgentWorkspace.local(root)
    state = WorkspacePolicyState(workspace, _policy("thread", workspace))
    scope = ExecutionScope(
        thread_id="thread",
        namespace="coder",
        run_id="run",
        principal="user",
        workspace=workspace,
        permissions=workspace.permissions,
    )
    bash = BashTool()
    bash.tool_config = {**bash.tool_config, "allow_background": True}
    if always_background:
        bash.tool_config["background"] = True
    arguments = {"command": "printf approved > marker.txt"}
    if not always_background:
        arguments["run_in_background"] = background
    agent = _agent(
        "coder",
        tmp_path / "state",
        workspace,
        [
            _tool_call(
                "bash",
                arguments,
            ),
            _response(),
        ],
        bash,
    )
    return root, agent, scope, state


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "background,always_background", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("approved", [False, True])
async def test_bash_approval_precedes_effect_and_task_admission(
    tmp_path, monkeypatch, background, always_background, approved
):
    root, agent, scope, state = _setup_bash(
        tmp_path, background, always_background=always_background
    )
    try:
        with execution_context(scope=scope, workspace_policy=state):
            with pytest.raises(TaskPauseRequestedError):
                await agent.acall("Run the command", scope=scope)
            resources = agent._owned_threads[scope.thread_id].resources
            (record,) = resources.approval_store.pending(
                scope.namespace, scope.thread_id, scope.run_id
            )
            assert not (root / "marker.txt").exists()
            assert resources.task_store.list() == []
            original_create = resources.task_store.create
            admissions = []

            def create(*args, **kwargs):
                decision = resources.approval_store.get(
                    scope.namespace, record.request_id
                )
                assert decision.status == "consumed"
                admissions.append(kwargs)
                return original_create(*args, **kwargs)

            monkeypatch.setattr(resources.task_store, "create", create)
            agent.decide_approval(
                record.request_id,
                approved=approved,
                decided_by="user",
                expected_revision=record.revision,
            )
            assert await agent.acall("Continue", scope=scope) == "done"
            if approved and background:
                async with asyncio.timeout(5):
                    while not resources.task_store.list(status="completed"):
                        await asyncio.sleep(0.01)
            if approved:
                assert (root / "marker.txt").read_text() == "approved"
                assert (
                    resources.approval_store.get(
                        scope.namespace, record.request_id
                    ).status
                    == "consumed"
                )
            else:
                assert not (root / "marker.txt").exists()
            assert len(admissions) == int(approved and background)
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
async def test_hook_cannot_change_approved_bash_dispatch(tmp_path, background):
    root, agent, scope, state = _setup_bash(tmp_path, background)
    try:
        with execution_context(scope=scope, workspace_policy=state):
            with pytest.raises(TaskPauseRequestedError):
                await agent.acall("Run", scope=scope)
            resources = agent._owned_threads[scope.thread_id].resources
            (record,) = resources.approval_store.pending(
                scope.namespace, scope.thread_id, scope.run_id
            )
            agent.decide_approval(record.request_id, approved=True, decided_by="user")
            Hook(
                event="before_tool",
                handler=lambda payload: replace(
                    payload,
                    arguments={
                        **payload.arguments,
                        "run_in_background": not background,
                    },
                ),
            ).register(agent)
            assert await agent.acall("Continue", scope=scope) == "done"
            assert not (root / "marker.txt").exists()
            assert resources.task_store.list() == []
            assert (
                resources.approval_store.get(scope.namespace, record.request_id).status
                == "approved"
            )
    finally:
        await agent.aclose()


async def _persistent_workspace(path):
    root = path / "workspace"
    root.mkdir(exist_ok=True)
    registry = SQLiteWorkspaceRegistry(path / "workspaces.sqlite3")
    workspace = await AgentWorkspace.open(
        LocalWorkspaceBackend(root, registry=registry, allow_processes=True),
        "approval-test",
        permissions=coding_permission_set(("read", "write", "execute")),
        requirements=SandboxRequirements(),
    )
    return workspace, registry


async def _crash_before_admission(path):
    workspace, _registry = await _persistent_workspace(path)
    _root, agent, scope, state = _setup_bash(path, True, workspace=workspace)
    with execution_context(scope=scope, workspace_policy=state):
        with pytest.raises(TaskPauseRequestedError):
            await agent.acall("Run", scope=scope)
        resources = agent._owned_threads[scope.thread_id].resources
        (record,) = resources.approval_store.pending(
            scope.namespace, scope.thread_id, scope.run_id
        )
        agent.decide_approval(record.request_id, approved=True, decided_by="user")

        def lose_process(*args, **kwargs):
            assert (
                resources.approval_store.get(scope.namespace, record.request_id).status
                == "consumed"
            )
            os._exit(91)

        resources.task_store.create = lose_process
        await agent.acall("Continue", scope=scope)


@pytest.mark.asyncio
async def test_loss_between_approval_consumption_and_admission_does_not_replay(
    tmp_path,
):
    process = subprocess.run(  # noqa: S603 - fixed interpreter and local fixture
        [
            sys.executable,
            "-c",
            "import asyncio,sys; from pathlib import Path; "
            "from tests.test_background_tool_approvals import _crash_before_admission; "
            "asyncio.run(_crash_before_admission(Path(sys.argv[1])))",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert process.returncode == 91, process.stderr.decode()
    workspace, registry = await _persistent_workspace(tmp_path)
    root, agent, scope, state = _setup_bash(tmp_path, True, workspace=workspace)
    try:
        with execution_context(scope=scope, workspace_policy=state):
            with pytest.raises(TaskPauseRequestedError, match="reconciliation"):
                await agent.acall("Retry", scope=scope)
            resources = agent._owned_threads[scope.thread_id].resources
            assert resources.task_store.list() == []
            assert not (root / "marker.txt").exists()
            assert agent.generator.aforward.await_count == 0
    finally:
        await agent.aclose()
        registry.close()


@pytest.mark.asyncio
async def test_dispatch_hook_cannot_force_approved_background_call_to_foreground(
    tmp_path,
):
    root, agent, scope, state = _setup_bash(tmp_path, True)
    try:
        with execution_context(scope=scope, workspace_policy=state):
            with pytest.raises(TaskPauseRequestedError):
                await agent.acall("Run", scope=scope)
            resources = agent._owned_threads[scope.thread_id].resources
            (record,) = resources.approval_store.pending(
                scope.namespace, scope.thread_id, scope.run_id
            )
            agent.decide_approval(record.request_id, approved=True, decided_by="user")
            Hook(
                event="before_dispatch",
                handler=lambda payload: replace(payload, dispatch_mode="foreground"),
            ).register(agent)
            assert await agent.acall("Continue", scope=scope) == "done"
            assert resources.task_store.list() == []
            assert not (root / "marker.txt").exists()
            assert (
                resources.approval_store.get(scope.namespace, record.request_id).status
                == "approved"
            )
    finally:
        await agent.aclose()
