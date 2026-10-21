"""Opt-in live native-tool approval recovery with recreated workspace dependencies.

MSGFLUX_LIVE_AGENT_WORKSPACE=1 enables billable requests. Docker additionally
requires MSGFLUX_TEST_DOCKER=1 and a locally installed python:3.12-slim image.
Independent-process/crash semantics are covered by the offline recovery tests;
this suite checks provider history and native tools after dependency recreation.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess

import pytest

import msgflux as mf
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.chat_transport import HTTPChatTransport
from msgflux.nn import Agent
from msgflux.nn.extensions import ToolTurnLimitExtension
from msgflux.runtime import (
    AgentApprovals,
    AgentWorkspace,
    DockerWorkspaceBackend,
    ExecutionScope,
    LocalWorkspaceBackend,
    PermissionSet,
    SandboxRequirements,
    SQLiteApprovalStore,
)
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.tools.builtin import ApplyPatchTool, ReadFileTool


@pytest.mark.skipif(
    os.getenv("MSGFLUX_LIVE_AGENT_WORKSPACE") != "1",
    reason="Set MSGFLUX_LIVE_AGENT_WORKSPACE=1 for live approval recovery",
)
@pytest.mark.parametrize("backend_kind", ["local", "docker"])
@pytest.mark.asyncio
async def test_live_native_patch_approval_survives_dependency_recreation(
    tmp_path, backend_kind
):
    mf.load_dotenv(os.getenv("MSGFLUX_TEST_DOTENV", ".env"))
    key_env = os.getenv("MSGFLUX_LIVE_WORKSPACE_KEY_ENV", "OPENAI_API_KEY")
    if not os.getenv(key_env):
        pytest.skip(f"{key_env} is not configured")
    image = None
    if backend_kind == "docker":
        if os.getenv("MSGFLUX_TEST_DOCKER") != "1":
            pytest.skip("Set MSGFLUX_TEST_DOCKER=1 for real Docker recovery")
        image = subprocess.check_output(  # noqa: S603 -- trusted image lookup
            [
                shutil.which("docker"),
                "image",
                "inspect",
                "python:3.12-slim",
                "--format",
                "{{.Id}}",
            ],
            text=True,
            timeout=15,
        ).strip()
    root = tmp_path / "project"
    root.mkdir()
    (root / "status.txt").write_text("status: BLUE-ORBIT-42\n")
    registry_path = tmp_path / "workspaces.sqlite3"
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    approval_path = tmp_path / "approvals.sqlite3"
    permissions = PermissionSet(
        ["filesystem.read", "filesystem.write", "process.execute", "process.workspace"]
    )
    system_prompt = (
        "Use apply_patch to update status.txt from the exact line "
        "'status: BLUE-ORBIT-42' to 'status: GREEN-ORBIT-42'. "
        "The file exists. Call apply_patch once; after that call succeeds, "
        "call read on status.txt to verify, then answer with GREEN-ORBIT-42. "
        "If the conversation already contains a successful patch output, "
        "do not patch again. Never finish before the read verifies the update."
    )

    def make_backend(registry):
        if backend_kind == "docker":
            return DockerWorkspaceBackend(root, image=image, registry=registry)
        return LocalWorkspaceBackend(root, registry=registry)

    def make_model():
        return mf.Model.chat_completion(
            os.getenv("MSGFLUX_LIVE_WORKSPACE_MODEL", "openai/gpt-6-luna"),
            api_key_env=key_env,
            api_mode="responses",
            reasoning_effort="medium",
            max_tokens=1536,
            chat_transport=HTTPChatTransport(timeout=60, max_retries=0),
            retry=False,
        )

    def make_agent(model, workspace, checkpoints, approvals):
        return Agent(
            name="reconnecting_editor",
            model=model,
            workspace=workspace,
            tools=[ApplyPatchTool(), ReadFileTool()],
            checkpoint_store=checkpoints,
            approvals=AgentApprovals(approvals, {"apply_patch": "v1"}, "p1"),
            extensions=[ToolTurnLimitExtension(5, warn_remaining=0)],
            system_prompt=system_prompt,
        )

    def scope(workspace):
        return ExecutionScope(
            thread_id="live-reconnect",
            run_id="edit",
            principal="host",
            workspace=workspace,
        )

    settings = {
        "permissions": permissions,
        "write_guarantee": "cooperative_compare",
        "requirements": SandboxRequirements(),
    }
    registry = SQLiteWorkspaceRegistry(registry_path)
    checkpoints = SQLiteCheckpointStore(checkpoint_path)
    approvals = SQLiteApprovalStore(approval_path)
    workspace = await AgentWorkspace.open(make_backend(registry), "project", **settings)
    model = make_model()
    try:
        agent = make_agent(model, workspace, checkpoints, approvals)
        async with asyncio.timeout(180):
            with pytest.raises(TaskPauseRequestedError):
                await agent.acall("Update the status now.", scope=scope(workspace))
        identity = workspace.identity
        pending = approvals.pending("reconnecting_editor", "live-reconnect", "edit")
        assert len(pending) == 1
        request_id = pending[0].request_id
        assert workspace.read_text("status.txt") == "status: BLUE-ORBIT-42\n"
        preview = agent.inspect_approval_preview("live-reconnect", "edit", request_id)
        assert "GREEN-ORBIT-42" in preview.diff
    finally:
        await model.aclose()
        await workspace.aclose()
        checkpoints.close()
        approvals.close()
        registry.close()

    # Recreate all runtime dependencies: the old binding and object registry are gone.
    registry = SQLiteWorkspaceRegistry(registry_path)
    checkpoints = SQLiteCheckpointStore(checkpoint_path)
    approvals = SQLiteApprovalStore(approval_path)
    workspace = await AgentWorkspace.reconnect(
        make_backend(registry), "project", identity, **settings
    )
    model = make_model()
    try:
        assert workspace.identity == identity
        agent = make_agent(model, workspace, checkpoints, approvals)
        assert (
            agent.inspect_approval_preview("live-reconnect", "edit", request_id)
            == preview
        )
        agent.decide_approval(request_id, approved=True, decided_by="host")
        async with asyncio.timeout(180):
            answer = await agent.acall(
                "Continue the approved edit.", scope=scope(workspace)
            )
        assert "GREEN-ORBIT-42" in str(answer)
        assert workspace.read_text("status.txt") == "status: GREEN-ORBIT-42\n"
        assert approvals.get("reconnecting_editor", request_id).status == "consumed"
        assert approvals.pending("reconnecting_editor", "live-reconnect", "edit") == []
        assert (
            checkpoints.load_state("reconnecting_editor", "live-reconnect", "edit")[
                "status"
            ]
            == "completed"
        )
        if backend_kind == "docker":
            result = await workspace.arun(["cat", "status.txt"])
            assert result.returncode == 0
            assert result.stdout == b"status: GREEN-ORBIT-42\n"
    finally:
        await model.aclose()
        await workspace.aclose()
        checkpoints.close()
        approvals.close()
        registry.close()
