"""Pending local approvals survive abrupt daemon death without replaying effects."""

import os
import signal

import pytest
import pytest_asyncio

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")
pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Local daemon uses POSIX locks"
)

from msgflux.runtime.service.local.discovery import connect_local_service
from msgflux.runtime.service.local.files import process_lock, read_record
from msgflux.runtime.service.records import ServiceRecoveryRequiredError
from tests.test_local_service_integration import (
    FACTORY,
    _until,
    _wait_completed,
    daemon_workspace,
)

_APPROVAL_FACTORY_CODE = """
import json
import time
from unittest.mock import AsyncMock, Mock
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentApprovals, SQLiteApprovalStore
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore


def create_service(runtime_dir):
    with (runtime_dir / "boots").open("a") as stream:
        stream.write("boot\\n")
    service = AgentService(store=SQLiteServiceStore(runtime_dir / "service.sqlite3"))

    def create_agent(thread):
        checkpoints = SQLiteCheckpointStore(runtime_dir / "approval-checkpoints.sqlite3")
        approvals = SQLiteApprovalStore(runtime_dir / "approvals.sqlite3")
        policy_file = runtime_dir / "policy-version"
        policy_version = policy_file.read_text() if policy_file.exists() else "policy-v1"

        def record_effect(value: str) -> str:
            '''Record a durable external effect exactly when invoked.'''
            with (runtime_dir / "effects.log").open("a") as stream:
                stream.write(value + "\\n")
            if (runtime_dir / "block-tool").exists():
                (runtime_dir / "tool-started").touch()
                while True:
                    time.sleep(0.02)
            return "recorded " + value

        agent = Agent(
            name="recovery-agent", model=Mock(model_type="chat_completion"),
            tools=[record_effect], checkpoint_store=checkpoints,
            approvals=AgentApprovals(approvals, {"record_effect": "v1"}, policy_version),
        )

        async def respond(**kwargs):
            with (runtime_dir / "model_calls").open("a") as stream:
                stream.write("call\\n")
            response = ModelResponse()
            if any(item.get("role") == "tool" for item in kwargs["messages"].to_chatml()):
                response.set_response_type("text_generation")
                response.add("finished")
            else:
                calls = ToolCallAggregator()
                calls.process(0, "effect-call", "record_effect", json.dumps({"value": "one"}))
                response.set_response_type("tool_call")
                response.add(calls)
            return response

        def close_stores():
            checkpoints.close()
            approvals.close()

        agent.generator.aforward = AsyncMock(side_effect=respond)
        return AgentSession(
            agent, on_close=close_stores, approval_reviewer="owner",
            scope_factory=lambda scope: scope.with_overrides(principal="executor"),
        )

    service.register("main", create_agent)
    return service
"""

_MANAGED_FACTORY_CODE = """
import json
from unittest.mock import AsyncMock, Mock
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentWorkspace
from msgflux.runtime.isolation import SandboxRequirements
from msgflux.runtime.permissions import PermissionSet
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.workspace.local import LocalWorkspaceBackend
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.tools.builtin import ApplyPatchTool


def create_service(runtime_dir):
    service = AgentService(store=SQLiteServiceStore(runtime_dir / "service.sqlite3"))

    async def create_agent(thread):
        root = runtime_dir / "workspace"
        root.mkdir(exist_ok=True)
        registry = SQLiteWorkspaceRegistry(runtime_dir / "workspaces.sqlite3")
        workspace = await AgentWorkspace.open(
            LocalWorkspaceBackend(root, registry=registry, allow_processes=True),
            "managed-approval-workspace",
            permissions=PermissionSet({"filesystem.read", "filesystem.list", "filesystem.write"}),
            approval_policy="on-request", write_guarantee="cooperative_compare",
            requirements=SandboxRequirements(),
        )
        agent = Agent(
            name="recovery-agent", model=Mock(model_type="chat_completion"),
            tools=[ApplyPatchTool()], agent_dir=runtime_dir / "agent",
            workspace=workspace,
        )

        async def respond(**kwargs):
            with (runtime_dir / "model_calls").open("a") as stream:
                stream.write("call\\n")
            response = ModelResponse()
            if any(item.get("role") == "tool" for item in kwargs["messages"].to_chatml()):
                response.set_response_type("text_generation")
                response.add("finished")
            else:
                calls = ToolCallAggregator()
                calls.process(0, "patch-call", "apply_patch", json.dumps({
                    "operation": "create", "path": "/hello.txt", "diff": "+Hello durable agent"
                }))
                response.set_response_type("tool_call")
                response.add(calls)
            return response

        async def close_workspace():
            try:
                await workspace.aclose()
            finally:
                registry.close()

        agent.generator.aforward = AsyncMock(side_effect=respond)
        return AgentSession(agent, on_close=close_workspace)

    service.register("main", create_agent)
    return service
"""


@pytest_asyncio.fixture
async def approval_daemon_workspace(daemon_workspace):
    project, project_b, project_c, runtime_dir = daemon_workspace
    for frontend in (project, project_b, project_c):
        (frontend / "daemon_fixture.py").write_text(_APPROVAL_FACTORY_CODE)
    yield project, runtime_dir


@pytest_asyncio.fixture
async def managed_approval_daemon_workspace(daemon_workspace):
    project, project_b, project_c, runtime_dir = daemon_workspace
    for frontend in (project, project_b, project_c):
        (frontend / "daemon_fixture.py").write_text(_MANAGED_FACTORY_CODE)
    yield project, runtime_dir


async def _crash_daemon(client, runtime_dir):
    record = read_record(runtime_dir)
    assert record is not None
    health = await client.health()
    assert health.instance_id == record.instance_id
    os.kill(record.pid, signal.SIGKILL)
    await client.aclose()

    def lock_released():
        with process_lock(runtime_dir / "daemon.lock") as handle:
            return handle is not None

    await _until(lock_released)
    return record


async def _start_paused(project, runtime_dir):
    client = await connect_local_service(FACTORY, runtime_dir=runtime_dir, cwd=project)
    try:
        thread = await client.open_thread("main", thread_id="approval-crash")
        admission = await client.prompt(
            thread.thread_id, "record one effect", request_id="one"
        )
        paused = await _wait_completed(client, thread.thread_id, "one")
        assert paused.status == "paused", paused.error
        (review,) = await client.approval_reviews(thread.thread_id, admission.run_id)
        assert review.status == "pending"
        assert not (runtime_dir / "effects.log").exists()
        return client, thread.thread_id, admission, review
    except BaseException:
        await client.aclose()
        raise


async def _approve(client, thread_id, admission, review):
    return await client.decide_approval(
        thread_id,
        admission.run_id,
        review.request_id,
        approved=True,
        expected_revision=review.revision,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("approved_before_crash", [False, True])
async def test_pending_approval_survives_sigkill_and_resumes_same_run_once(
    approval_daemon_workspace, approved_before_crash
):
    project, runtime_dir = approval_daemon_workspace
    client, thread_id, admission, review = await _start_paused(project, runtime_dir)
    if approved_before_crash:
        review = await _approve(client, thread_id, admission, review)
    previous = await _crash_daemon(client, runtime_dir)
    restarted = await connect_local_service(
        FACTORY, runtime_dir=runtime_dir, cwd=project
    )
    try:
        assert (await restarted.health()).instance_id != previous.instance_id
        (recovered,) = await restarted.approval_reviews(thread_id, admission.run_id)
        assert recovered.request_id == review.request_id
        assert recovered.revision == review.revision
        assert recovered.status == review.status
        assert (await restarted.active_runs()) == ()
        assert not (runtime_dir / "effects.log").exists()
        assert (runtime_dir / "model_calls").read_text().splitlines() == ["call"]
        if not approved_before_crash:
            waiting = await restarted.resume_checkpoint(thread_id, admission.run_id)
            assert waiting.run_id == admission.run_id
            still_paused = await _wait_completed(
                restarted, thread_id, admission.request_id
            )
            assert still_paused.status == "paused"
            assert not (runtime_dir / "effects.log").exists()
            assert (runtime_dir / "model_calls").read_text().splitlines() == ["call"]
            await _approve(restarted, thread_id, admission, recovered)
        # A recorded decision alone never authorizes automatic execution at startup.
        assert not (runtime_dir / "effects.log").exists()
        resumed = await restarted.resume_checkpoint(thread_id, admission.run_id)
        assert resumed.run_id == admission.run_id
        assert resumed.request_id == admission.request_id
        finished = await _wait_completed(restarted, thread_id, admission.request_id)
        assert finished.status == "completed", finished.error
        assert (
            await restarted.run_result(thread_id, admission.run_id)
        ).output == "finished"
        assert (runtime_dir / "effects.log").read_text() == "one\n"
        assert (runtime_dir / "model_calls").read_text().splitlines() == [
            "call",
            "call",
        ]
        duplicate = await restarted.prompt(
            thread_id, "record one effect", request_id="one"
        )
        assert duplicate.run_id == admission.run_id
        assert duplicate.status == "completed"
        assert (runtime_dir / "effects.log").read_text() == "one\n"
    finally:
        await restarted.aclose()


@pytest.mark.asyncio
async def test_sigkill_after_approved_effect_requires_reconciliation(
    approval_daemon_workspace,
):
    project, runtime_dir = approval_daemon_workspace
    client, thread_id, admission, review = await _start_paused(project, runtime_dir)
    (runtime_dir / "block-tool").touch()
    await _approve(client, thread_id, admission, review)
    await client.resume_checkpoint(thread_id, admission.run_id)
    await _until((runtime_dir / "tool-started").exists)
    assert (runtime_dir / "effects.log").read_text() == "one\n"
    await _crash_daemon(client, runtime_dir)
    restarted = await connect_local_service(
        FACTORY, runtime_dir=runtime_dir, cwd=project
    )
    try:
        assert (await restarted.active_runs()) == ()
        with pytest.raises(ServiceRecoveryRequiredError):
            await restarted.resume_checkpoint(thread_id, admission.run_id)
        assert (await restarted.run_result(thread_id, admission.run_id)).output is None
        assert (runtime_dir / "effects.log").read_text() == "one\n"
        assert (runtime_dir / "model_calls").read_text().splitlines() == ["call"]
    finally:
        await restarted.aclose()


@pytest.mark.asyncio
async def test_changed_approval_policy_after_sigkill_does_not_reuse_old_decision(
    approval_daemon_workspace,
):
    project, runtime_dir = approval_daemon_workspace
    client, thread_id, admission, review = await _start_paused(project, runtime_dir)
    await _approve(client, thread_id, admission, review)
    await _crash_daemon(client, runtime_dir)
    (runtime_dir / "policy-version").write_text("policy-v2")
    restarted = await connect_local_service(
        FACTORY, runtime_dir=runtime_dir, cwd=project
    )
    try:
        with pytest.raises(ServiceRecoveryRequiredError):
            await restarted.resume_checkpoint(thread_id, admission.run_id)
        assert (await restarted.active_runs()) == ()
        assert not (runtime_dir / "effects.log").exists()
        assert (runtime_dir / "model_calls").read_text().splitlines() == ["call"]
    finally:
        await restarted.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("workspace_changed", [False, True])
async def test_managed_apply_patch_review_survives_sigkill_and_checks_preview(
    managed_approval_daemon_workspace, workspace_changed
):
    project, runtime_dir = managed_approval_daemon_workspace
    client, thread_id, admission, review = await _start_paused(project, runtime_dir)
    target = runtime_dir / "workspace" / "hello.txt"
    assert not target.exists()
    assert review.tool_name == "apply_patch"
    assert "Hello durable agent" in review.diff
    previous = await _crash_daemon(client, runtime_dir)
    if workspace_changed:
        target.write_text("A user created this file while the service was down")
    restarted = await connect_local_service(
        FACTORY, runtime_dir=runtime_dir, cwd=project
    )
    try:
        assert (await restarted.health()).instance_id != previous.instance_id
        (recovered,) = await restarted.approval_reviews(thread_id, admission.run_id)
        assert recovered == review
        assert (await restarted.active_runs()) == ()
        assert (runtime_dir / "model_calls").read_text().splitlines() == ["call"]
        await _approve(restarted, thread_id, admission, recovered)
        if workspace_changed:
            with pytest.raises(ServiceRecoveryRequiredError, match="exists"):
                await restarted.resume_checkpoint(thread_id, admission.run_id)
            assert (
                target.read_text()
                == "A user created this file while the service was down"
            )
            assert (runtime_dir / "model_calls").read_text().splitlines() == ["call"]
            assert (await restarted.active_runs()) == ()
        else:
            assert not target.exists()
            resumed = await restarted.resume_checkpoint(thread_id, admission.run_id)
            assert resumed.run_id == admission.run_id
            settled = await _wait_completed(restarted, thread_id, admission.request_id)
            assert settled.status == "completed", settled.error
            assert target.read_text() == "Hello durable agent"
            assert (runtime_dir / "model_calls").read_text().splitlines() == [
                "call",
                "call",
            ]
            duplicate = await restarted.prompt(
                thread_id, "record one effect", request_id="one"
            )
            assert duplicate.run_id == admission.run_id
            assert duplicate.status == "completed"
            assert target.read_text() == "Hello durable agent"
            assert (runtime_dir / "model_calls").read_text().splitlines() == [
                "call",
                "call",
            ]
    finally:
        await restarted.aclose()
