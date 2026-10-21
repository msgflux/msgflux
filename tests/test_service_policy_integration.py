"""Managed workspace policy updates re-evaluate pending approval batches."""

import asyncio
import os
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.coding import CodingSession
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime.events import EventType
from msgflux.runtime.permissions import PermissionSet
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.tools.builtin import ApplyPatchTool
from msgflux.utils.msgspec import msgspec_dumps


def _response(*, tool_call=False):
    response = ModelResponse()
    if tool_call:
        calls = ToolCallAggregator()
        calls.process(
            0,
            "patch-call",
            "apply_patch",
            msgspec_dumps(
                {
                    "operation": "update",
                    "path": "note.txt",
                    "diff": "@@\n-old\n+new",
                }
            ),
        )
        response.set_response_type("tool_call")
        response.add(calls)
    else:
        response.set_response_type("text_generation")
        response.add("finished")
    return response


def _managed_session(tmp_path, *, read_only=False):
    root = tmp_path / "workspace"
    root.mkdir()
    target = root / "note.txt"
    target.write_text("old")
    workspace = AgentWorkspace.local(
        root, read_only=read_only, approval_policy="on-request"
    )
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="policy-agent",
        model=model,
        agent_dir=tmp_path / "agent-state",
        workspace=workspace,
        tools=[ApplyPatchTool()],
    )
    agent.generator.aforward = AsyncMock(
        side_effect=[_response(tool_call=True), _response()]
    )
    session = CodingSession(agent, thread_id="policy-thread")
    return session, workspace, agent, target


async def _pause_with_patch(session):
    admission = await session.prompt("update note", request_id="patch-request")
    settled = await asyncio.wait_for(session.wait("patch-request"), timeout=5)
    assert settled.status == "paused"
    return admission


@pytest.mark.asyncio
async def test_full_access_never_update_resumes_valid_patch_once(tmp_path):
    if os.name != "posix":
        pytest.skip("Local workspace requires POSIX")
    session, workspace, agent, target = _managed_session(tmp_path)
    try:
        admission = await _pause_with_patch(session)
        before = await session.workspace_policy()
        updated = await session.update_workspace_policy(
            permissions="full-access",
            approval_policy="never",
            expected_revision=before.revision,
        )
        assert updated.approval_policy == "never"
        settled = await asyncio.wait_for(session.wait("patch-request"), timeout=5)
        assert settled.status == "completed"
        assert settled.run_id == admission.run_id
        assert target.read_text() == "new"
        assert agent.generator.aforward.await_count == 2
        records = agent._owned_threads[
            session.thread_id
        ].resources.approval_store._records(
            session.namespace, session.thread_id, admission.run_id
        )
        assert [record.status for record in records] == ["consumed"]
    finally:
        await session.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_read_only_never_keeps_pending_review_and_reports_permission_block(
    tmp_path,
):
    if not os.name == "posix":
        pytest.skip("Local workspace requires POSIX")
    session, workspace, agent, target = _managed_session(tmp_path)
    try:
        admission = await _pause_with_patch(session)
        async with session.watch() as watcher:
            before = await session.workspace_policy()
            await session.update_workspace_policy(
                permissions="read-only",
                approval_policy="never",
                expected_revision=before.revision,
            )
            policy_event = None
            for _ in range(10):
                event = await asyncio.wait_for(anext(watcher), timeout=5)
                if event.type == EventType.WORKSPACE_POLICY_UPDATED:
                    policy_event = event
                    break
        assert policy_event is not None
        error = policy_event.data["resume_error"]
        assert error["code"] == "APPROVAL_REEVALUATION_BLOCKED"
        assert "permission" in error["message"].lower()
        settled = await asyncio.wait_for(session.wait("patch-request"), timeout=5)
        assert settled.status == "paused"
        assert settled.run_id == admission.run_id
        assert target.read_text() == "old"
        (review,) = await session.approval_reviews(admission.run_id)
        assert review.status == "pending"
        assert review.diff is not None
        assert agent.generator.aforward.await_count == 1
    finally:
        await session.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalidator",
    ["modified-file", "inconsistent-batch", "denied", "expired", "consumed"],
)
async def test_never_update_does_not_resume_stale_or_settled_approval(
    tmp_path, invalidator
):
    if not os.name == "posix":
        pytest.skip("Local workspace requires POSIX")
    session, workspace, agent, target = _managed_session(tmp_path)
    try:
        admission = await _pause_with_patch(session)
        resources = agent._owned_threads[session.thread_id].resources
        pending_state = resources.checkpoint_store.load_state(
            session.namespace, session.thread_id, admission.run_id
        )
        pending = pending_state["runtime"]["extensions"]["pending_approvals"]
        request_id = next(iter(pending["requests"].values()))
        approval = resources.approval_store.get(session.namespace, request_id)

        if invalidator == "modified-file":
            target.write_text("changed externally")
        elif invalidator == "inconsistent-batch":
            pending["intents"][0]["arguments"]["diff"] = "@@\n-old\n+tampered"
            resources.checkpoint_store.save_state(
                session.namespace, session.thread_id, admission.run_id, pending_state
            )
        elif invalidator == "denied":
            await session.decide_approval(
                admission.run_id,
                request_id,
                approved=False,
                expected_revision=approval.revision,
            )
        elif invalidator == "expired":
            store = resources.approval_store
            store._conn.execute(
                "UPDATE runtime_approvals SET payload=json_set(payload, "
                "'$.created_at', 0, '$.expires_at', 1, '$.updated_at', 0) "
                "WHERE namespace=? AND request_id=?",
                (session.namespace, request_id),
            )
            store._conn.commit()
            assert store.get(session.namespace, request_id).status == "expired"
        else:
            approved = await session.decide_approval(
                admission.run_id,
                request_id,
                approved=True,
                expected_revision=approval.revision,
            )
            with session._binding.context(
                session._binding.scope(session.thread_id, run_id=admission.run_id)
            ):
                policy = agent._get_workspace_approvals(
                    force=True, _policy_version=approval.binding.policy_version
                )
                record = policy.store.get(session.namespace, request_id)
                policy.store.consume(request_id, binding=record.binding)

        before = await session.workspace_policy()
        await session.update_workspace_policy(
            permissions="full-access",
            approval_policy="never",
            expected_revision=before.revision,
        )
        settled = await asyncio.wait_for(session.wait("patch-request"), timeout=5)
        assert settled.status == "paused"
        assert settled.run_id == admission.run_id
        assert agent.generator.aforward.await_count == 1
        if invalidator == "modified-file":
            assert target.read_text() == "changed externally"
        elif invalidator in {"denied", "expired"}:
            assert (
                resources.approval_store.get(session.namespace, request_id).status
                == invalidator
            )
        elif invalidator == "consumed":
            assert (
                resources.approval_store.get(session.namespace, request_id).status
                == "consumed"
            )
        else:
            assert target.read_text() == "old"
    finally:
        await session.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_policy_reopens_per_thread_and_preserves_new_readonly_host_ceiling(
    tmp_path,
):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = AgentWorkspace.local(root)
    first_model = Mock()
    first_model.model_type = "chat_completion"
    first_agent = Agent(
        name="reopen-agent",
        model=first_model,
        agent_dir=tmp_path / "agent-state",
        workspace=workspace,
    )
    first = CodingSession(first_agent, thread_id="reopen-thread")
    try:
        initial = await first.workspace_policy()
        persisted = await first.update_workspace_policy(
            permissions="read-only",
            approval_policy="on-request",
            expected_revision=initial.revision,
        )
        assert persisted.permissions == ("filesystem.list", "filesystem.read")
        await first.aclose()

        readonly_workspace = AgentWorkspace.from_environment(
            workspace._environment,
            permissions=PermissionSet({"filesystem.read", "filesystem.list"}),
        )
        second_model = Mock()
        second_model.model_type = "chat_completion"
        second_agent = Agent(
            name="reopen-agent",
            model=second_model,
            agent_dir=tmp_path / "agent-state",
            workspace=readonly_workspace,
        )
        second = CodingSession(second_agent, thread_id="reopen-thread")
        try:
            reopened = await second.workspace_policy()
            assert reopened.revision == persisted.revision
            assert reopened.approval_policy == "on-request"
            assert reopened.permissions == ("filesystem.list", "filesystem.read")
            policy = await second.update_workspace_policy(
                permissions="full-access", expected_revision=reopened.revision
            )
            assert policy.permissions == ("filesystem.list", "filesystem.read")
        finally:
            await second.aclose()
            await readonly_workspace.aclose()
    finally:
        await first.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_observing_managed_policy_does_not_create_thread_databases(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = AgentWorkspace.local(root)
    agent_dir = tmp_path / "agent-state"
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="lazy-policy",
        model=model,
        agent_dir=agent_dir,
        workspace=workspace,
    )
    session = CodingSession(agent, thread_id="lazy-thread")
    try:
        assert not (agent_dir / "threads").exists()
        initial = await session.workspace_policy()
        assert initial.thread_id == session.thread_id
        assert not (agent_dir / "threads").exists()
        assert not (agent_dir / "runtime" / "service.sqlite3").exists()
    finally:
        await session.aclose()
        await workspace.aclose()
