"""Actual workspace edits reviewed through the native remote session API."""

import asyncio
import os
from unittest.mock import AsyncMock, Mock

import httpx2
import pytest

pytest.importorskip("litestar")

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentApprovals, SQLiteApprovalStore
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http import AgentServiceClient, AgentSessionClient
from msgflux.runtime.service.http.app import create_service_app
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.tools.builtin import EditTool
from msgflux.utils.msgspec import msgspec_dumps


def _response(tool=False):
    response = ModelResponse()
    if tool:
        calls = ToolCallAggregator()
        calls.process(
            0,
            "edit-call",
            "edit",
            msgspec_dumps({"path": "note.txt", "old": "old", "new": "new"}),
        )
        response.set_response_type("tool_call")
        response.add(calls)
    else:
        response.set_response_type("text_generation")
        response.add("finished")
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "deny", "changed"])
async def test_remote_review_decision_and_explicit_resume(tmp_path, decision):
    if os.name != "posix":
        pytest.skip("Local workspace requires POSIX")
    target = tmp_path / "note.txt"
    target.write_text("old")
    workspace = AgentWorkspace.local(tmp_path)
    checkpoints = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite3")
    approvals = SQLiteApprovalStore(tmp_path / "approvals.sqlite3")
    journal = SQLiteServiceStore(tmp_path / "service.sqlite3")
    agent = Agent(
        name="main",
        model=Mock(model_type="chat_completion"),
        tools=[EditTool()],
        workspace=workspace,
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(approvals, {"edit": "v1"}, "p1"),
    )
    agent.generator.aforward = AsyncMock(side_effect=[_response(True), _response()])
    service = AgentService(store=journal)
    service.register(
        "main",
        lambda _: AgentSession(
            agent,
            approval_reviewer="owner",
            scope_factory=lambda scope: scope.with_overrides(principal="executor"),
        ),
    )
    app = create_service_app(service, token="owner-token")
    transport = httpx2.ASGITransport(app=app)
    try:
        async with httpx2.AsyncClient(transport=transport) as http:
            client = AgentServiceClient(
                "http://testserver", token="owner-token", client=http
            )
            session = await AgentSessionClient.open(client, thread_id="project")
            admitted = await session.prompt("update file", request_id="edit-request")
            paused = await asyncio.wait_for(session.wait("edit-request"), 5)
            assert paused.status == "paused"
            assert target.read_text() == "old"
            (review,) = await session.approval_reviews(admitted.run_id)
            assert "-old" in review.diff and "+new" in review.diff
            assert review.status == "pending"
            chosen = await session.decide_approval(
                admitted.run_id,
                review.request_id,
                approved=decision != "deny",
                expected_revision=review.revision,
            )
            # Retrying an acknowledgement never changes the journal revision.
            repeated = await session.decide_approval(
                admitted.run_id,
                review.request_id,
                approved=decision != "deny",
                expected_revision=review.revision,
            )
            assert chosen == repeated
            assert target.read_text() == "old"
            if decision == "changed":
                target.write_text("external update")
            resumed = await session.resume(admitted.run_id)
            assert resumed.run_id == admitted.run_id
            settled = await asyncio.wait_for(session.wait("edit-request"), 5)
            if decision == "approve":
                assert settled.status == "completed"
                assert target.read_text() == "new"
                assert agent.generator.aforward.await_count == 2
            elif decision == "deny":
                assert settled.status == "completed"
                assert target.read_text() == "old"
            else:
                assert target.read_text() == "external update"
                assert settled.status != "completed"
    finally:
        await service.aclose()
        await workspace.aclose()
        checkpoints.close()
        approvals.close()
        journal.close()
