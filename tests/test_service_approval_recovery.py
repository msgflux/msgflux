"""SQLite approval review survives service restart and recovers the saved call."""

import asyncio
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
from msgflux.runtime.service.http import (
    AgentServiceClient,
    AgentSessionClient,
    create_service_app,
)
from msgflux.runtime.service.records import ServiceRecoveryRequiredError
from msgflux.utils.msgspec import msgspec_dumps


def _tool_response():
    calls = ToolCallAggregator()
    calls.process(0, "effect-call", "record_effect", msgspec_dumps({"value": "one"}))
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _text_response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


def _service(agent, checkpoints, approvals, service_store):
    service = AgentService(store=service_store)
    service.register(
        "main",
        lambda _thread: AgentSession(
            agent,
            checkpoint_store=checkpoints,
            approval_reviewer="owner",
            scope_factory=lambda scope: scope.with_overrides(principal="executor"),
        ),
    )
    return service


def _agent(checkpoints, approvals, tool):
    model = Mock(model_type="chat_completion")
    return Agent(
        name="recovery-agent",
        model=model,
        tools=[tool],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(approvals, {"record_effect": "v1"}, "policy-v1"),
    )


@pytest.mark.asyncio
async def test_sqlite_approval_review_and_decision_survive_service_restart(tmp_path):
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    approval_path = tmp_path / "approvals.sqlite3"
    service_path = tmp_path / "service.sqlite3"
    effects_path = tmp_path / "effects.log"
    effects = []

    def record_effect(value: str) -> str:
        """Record one externally visible effect."""
        with effects_path.open("a", encoding="utf-8") as handle:
            handle.write(value + "\n")
        effects.append(value)
        return f"recorded {value}"

    checkpoints = SQLiteCheckpointStore(checkpoint_path)
    approvals = SQLiteApprovalStore(approval_path)
    service_store = SQLiteServiceStore(service_path)
    first_agent = _agent(checkpoints, approvals, record_effect)
    first_agent.generator.aforward = AsyncMock(
        side_effect=[_tool_response(), _text_response("finished")]
    )
    first_service = _service(first_agent, checkpoints, approvals, service_store)
    first_app = create_service_app(first_service, token="owner-token")
    transport = httpx2.ASGITransport(app=first_app)
    request_id = "durable-approval-request"
    thread_id = "durable-approval-thread"

    try:
        async with httpx2.AsyncClient(transport=transport) as http:
            client = AgentServiceClient(
                "http://testserver", token="owner-token", client=http
            )
            session = await AgentSessionClient.open(client, thread_id=thread_id)
            admission = await session.prompt("record one effect", request_id=request_id)
            paused = await asyncio.wait_for(session.wait(request_id), timeout=5)
            assert paused.status == "paused"
            (before_restart,) = await session.approval_reviews(admission.run_id)
            assert before_restart.revision == 1
            assert effects == []
    finally:
        await first_service.aclose()
        checkpoints.close()
        approvals.close()
        service_store.close()

    reopened_checkpoints = SQLiteCheckpointStore(checkpoint_path)
    reopened_approvals = SQLiteApprovalStore(approval_path)
    reopened_service_store = SQLiteServiceStore(service_path)
    recovered_agent = _agent(reopened_checkpoints, reopened_approvals, record_effect)

    async def continue_after_saved_tool(**_kwargs):
        # If recovery asks the model to recreate the saved tool call, the
        # external effect will not exist yet and the run fails here.
        assert effects_path.read_text(encoding="utf-8") == "one\n"
        return _text_response("finished")

    recovered_agent.generator.aforward = AsyncMock(
        side_effect=continue_after_saved_tool
    )
    recovered_service = _service(
        recovered_agent,
        reopened_checkpoints,
        reopened_approvals,
        reopened_service_store,
    )
    recovered_app = create_service_app(recovered_service, token="owner-token")
    recovered_transport = httpx2.ASGITransport(app=recovered_app)

    try:
        async with httpx2.AsyncClient(transport=recovered_transport) as http:
            client = AgentServiceClient(
                "http://testserver", token="owner-token", client=http
            )
            session = await AgentSessionClient.open(client, thread_id=thread_id)
            (after_restart,) = await session.approval_reviews(admission.run_id)
            assert after_restart.request_id == before_restart.request_id
            assert after_restart.revision == before_restart.revision

            decided = await session.decide_approval(
                admission.run_id,
                after_restart.request_id,
                approved=True,
                expected_revision=after_restart.revision,
            )
            assert decided.status == "approved"
            assert decided.revision == after_restart.revision + 1
            assert effects == []

            with pytest.raises(ServiceRecoveryRequiredError):
                await session.resume(admission.run_id)

            # Only the trusted local host can assert that the old worker stopped.
            receipt = await recovered_service.resume_checkpoint(
                thread_id, admission.run_id, worker_stopped=True
            )
            assert receipt.run_id == admission.run_id
            settled = await asyncio.wait_for(session.wait(request_id), timeout=5)
            assert settled.status == "completed", settled.error
            assert effects_path.read_text(encoding="utf-8") == "one\n"
            assert effects == ["one"]
            assert recovered_agent.generator.aforward.await_count == 1
            consumed = reopened_approvals.get(
                "recovery-agent", before_restart.request_id
            )
            assert consumed.status == "consumed"
            assert consumed.revision == after_restart.revision + 2
    finally:
        await recovered_service.aclose()
        reopened_checkpoints.close()
        reopened_approvals.close()
        reopened_service_store.close()
