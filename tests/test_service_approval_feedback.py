"""Service/HTTP/SDK approval feedback pass-through."""

from unittest.mock import AsyncMock, Mock

import httpx2
import msgspec
import pytest

pytest.importorskip("litestar")

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentApprovals, InMemoryApprovalStore
from msgflux.runtime.approvals import ApprovalConflictError
from msgflux.runtime.approvals.records import MAX_APPROVAL_FEEDBACK_LENGTH
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http import (
    AgentServiceClient,
    create_service_app,
)
from msgflux.runtime.service.http.records import ApprovalDecisionRequest


def _text(text="done"):
    r = ModelResponse()
    r.set_response_type("text_generation")
    r.add(text)
    return r


async def _fixture():
    checkpoints = InMemoryCheckpointStore()
    approvals = InMemoryApprovalStore()
    calls = []

    def lookup(query: str) -> str:
        calls.append(query)
        return "found"

    model = Mock(model_type="chat_completion")
    agent = Agent(
        name="feedback-service",
        model=model,
        tools=[lookup],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(approvals, {"lookup": "v1"}, "policy-v1"),
    )
    tc = ToolCallAggregator()
    tc.process(0, "call_1", "lookup", '{"query":"q"}')
    tool = ModelResponse()
    tool.set_response_type("tool_call")
    tool.add(tc)
    agent.generator.aforward = AsyncMock(side_effect=[tool, _text()])
    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "main",
        lambda _t: AgentSession(
            agent,
            checkpoint_store=checkpoints,
            approval_reviewer="operator",
            scope_factory=lambda s: s.with_overrides(
                principal="executor", workspace=None
            ),
        ),
    )
    thread = await service.open_thread("main", thread_id="feedback")
    receipt = await service.prompt(thread.thread_id, "lookup", request_id="r1")
    await service.wait(thread.thread_id, "r1")
    review = (await service.approval_reviews(thread.thread_id, receipt.run_id))[0]
    return service, agent, approvals, calls, thread, receipt, review


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [True, False])
async def test_service_feedback_both_decisions_no_auto_run(approved):
    service, agent, approvals, calls, thread, receipt, review = await _fixture()
    try:
        decided = await service.decide_approval(
            thread.thread_id,
            receipt.run_id,
            review.request_id,
            approved=approved,
            expected_revision=review.revision,
            feedback="  please retry carefully  ",
        )
        assert decided.status == ("approved" if approved else "denied")
        assert decided.feedback == "please retry carefully"
        record = approvals.get("feedback-service", review.request_id)
        assert record.feedback == "please retry carefully"
        assert record.decided_by == "operator"
        assert calls == []
        assert agent.generator.aforward.await_count == 1
        # Exact retry (same normalized feedback) is idempotent.
        retried = await service.decide_approval(
            thread.thread_id,
            receipt.run_id,
            review.request_id,
            approved=approved,
            expected_revision=review.revision,
            feedback="please retry carefully",
        )
        assert retried == decided
        # Changed text conflicts; original is immutable.
        with pytest.raises(ApprovalConflictError):
            await service.decide_approval(
                thread.thread_id,
                receipt.run_id,
                review.request_id,
                approved=approved,
                expected_revision=review.revision,
                feedback="different guidance",
            )
        assert approvals.get("feedback-service", review.request_id).feedback == (
            "please retry carefully"
        )
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_service_feedback_invalid_and_boundary():
    service, _a, approvals, _c, thread, receipt, review = await _fixture()
    try:
        for bad in ("   ", "", 123, "x" * (MAX_APPROVAL_FEEDBACK_LENGTH + 1)):
            with pytest.raises((TypeError, ValueError)):
                await service.decide_approval(
                    thread.thread_id,
                    receipt.run_id,
                    review.request_id,
                    approved=True,
                    expected_revision=review.revision,
                    feedback=bad,
                )
        assert approvals.get("feedback-service", review.request_id).status == "pending"
        ok = await service.decide_approval(
            thread.thread_id,
            receipt.run_id,
            review.request_id,
            approved=False,
            expected_revision=review.revision,
            feedback="x" * MAX_APPROVAL_FEEDBACK_LENGTH,
        )
        assert ok.feedback == "x" * MAX_APPROVAL_FEEDBACK_LENGTH
    finally:
        await service.aclose()
        service.store.close()


def test_http_request_bounded_and_defaults_none():
    assert ApprovalDecisionRequest(True, 1).feedback is None
    assert (
        msgspec.json.decode(
            b'{"approved":true,"expected_revision":1}', type=ApprovalDecisionRequest
        ).feedback
        is None
    )
    with pytest.raises(msgspec.ValidationError):
        msgspec.json.decode(
            msgspec.json.encode(
                {
                    "approved": True,
                    "expected_revision": 1,
                    "feedback": "x" * (MAX_APPROVAL_FEEDBACK_LENGTH + 1),
                }
            ),
            type=ApprovalDecisionRequest,
        )


@pytest.mark.asyncio
async def test_http_sdk_feedback_pass_through_denied():
    service, _agent, approvals, calls, thread, receipt, review = await _fixture()
    app = create_service_app(service, token="secret")
    transport = httpx2.ASGITransport(app=app)
    try:
        async with httpx2.AsyncClient(transport=transport, base_url="http://t") as http:
            client = AgentServiceClient("http://t", token="secret", client=http)
            # Raw HTTP: blank feedback is invalid and does not settle.
            raw = await http.post(
                f"/v1/threads/{thread.thread_id}/runs/{receipt.run_id}/approvals/{review.request_id}/decision",
                headers={"Authorization": "Bearer secret"},
                json={"approved": False, "expected_revision": 1, "feedback": "  "},
            )
            assert raw.status_code == 422
            decided = await client.decide_approval(
                thread.thread_id,
                receipt.run_id,
                review.request_id,
                approved=False,
                expected_revision=review.revision,
                feedback="do not run this",
            )
            assert decided.feedback == "do not run this"
            assert decided.status == "denied"
            assert (
                approvals.get("feedback-service", review.request_id).feedback
                == "do not run this"
            )
            assert calls == []
            with pytest.raises(ApprovalConflictError):
                await client.decide_approval(
                    thread.thread_id,
                    receipt.run_id,
                    review.request_id,
                    approved=False,
                    expected_revision=review.revision,
                    feedback="changed",
                )
            await client.aclose()
    finally:
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
async def test_explicit_resume_delivers_feedback_as_separate_user_context():
    service, agent, _approvals, calls, thread, receipt, review = await _fixture()
    try:
        decided = await service.decide_approval(
            thread.thread_id,
            receipt.run_id,
            review.request_id,
            approved=True,
            expected_revision=review.revision,
            feedback="use the smaller range",
        )
        assert decided.feedback == "use the smaller range"
        assert agent.generator.aforward.await_count == 1 and calls == []
        await service.resume_checkpoint(thread.thread_id, receipt.run_id)
        settled = await service.wait(thread.thread_id, "r1")
        assert settled.status == "completed"
        assert calls == ["q"]  # approved invocation executes unchanged
        assert agent.generator.aforward.await_count == 2
        messages = agent.generator.aforward.call_args_list[1].kwargs["messages"]
        items = messages.to_items()
        feedback_items = [
            item
            for item in items
            if isinstance(item.get("metadata"), dict)
            and "approval_feedback" in item["metadata"]
        ]
        assert len(feedback_items) == 1
        assert feedback_items[0]["role"] == "user"
        assert "use the smaller range" in str(feedback_items[0].get("content"))
        reviews = feedback_items[0]["metadata"]["approval_feedback"]["reviews"]
        assert reviews[0]["feedback"] == "use the smaller range"
        assert reviews[0]["decided_by"] == "operator"
        assert reviews[0]["approved"] is True
        # Tool result itself is unchanged and carries no feedback text.
        tool_items = [i for i in items if i.get("type") == "function_call_output"]
        assert len(tool_items) == 1 and tool_items[0].get("output") == "found"
    finally:
        await service.aclose()
        service.store.close()
