"""Service level approval review and decision authorization."""

from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import (
    AgentApprovals,
    ApprovalExpiredError,
    InMemoryApprovalStore,
)
from msgflux.runtime.service import AgentService, AgentSession
from msgflux.runtime.service.store import SQLiteServiceStore
from msgflux.runtime.workspace.api import AgentWorkspace


def _text_response(text="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    return response


def _fixture(reviewer="operator"):
    checkpoints = InMemoryCheckpointStore()
    approvals = InMemoryApprovalStore()
    calls = []

    def lookup(query: str) -> str:
        calls.append(query)
        return "found"

    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="approval-review-service",
        model=model,
        tools=[lookup],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(approvals, {"lookup": "v1"}, "policy-v1"),
    )
    tool_calls = ToolCallAggregator()
    tool_calls.process(0, "call_1", "lookup", '{"query":"sensitive query"}')
    tool_response = ModelResponse()
    tool_response.set_response_type("tool_call")
    tool_response.add(tool_calls)
    agent.generator.aforward = AsyncMock(
        side_effect=[tool_response, _text_response("done")]
    )
    service = AgentService(store=SQLiteServiceStore())
    scope_options = {"principal": "executor", "workspace": None}
    service.register(
        "main",
        lambda _thread: AgentSession(
            agent,
            checkpoint_store=checkpoints,
            approval_reviewer=reviewer,
            scope_factory=lambda scope: scope.with_overrides(**scope_options),
        ),
    )
    return service, agent, approvals, calls, scope_options


async def _close_service(service):
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
async def test_review_requires_configured_reviewer_and_projects_no_arguments():
    service, _agent, _approvals, calls, _scope_options = _fixture(reviewer=None)
    thread = await service.open_thread("main", thread_id="approval-review")
    try:
        receipt = await service.prompt(thread.thread_id, "lookup", request_id="request")
        await service.wait(thread.thread_id, "request")
        with pytest.raises(PermissionError):
            await service.approval_reviews(thread.thread_id, receipt.run_id)
        assert calls == []
    finally:
        await _close_service(service)


@pytest.mark.asyncio
async def test_service_review_decision_is_bound_and_does_not_resume_execution():
    service, agent, approvals, calls, _scope_options = _fixture()
    thread = await service.open_thread("main", thread_id="approval-review")
    try:
        receipt = await service.prompt(thread.thread_id, "lookup", request_id="request")
        paused = await service.wait(thread.thread_id, "request")
        assert paused.status == "paused"
        reviews = await service.approval_reviews(thread.thread_id, receipt.run_id)
        assert len(reviews) == 1
        review = reviews[0]
        assert review.tool_name == "lookup"
        assert review.status == "pending"
        assert review.revision == 1
        assert "sensitive query" not in repr(review)
        assert not hasattr(review, "arguments")

        decided = await service.decide_approval(
            thread.thread_id,
            receipt.run_id,
            review.request_id,
            approved=True,
            expected_revision=review.revision,
        )
        assert decided.status == "approved"
        assert decided.revision == 2
        assert (
            approvals.get("approval-review-service", review.request_id).decided_by
            == "operator"
        )
        assert calls == []
        assert agent.generator.aforward.await_count == 1

        # The same reviewer can retry a committed decision after losing the response.
        retried = await service.decide_approval(
            thread.thread_id,
            receipt.run_id,
            review.request_id,
            approved=True,
            expected_revision=review.revision,
        )
        assert retried == decided
        assert agent.generator.aforward.await_count == 1
    finally:
        await _close_service(service)


async def _paused_review_fixture(reviewer="operator"):
    service, agent, approvals, calls, scope_options = _fixture(reviewer)
    thread = await service.open_thread("main", thread_id="approval-review")
    receipt = await service.prompt(thread.thread_id, "lookup", request_id="request")
    paused = await service.wait(thread.thread_id, "request")
    assert paused.status == "paused"
    review = (await service.approval_reviews(thread.thread_id, receipt.run_id))[0]
    return service, agent, approvals, calls, scope_options, thread, receipt, review


@pytest.mark.asyncio
async def test_review_rejects_wrong_run_thread_and_request_membership():
    (
        service,
        _agent,
        _approvals,
        _calls,
        _scope_options,
        thread,
        receipt,
        review,
    ) = await _paused_review_fixture()
    try:
        with pytest.raises((KeyError, ValueError)):
            await service.approval_reviews(thread.thread_id, "wrong-run")
        with pytest.raises(KeyError):
            await service.approval_reviews("wrong-thread", receipt.run_id)
        with pytest.raises(KeyError):
            await service.decide_approval(
                thread.thread_id,
                receipt.run_id,
                "wrong-request",
                approved=True,
                expected_revision=review.revision,
            )
    finally:
        await _close_service(service)


@pytest.mark.asyncio
async def test_review_rejects_changed_executor_policy_and_checkpoint_arguments():
    (
        service,
        agent,
        approvals,
        _calls,
        scope_options,
        thread,
        receipt,
        _review,
    ) = await _paused_review_fixture()
    try:
        scope_options["principal"] = "different-executor"
        with pytest.raises(PermissionError):
            await service.approval_reviews(thread.thread_id, receipt.run_id)
        scope_options["principal"] = "executor"

        agent.approvals = AgentApprovals(approvals, {"lookup": "v2"}, "policy-v2")
        with pytest.raises(ValueError, match="policy or tool revision"):
            await service.approval_reviews(thread.thread_id, receipt.run_id)
        agent.approvals = AgentApprovals(approvals, {"lookup": "v1"}, "policy-v1")

        state = agent.checkpoint_store.load_state(
            agent.get_module_name(), thread.thread_id, receipt.run_id
        )
        state["runtime"]["extensions"]["pending_approvals"]["intents"][0]["arguments"][
            "query"
        ] = "changed query"
        agent.checkpoint_store.save_state(
            agent.get_module_name(), thread.thread_id, receipt.run_id, state
        )
        with pytest.raises(ValueError, match=r"arguments|binding"):
            await service.approval_reviews(thread.thread_id, receipt.run_id)
    finally:
        await _close_service(service)


@pytest.mark.asyncio
async def test_review_rejects_changed_workspace_resource_binding(tmp_path):
    (
        service,
        _agent,
        _approvals,
        _calls,
        scope_options,
        thread,
        receipt,
        _review,
    ) = await _paused_review_fixture()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace = AgentWorkspace.local(workspace_root)
    try:
        scope_options["workspace"] = workspace
        with pytest.raises(ValueError, match="binding"):
            await service.approval_reviews(thread.thread_id, receipt.run_id)
    finally:
        await _close_service(service)
        await workspace.aclose()


@pytest.mark.asyncio
async def test_decision_rejects_invalid_expected_revision():
    (
        service,
        _agent,
        _approvals,
        _calls,
        _scope_options,
        thread,
        receipt,
        review,
    ) = await _paused_review_fixture()
    try:
        for invalid in (None, True, 1.0):
            with pytest.raises((TypeError, ValueError)):
                await service.decide_approval(
                    thread.thread_id,
                    receipt.run_id,
                    review.request_id,
                    approved=True,
                    expected_revision=invalid,
                )
    finally:
        await _close_service(service)


@pytest.mark.asyncio
async def test_expired_approval_cannot_be_decided():
    service, _agent, approvals, calls, _scope_options = _fixture()
    thread = await service.open_thread("main", thread_id="approval-review")
    try:
        receipt = await service.prompt(thread.thread_id, "lookup", request_id="request")
        await service.wait(thread.thread_id, "request")
        review = (await service.approval_reviews(thread.thread_id, receipt.run_id))[0]
        # Advance the journal clock beyond its absolute deadline.
        approvals._clock = lambda: review.expires_at + 1
        with pytest.raises(ApprovalExpiredError):
            await service.decide_approval(
                thread.thread_id,
                receipt.run_id,
                review.request_id,
                approved=True,
                expected_revision=review.revision,
            )
        assert calls == []
    finally:
        await _close_service(service)
