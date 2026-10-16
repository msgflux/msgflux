"""Restart adoption validates the saved batch without settling decisions."""

from dataclasses import replace

import pytest

from msgflux.runtime import AgentApprovals
from msgflux.runtime.service import ServiceRecoveryRequiredError
from msgflux.runtime.service.recovery import validate_approval_recovery
from msgflux.runtime.workspace.receipts import CommandReceipt
from msgflux.tasks import InMemoryTaskStore
from msgflux.utils.time import utc_now_isoformat
from tests.test_agent_service_approvals import _close_service, _paused_review_fixture
from tests.test_service_policy_integration import _managed_session, _pause_with_patch


@pytest.mark.asyncio
@pytest.mark.parametrize("approved", [False, True])
async def test_recovery_validation_keeps_decisions_and_effects_unchanged(approved):
    (
        service,
        agent,
        approvals,
        calls,
        _scope,
        thread,
        receipt,
        review,
    ) = await _paused_review_fixture()
    try:
        if approved:
            await service.decide_approval(
                thread.thread_id,
                receipt.run_id,
                review.request_id,
                approved=True,
                expected_revision=review.revision,
            )
        before = approvals.get(agent.get_module_name(), review.request_id)
        record = service.store.get(thread.thread_id, receipt.request_id)
        async with service._use_session(thread.thread_id) as session:
            state = validate_approval_recovery(service, session, record)
        assert state["status"] == "paused"
        assert approvals.get(agent.get_module_name(), review.request_id) == before
        assert calls == []
        assert agent.generator.aforward.await_count == 1
    finally:
        await _close_service(service)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason", ["consumed", "denied", "expired", "tool_revision", "principal", "task"]
)
async def test_recovery_refuses_changed_or_uncertain_execution(reason):
    (
        service,
        agent,
        approvals,
        calls,
        scope,
        thread,
        receipt,
        review,
    ) = await _paused_review_fixture()
    try:
        record = service.store.get(thread.thread_id, receipt.request_id)
        async with service._use_session(thread.thread_id) as session:
            if reason == "consumed":
                approvals._update(
                    session.namespace,
                    review.request_id,
                    lambda item: replace(
                        item, status="consumed", decided_by="operator"
                    ),
                )
            elif reason == "denied":
                await service.decide_approval(
                    thread.thread_id,
                    receipt.run_id,
                    review.request_id,
                    approved=False,
                    expected_revision=review.revision,
                )
            elif reason == "expired":
                approvals._clock = lambda: review.expires_at + 1
            elif reason == "tool_revision":
                agent.approvals = AgentApprovals(
                    approvals, {"lookup": "v2"}, "policy-v1"
                )
            elif reason == "principal":
                scope["principal"] = "different-executor"
            else:
                session.task_store = InMemoryTaskStore()
                session.task_store.create(
                    "lookup",
                    metadata={
                        "thread_id": thread.thread_id,
                        "root_run_id": receipt.run_id,
                    },
                )
            with pytest.raises(ServiceRecoveryRequiredError):
                validate_approval_recovery(service, session, record)
            # Remove only the synthetic background row before normal cleanup.
            if reason == "task":
                session.task_store = None
        assert calls == []
        assert agent.generator.aforward.await_count == 1
    finally:
        await _close_service(service)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("malformation", "message"),
    [
        ("duplicate", "call IDs are duplicated"),
        ("missing", "has no matching intent"),
        ("missing_arguments", "intent is malformed"),
        ("executing", "requires host reconciliation"),
        ("empty_requests", "requires host reconciliation"),
        ("requests_list", "requests are malformed"),
        ("unsupported_schema", "unsupported"),
        ("malformed_intents", "intents are malformed"),
        ("request_id", "request IDs are malformed"),
    ],
)
async def test_recovery_refuses_malformed_or_executing_batches(malformation, message):
    (
        service,
        agent,
        _approvals,
        calls,
        _scope,
        thread,
        receipt,
        _review,
    ) = await _paused_review_fixture()
    try:
        state = agent.checkpoint_store.load_state(
            agent.get_module_name(), thread.thread_id, receipt.run_id
        )
        pending = state["runtime"]["extensions"]["pending_approvals"]
        if malformation == "duplicate":
            pending["intents"].append(dict(pending["intents"][0]))
        elif malformation == "missing":
            pending["intents"][0]["id"] = "unknown-call"
        elif malformation == "missing_arguments":
            del pending["intents"][0]["arguments"]
        elif malformation == "executing":
            pending["phase"] = "executing"
        elif malformation == "empty_requests":
            pending["requests"] = {}
        elif malformation == "requests_list":
            pending["requests"] = []
        elif malformation == "unsupported_schema":
            pending["schema_version"] = 2
        elif malformation == "malformed_intents":
            pending["intents"] = None
        else:
            pending["requests"]["call_1"] = None
        agent.checkpoint_store.save_state(
            agent.get_module_name(), thread.thread_id, receipt.run_id, state
        )
        record = service.store.get(thread.thread_id, receipt.request_id)
        async with service._use_session(thread.thread_id) as session:
            with pytest.raises(ServiceRecoveryRequiredError, match=message):
                validate_approval_recovery(service, session, record)
        assert calls == []
    finally:
        await _close_service(service)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["permissions", "preview"])
async def test_recovery_refuses_managed_permission_or_preview_drift(tmp_path, change):
    session, workspace, agent, target = _managed_session(tmp_path)
    try:
        receipt = await _pause_with_patch(session)
        before = await session.approval_reviews(receipt.run_id)
        if change == "permissions":
            policy = await session.workspace_policy()
            await session.update_workspace_policy(
                permissions="read-only", expected_revision=policy.revision
            )
            message = "Missing tool resource permissions"
        else:
            target.write_text("externally changed")
            message = "Invalid Context 0"
        service = session.service
        record = service.store.get(session.thread_id, receipt.request_id)
        async with service._use_session(session.thread_id) as binding:
            with pytest.raises(ServiceRecoveryRequiredError, match=message):
                validate_approval_recovery(service, binding, record)
        after = await session.approval_reviews(receipt.run_id)
        assert after == before
        assert agent.generator.aforward.await_count == 1
        assert target.read_text() == (
            "old" if change == "permissions" else "externally changed"
        )
    finally:
        await session.aclose()
        await workspace.aclose()


@pytest.mark.asyncio
async def test_recovery_refuses_terminal_background_task_with_uncertain_command():
    (
        service,
        _agent,
        _approvals,
        calls,
        _scope,
        thread,
        receipt,
        _review,
    ) = await _paused_review_fixture()
    try:
        record = service.store.get(thread.thread_id, receipt.request_id)
        async with service._use_session(thread.thread_id) as session:
            tasks = InMemoryTaskStore()
            task = tasks.create(
                "bash",
                metadata={"thread_id": thread.thread_id, "root_run_id": receipt.run_id},
            )
            timestamp = utc_now_isoformat()
            command = CommandReceipt(
                version=1,
                execution_id="uncertain-command",
                task_id=task.task_id,
                run_id=receipt.run_id,
                state="launched",
                workspace_reference=None,
                backend="local",
                created_at=timestamp,
                updated_at=timestamp,
            )
            tasks.add_activity(
                task.task_id,
                kind="command_receipt",
                summary="Launched before host crash",
                metadata={"receipt": command.to_dict()},
            )
            tasks.fail(task.task_id, "Host lost execution")
            session.task_store = tasks
            with pytest.raises(
                ServiceRecoveryRequiredError, match="Background command outcome"
            ):
                validate_approval_recovery(service, session, record)
            session.task_store = None
        assert calls == []
    finally:
        await _close_service(service)
