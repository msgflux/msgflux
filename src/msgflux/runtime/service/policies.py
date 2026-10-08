"""Host-side policy selection; persisted policy never supplies new authority."""

from contextlib import contextmanager

import msgspec

from msgflux.runtime.approvals.agent import prepare_workspace_change
from msgflux.runtime.permissions import (
    PermissionSet,
    intersect_permissions,
    require_permissions,
)
from msgflux.runtime.service.approvals import review_record
from msgflux.runtime.service.records import ServiceRecoveryRequiredError
from msgflux.runtime.workspace.policy import WorkspacePolicy
from msgflux.tools.runtime import ToolError, ToolIntent, ToolOutcome


def initial_policy(thread_id, workspace):
    permissions = workspace.permissions if workspace is not None else PermissionSet()
    return WorkspacePolicy(
        thread_id=thread_id,
        permissions=tuple(sorted(permissions.grants)),
        resources=tuple(
            sorted(permissions.resources, key=lambda r: (r.resource, r.action))
        ),
        approval_policy=workspace.approval_policy if workspace is not None else "never",
    )


def clipped_policy(policy, workspace):
    permissions = (
        workspace.effective_permissions(
            intersect_permissions(
                workspace.permissions,
                policy.permission_set(),
                workspace_id=workspace.workspace_id,
            )
        )
        if workspace is not None
        else PermissionSet()
    )
    return msgspec.structs.replace(
        policy,
        permissions=tuple(sorted(permissions.grants)),
        resources=tuple(
            sorted(permissions.resources, key=lambda r: (r.resource, r.action))
        ),
    )


def requested_permissions(value, workspace):
    if isinstance(value, PermissionSet):
        return value
    if value == "full-access":
        return workspace.permissions
    if value == "read-only":
        return PermissionSet({"filesystem.read", "filesystem.list"})
    raise ValueError("permissions must be PermissionSet, 'read-only' or 'full-access'")


@contextmanager
def approval_review_context(session, thread_id, run_id):
    """Inspect a pending batch using its recorded policy, without executing it."""
    with session.context(session.scope(thread_id, run_id=run_id)):
        policy = session.agent.approvals
        if policy is None and session._managed:
            state = session.checkpoint_store.load_state(
                session.namespace, thread_id, run_id
            )
            pending = (
                (state or {})
                .get("runtime", {})
                .get("extensions", {})
                .get("pending_approvals", {})
            )
            policy = session.agent._get_workspace_approvals(
                force=True, _policy_version=pending.get("policy_version")
            )
        with session.agent._approval_context(policy):
            yield policy


def _approval_candidate(service, session, thread_id):
    store = session.checkpoint_store
    state = store.load_latest_run(session.namespace, thread_id) if store else None
    if state is None or state.get("status") != "paused":
        return None
    pending = state.get("runtime", {}).get("extensions", {}).get("pending_approvals")
    if not pending or not pending.get("requests"):
        return None
    if pending.get("schema_version") != 1 or pending.get("phase") == "executing":
        raise ServiceRecoveryRequiredError(
            "Pending approval batch requires reconciliation"
        )
    run_id = state.get("scope", {}).get("run_id")
    record = service.store.get_for_run(thread_id, run_id)
    if record is None or record.receipt.status != "paused":
        raise ServiceRecoveryRequiredError("Pending approvals have no paused admission")
    if record.owner_id != service._owner_id:
        raise ServiceRecoveryRequiredError(
            "Establish old-worker quiescence before resuming"
        )
    if (thread_id, record.receipt.request_id) in service._workers:
        raise ServiceRecoveryRequiredError("The pending run still has a local worker")
    service._recovery_checkpoint(session, record.receipt)
    return record, pending


def _validate_pending_decisions(session, thread_id, run_id, pending, policy):
    decisions = []
    intents = {item["id"]: item for item in pending["intents"]}
    for call_id, request_id in pending["requests"].items():
        review = review_record(session, thread_id, run_id, request_id)
        if review.status not in {"pending", "approved"}:
            raise ServiceRecoveryRequiredError(
                f"Approval {request_id} is {review.status}"
            )
        intent = intents[call_id]
        definition = session.agent.tool_library.get_tool_definition(intent["name"])
        require_permissions(
            definition.required_permissions, definition.required_resources
        )
        change = prepare_workspace_change(definition, intent["arguments"])
        if isinstance(change, ToolOutcome):
            message = (
                change.error.message
                if change.error
                else "Cannot prepare workspace change"
            )
            raise ServiceRecoveryRequiredError(message)
        current = policy.binding(
            session.agent.tool_library,
            ToolIntent(id=call_id, name=intent["name"], arguments=intent["arguments"]),
            change=change,
        )
        saved = policy.store.get(session.namespace, request_id)
        if saved is None or saved.binding != current:
            raise ServiceRecoveryRequiredError(
                "Workspace content changed since the approval preview was prepared"
            )
        decisions.append(review)
    return decisions


def ready_approval_resume(service, session, thread_id):
    """Return a validated admission, or a concrete reason it remains paused."""
    if not session._managed or session.approval_reviewer is None:
        return None, None
    try:
        candidate = _approval_candidate(service, session, thread_id)
        if candidate is None:
            return None, None
        record, pending = candidate
        run_id = record.receipt.run_id
        with approval_review_context(session, thread_id, run_id) as policy:
            if policy is None:
                raise ServiceRecoveryRequiredError(
                    "Pending batch has no live approval policy"
                )
            decisions = _validate_pending_decisions(
                session, thread_id, run_id, pending, policy
            )
            # Validate the whole batch before recording any new decision.
            for review in decisions:
                if review.status == "pending":
                    session.agent.decide_approval(
                        review.request_id,
                        approved=True,
                        decided_by=session.approval_reviewer,
                        expected_revision=review.revision,
                    )
            return record, None
    except (ValueError, KeyError, PermissionError, RuntimeError) as error:
        return None, ToolError(
            code="APPROVAL_REEVALUATION_BLOCKED",
            message=str(error),
            details={"exception_type": type(error).__name__},
        )
