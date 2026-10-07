"""Argument-free approval review projections for AgentService."""

from msgflux.runtime.approvals.records import ApprovalRecord
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.service.records import ApprovalReview
from msgflux.tools.runtime import ToolIntent


def _checkpoint_intent(pending, record, thread_id, run_id, scope):
    binding = record.binding
    if binding.thread_id != thread_id or binding.run_id != run_id:
        raise ValueError("Approval does not belong to this run")
    if scope is None or binding.principal != scope.principal:
        raise PermissionError("Approval executor does not match this session")
    call_id = binding.tool_call_id
    if pending.get("requests", {}).get(call_id) != record.request_id:
        raise ValueError("Approval is not in the pending checkpoint")
    intent = next(
        (item for item in pending.get("intents", ()) if item.get("id") == call_id),
        None,
    )
    if intent is None or intent.get("name") != binding.tool_name:
        raise ValueError("Approval call does not match the pending checkpoint")
    return intent


def review_record(session, thread_id: str, run_id: str, request_id: str):
    """Validate a journal entry against its live checkpoint and host policy."""
    agent = session.agent
    policy = agent._get_effective_approvals()
    if policy is None:
        raise ValueError("Agent has no approval configuration")
    state = agent.inspect_approval_batch(thread_id, run_id)
    pending = (
        state.get("runtime", {}).get("extensions", {}).get("pending_approvals", {})
    )
    record = policy.store.get(session.namespace, request_id)
    if record is None:
        raise KeyError(request_id)
    binding = record.binding
    scope = get_execution_scope()
    intent = _checkpoint_intent(pending, record, thread_id, run_id, scope)
    if binding.namespace != session.namespace:
        raise ValueError("Approval namespace does not belong to this session")
    call_id = binding.tool_call_id
    if (
        policy.policy_version != binding.policy_version
        or policy.tools.get(binding.tool_name) != binding.tool_revision
    ):
        raise ValueError("Approval policy or tool revision has changed")
    arguments = intent.get("arguments", {})
    diff = None
    change = None
    if call_id in pending.get("prepared_changes", {}):
        preview = agent.inspect_approval_preview(thread_id, run_id, request_id)
        if preview is None:
            raise ValueError("Verified file preview is missing")
        change = preview
        diff = preview.diff
    current_binding = policy.binding(
        agent.tool_library,
        ToolIntent(id=call_id, name=binding.tool_name, arguments=arguments),
        change=change,
    )
    if current_binding != binding:
        raise ValueError("Approval invocation binding has changed")
    return project(record, diff=diff)


def project(record: ApprovalRecord, *, diff: str | None = None) -> ApprovalReview:
    """Project an already validated decision without exposing its binding."""
    binding = record.binding
    return ApprovalReview(
        request_id=record.request_id,
        tool_call_id=binding.tool_call_id,
        tool_name=binding.tool_name,
        status=record.status,
        revision=record.revision,
        expires_at=record.expires_at,
        diff=diff,
    )
