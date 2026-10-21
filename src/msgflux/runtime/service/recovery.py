"""Approval recovery guards; worker ownership is established by the host."""

from msgflux.runtime.service.policies import (
    _validate_pending_decisions,
    approval_review_context,
)
from msgflux.runtime.service.records import ServiceRecoveryRequiredError
from msgflux.runtime.workspace.receipts import (
    task_command_receipts,
    unresolved_command_receipts,
)


def validate_approval_recovery(service, session, record):
    """Validate an unexecuted approval batch without deciding or dispatching it.

    This does not prove the old owner stopped. The caller must separately retain
    exclusive evidence of quiescence until its conditional ownership transfer.
    """
    try:
        return _validate_approval_recovery(service, session, record)
    except ServiceRecoveryRequiredError:
        raise
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        raise ServiceRecoveryRequiredError(str(error)) from error


def _validate_approval_recovery(service, session, record):
    receipt = record.receipt
    thread_id, run_id = receipt.thread_id, receipt.run_id
    if record.namespace != session.namespace:
        raise ServiceRecoveryRequiredError(
            "The admission namespace does not match the current session"
        )
    if receipt.status != "paused":
        raise ServiceRecoveryRequiredError("Only paused approvals can be recovered")
    if (thread_id, receipt.request_id) in service._workers:
        raise ServiceRecoveryRequiredError("The pending run still has a local worker")
    state = service._recovery_checkpoint(session, receipt)
    if state.get("status") != "paused":
        raise ServiceRecoveryRequiredError("The approval checkpoint is not paused")
    reason = session.agent._uncertain_checkpoint_approval_reason(state)
    if reason is not None:
        raise ServiceRecoveryRequiredError(reason)
    pending = state.get("runtime", {}).get("extensions", {}).get("pending_approvals")
    if not pending or not pending.get("requests"):
        raise ServiceRecoveryRequiredError("The paused run has no pending approvals")
    _validate_pending_shape(pending)
    _validate_background_work(session, thread_id, run_id)
    with approval_review_context(session, thread_id, run_id) as policy:
        if policy is None:
            raise ServiceRecoveryRequiredError(
                "Pending batch has no live approval policy"
            )
        _validate_pending_decisions(session, thread_id, run_id, pending, policy)
    return state


def _validate_pending_shape(pending):
    intents = pending.get("intents")
    if not isinstance(intents, list) or not intents:
        raise ServiceRecoveryRequiredError("Pending approval intents are malformed")
    identifiers = set()
    for intent in intents:
        if (
            not isinstance(intent, dict)
            or not isinstance(intent.get("id"), str)
            or not intent["id"]
            or not isinstance(intent.get("name"), str)
            or not intent["name"]
            or not isinstance(intent.get("arguments"), dict)
        ):
            raise ServiceRecoveryRequiredError("Pending approval intent is malformed")
        if intent["id"] in identifiers:
            raise ServiceRecoveryRequiredError(
                "Pending approval call IDs are duplicated"
            )
        identifiers.add(intent["id"])
    requests = pending["requests"]
    if not set(requests).issubset(identifiers):
        raise ServiceRecoveryRequiredError(
            "Pending approval request has no matching intent"
        )
    values = tuple(requests.values())
    if not all(isinstance(value, str) and value for value in values):
        raise ServiceRecoveryRequiredError("Pending approval request IDs are malformed")
    if len(set(values)) != len(values):
        raise ServiceRecoveryRequiredError(
            "Pending approval request IDs are duplicated"
        )


def _validate_background_work(session, thread_id, run_id):
    reason = session.agent._active_background_future_reason(
        (session.agent.tool_library,)
    )
    if reason is not None:
        raise ServiceRecoveryRequiredError(reason)
    store = session.task_store
    if store is None:
        return
    reason = session.agent._unfinished_task_reason(thread_id, store)
    if reason is not None:
        raise ServiceRecoveryRequiredError(reason)
    for task in store.list_summaries(thread_id=thread_id, run_id=run_id):
        if unresolved_command_receipts(task_command_receipts(store, task.task_id)):
            raise ServiceRecoveryRequiredError(
                "Background command outcome requires host reconciliation"
            )
