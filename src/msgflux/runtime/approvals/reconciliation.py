"""Host-only reconciliation of uncertain Agent tool batches."""

from collections.abc import Mapping
from copy import deepcopy

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores.base import CheckpointConflictError
from msgflux.runtime.approvals.records import require_name

PENDING_KEY = "pending_approvals"
RECEIPTS_KEY = "approval_reconciliations"


def inspect_batch(store, namespace, thread_id, run_id):
    if store is None or not store.supports_atomic_commit:
        raise ValueError("Reconciliation requires atomic checkpoints")
    state = store.load_state(namespace, thread_id, run_id)
    if state is None:
        raise ValueError("Checkpoint run not found")
    return deepcopy(dict(state))


def _validate_decision(
    expected_revision,
    decision_id,
    decided_by,
    reason,
    worker_stopped,
    results,
    abandon,
):
    require_name(decision_id)
    require_name(decided_by)
    require_name(reason)
    if worker_stopped is not True:
        raise ValueError("Host must confirm that the previous worker has stopped")
    if type(expected_revision) is not int or expected_revision < 0:
        raise ValueError("expected_revision must be a non-negative integer")
    if type(abandon) is not bool or (abandon and results is not None):
        raise ValueError("Choose confirmed results or abandon, not both")
    if not abandon and (
        not isinstance(results, Mapping)
        or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in results.items()
        )
    ):
        raise ValueError("Results must map every call ID to confirmed text")
    return {
        "decision_id": decision_id,
        "decided_by": decided_by,
        "reason": reason,
        "expected_revision": expected_revision,
        "abandon": abandon,
        "results": dict(results) if results is not None else None,
    }


def reconcile_batch(
    store,
    namespace,
    thread_id,
    run_id,
    *,
    expected_revision,
    decision_id,
    decided_by,
    reason,
    worker_stopped,
    results=None,
    abandon=False,
):
    """Commit confirmed observations; host authentication/quiescence are required."""
    decision = _validate_decision(
        expected_revision,
        decision_id,
        decided_by,
        reason,
        worker_stopped,
        results,
        abandon,
    )
    state = inspect_batch(store, namespace, thread_id, run_id)
    extensions = state.get("runtime", {}).get("extensions", {})
    receipts = extensions.setdefault(RECEIPTS_KEY, {})
    previous = receipts.get(decision_id)
    if previous is not None:
        if previous != decision:
            raise CheckpointConflictError("Reconciliation decision ID was reused")
        return deepcopy(previous)
    if state.get("_checkpoint", {}).get("revision", 0) != expected_revision:
        raise CheckpointConflictError("Reconciliation checkpoint revision changed")
    pending = extensions.get(PENDING_KEY)
    if not pending or pending.get("schema_version") != 1:
        raise ValueError("No supported pending approval batch")
    if pending.get("phase") != "executing":
        raise ValueError("Only uncertain executing batches can be reconciled")
    call_ids = [intent["id"] for intent in pending["intents"]]
    if not abandon and set(results) != set(call_ids):
        raise ValueError("Reconciliation requires results for the entire batch")
    messages = ChatMessages()
    messages._hydrate_state(state["messages"])
    if messages.get_active_turn() is None:
        messages.resume_turn(run_id, metadata={"source": "reconciliation"})
    for call_id in call_ids:
        messages.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": (
                    "Host abandoned this run; external effects are unconfirmed."
                    if abandon
                    else results[call_id]
                ),
            }
        )
    messages.end_turn(event="interrupt" if abandon else "pause")
    state["messages"] = messages._to_state()
    state["status"] = "interrupted" if abandon else "paused"
    del extensions[PENDING_KEY]
    receipts[decision_id] = decision
    store.commit_state(
        namespace,
        thread_id,
        run_id,
        state,
        expected_revision=expected_revision,
        extension_state=extensions,
        head_item_id=messages[-1]["item_id"],
        event={
            "event_type": "approval.reconciled",
            "decision_id": decision_id,
            "decided_by": decided_by,
            "abandon": abandon,
            "tool_call_ids": call_ids,
        },
    )
    return deepcopy(decision)
