"""Read-only evidence projections for foreground run inspection."""

from msgflux.exceptions import TaskPauseRequestedError


def admission_evidence(admission, namespace, owner_id, local_worker):
    """Describe ownership without exposing identities or asserting quiescence."""
    if admission is None:
        return True, [
            "No admission exists; checkpoint import requires host-established "
            "worker quiescence."
        ]
    receipt = admission.receipt
    reasons = [receipt.error] if receipt.error else []
    requires_quiescence = (
        receipt.status in {"running", "paused", "failed"}
        and not local_worker
        and admission.owner_id != owner_id
    )
    if local_worker:
        reasons.append("A local worker is currently active.")
    elif requires_quiescence:
        reasons.append(
            "This started attempt has no local worker; establish "
            "old-worker quiescence before recovery."
        )
    if admission.namespace != namespace:
        reasons.append("The admission namespace does not match the current session.")
    return requires_quiescence, reasons


def approval_evidence(checkpoint):
    """Summarize the saved batch without returning invocation or review data."""
    pending = (
        checkpoint.get("runtime", {}).get("extensions", {}).get("pending_approvals")
    )
    if pending is None:
        return None, 0, ()
    if not isinstance(pending, dict):
        return "uncertain", 0, ("Pending approval record is malformed.",)
    if pending.get("schema_version") != 1:
        return "uncertain", 0, ("Pending approval schema is unsupported.",)
    requests = pending.get("requests")
    if not isinstance(requests, dict):
        return "uncertain", 0, ("Pending approval requests are malformed.",)
    return _approval_phase_evidence(pending.get("phase"), len(requests))


def _approval_phase_evidence(phase, count):
    if phase is None:
        if count:
            return (
                "awaiting_decision",
                count,
                ("Pending approvals await a host decision.",),
            )
        return None, 0, ()
    if not isinstance(phase, str) or not phase:
        return "uncertain", count, ("Pending approval phase is malformed.",)
    if phase in {"awaiting_decision", "awaiting-decision"}:
        reasons = ("Pending approvals await a host decision.",) if count else ()
        return phase, count, reasons
    if phase == "approved":
        return phase, count, ()
    if phase in {"executing", "uncertain"}:
        return (
            phase,
            count,
            ("Pending approval execution requires host reconciliation.",),
        )
    return "uncertain", count, ("Pending approval phase is unsupported.",)


def checkpoint_reasons(session, checkpoint, thread_id, run_id):
    """Use Agent recovery guards to retain their concrete refusal reasons."""
    reasons = []
    if checkpoint.get("status") not in {"completed", "interrupted"}:
        latest = session.checkpoint_store.load_latest_run(session.namespace, thread_id)
        if latest is not None and latest.get("scope", {}).get("run_id") != run_id:
            reasons.append("Only the latest unfinished run can resume.")
    try:
        session.agent._validate_checkpoint_command_receipts(checkpoint)
    except TaskPauseRequestedError as exc:
        reasons.append(str(exc))
    try:
        with session.context(session.scope(thread_id, run_id=run_id)):
            session.agent._validate_checkpoint_workspace(checkpoint)
    except TaskPauseRequestedError as exc:
        reasons.append(str(exc))
    return reasons
