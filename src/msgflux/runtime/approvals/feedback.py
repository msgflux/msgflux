"""Carry reviewer comments into canonical context without changing tool outputs."""

from collections.abc import Mapping

from msgflux.chat_messages import ChatMessages
from msgflux.runtime.approvals.records import normalize_feedback, require_name
from msgflux.utils.msgspec import msgspec_dumps

REVIEW_FEEDBACK_KEY = "review_feedback"
_DELIVERED_KEY = "review_feedback_delivered"


def capture_review_feedback(records):
    """Snapshot immutable host decisions before a protected batch executes."""
    return [
        {
            "request_id": record.request_id,
            "tool_call_id": record.binding.tool_call_id,
            "tool_name": record.binding.tool_name,
            "approved": record.status != "denied",
            "status": record.status,
            "decided_by": record.decided_by,
            "feedback": record.feedback,
        }
        for record in records
        if record.feedback is not None
        and record.decided_by is not None
        and record.status in {"approved", "denied", "consumed", "expired"}
    ]


def _review_snapshot(value):
    if not isinstance(value, Mapping):
        raise ValueError("Approval feedback snapshot must be a mapping")
    review = {
        name: require_name(value.get(name))
        for name in ("request_id", "tool_call_id", "tool_name", "decided_by")
    }
    approved = value.get("approved")
    if type(approved) is not bool:
        raise ValueError("Approval feedback snapshot requires a boolean decision")
    feedback = normalize_feedback(value.get("feedback"))
    if feedback is None:
        raise ValueError("Approval feedback snapshot is missing its comment")
    status = value.get("status", "approved" if approved else "denied")
    if status not in {"approved", "denied", "consumed", "expired"}:
        raise ValueError("Invalid approval feedback decision status")
    if (status == "denied") == approved:
        raise ValueError("Approval feedback decision and status disagree")
    return {**review, "approved": approved, "status": status, "feedback": feedback}


def append_review_feedback(messages: ChatMessages, pending: dict) -> None:
    """Append after tool results; checkpoint the message and receipt together.

    The receipt belongs to the existing pending batch. A partial batch control
    may checkpoint its settled results before host reconciliation; the latter
    reuses this function and receipt instead of repeating the human comment.
    """
    delivered = pending.get(_DELIVERED_KEY, False)
    if type(delivered) is not bool:
        raise ValueError("Approval feedback delivery receipt must be a bool")
    if delivered:
        return
    snapshots = pending.get(REVIEW_FEEDBACK_KEY, [])
    if not isinstance(snapshots, list):
        raise ValueError("Approval feedback snapshots must be a list")
    reviews = [_review_snapshot(value) for value in snapshots]
    if not reviews:
        return
    messages.append(
        {
            "role": "user",
            "content": (
                "Approval reviewer feedback for the preceding tool calls:\n"
                + msgspec_dumps({"reviews": reviews})
            ),
            "metadata": {"approval_feedback": {"reviews": reviews}},
        }
    )
    pending[_DELIVERED_KEY] = True
