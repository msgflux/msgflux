"""Serializable records for the Agent service and its portable projections."""

from typing import Any, Literal

import msgspec

AdmissionStatus = Literal[
    "accepted", "running", "completed", "paused", "interrupted", "failed"
]


class RunSummary(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Public metadata for a saved execution, without checkpoint state."""

    run_id: str
    status: str
    updated_at: float | None = None


class ApprovalReview(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Safe review metadata for one host-authorized approval request."""

    request_id: str
    tool_call_id: str
    tool_name: str
    status: str
    revision: int
    expires_at: float
    diff: str | None = None


class ServiceConflictError(RuntimeError):
    """A request identity, thread binding, or ownership conflicts with stored state."""


class ServiceBusyError(RuntimeError):
    """A thread already has work that must settle or be resumed."""


class ServiceRecoveryRequiredError(RuntimeError):
    """An old execution cannot be safely dispatched without host reconciliation."""


class ServiceThread(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    thread_id: str
    agent_id: str
    cwd: str | None = None


class AdmissionReceipt(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    thread_id: str
    request_id: str
    run_id: str
    status: AdmissionStatus
    error: str | None = None
    version: Literal[1] = 1
    revision: int = 0


class AdmissionRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    receipt: AdmissionReceipt
    namespace: str
    prompt: str
    owner_id: str | None = None


class SnapshotRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Portable projection of a thread's durable and live presentation state."""

    thread_id: str
    namespace: str | None = None
    messages: tuple[dict[str, Any], ...] | None = None
    active_runs: tuple[dict[str, Any], ...] = ()
    running_tools: tuple[dict[str, Any], ...] = ()
    background_tasks: tuple[dict[str, Any], ...] = ()
    approvals: tuple[dict[str, Any], ...] = ()
    version: Literal[1] = 1


class EventRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Portable event projection shared by local and remote observers."""

    type: str
    timestamp: str
    data: dict[str, Any]
    run_id: str | None = None
    source_path: tuple[str, ...] = ()
    version: Literal[1] = 1
