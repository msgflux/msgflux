"""Serializable identities and admission states for the embedded Agent service."""

from typing import Literal

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
