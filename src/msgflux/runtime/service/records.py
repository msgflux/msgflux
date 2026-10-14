"""Serializable records for the Agent service and its portable projections."""

from typing import Any, Literal

import msgspec

from msgflux.tasks.records import TaskSummary

AdmissionStatus = Literal[
    "accepted", "running", "completed", "paused", "interrupted", "failed"
]


class AgentConfiguration(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Host-published registration defaults, never live sessions or credentials."""

    harness: str | None = None
    factory: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    permissions: tuple[str, ...] | None = None
    tools_active: tuple[str, ...] | None = None
    tools_deferred: tuple[str, ...] | None = None
    subagents: tuple[str, ...] | None = None
    preset: str | None = None
    prompt_source: Literal["default", "preset", "file"] | None = None
    prompt_file: str | None = None
    config_path: str | None = None
    config_digest: str | None = None


class AgentInspection(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Loaded service registration, without thread/run-specific overrides."""

    agent_id: str
    is_default: bool
    account_name: str
    configuration: AgentConfiguration | None = None


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
    interactive: bool = True


class AdmissionReceipt(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    thread_id: str
    request_id: str
    run_id: str
    status: AdmissionStatus
    error: str | None = None
    version: Literal[1] = 1
    revision: int = 0


class RunResult(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Settled answer from one recorded run, without events or full history."""

    thread_id: str
    run_id: str
    status: AdmissionStatus
    output: str | None = None
    error: str | None = None


class RunInspection(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Read-only recovery evidence for one foreground service run."""

    run_id: str
    receipt: AdmissionReceipt | None = None
    checkpoint_status: str | None = None
    checkpoint_revision: int | None = None
    local_worker: bool = False
    requires_quiescence: bool = False
    approval_phase: str | None = None
    approval_request_count: int = 0
    background_tasks: tuple[TaskSummary, ...] = ()
    reasons: tuple[str, ...] = ()


class AdmissionRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    receipt: AdmissionReceipt
    namespace: str
    prompt: str
    owner_id: str | None = None
    account_name: str | None = None


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
