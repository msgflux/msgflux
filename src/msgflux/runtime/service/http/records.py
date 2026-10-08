"""Versioned native JSON/SSE contracts for AgentService clients."""

from typing import Annotated, Literal

import msgspec

from msgflux.runtime.permissions import ResourcePermission
from msgflux.runtime.service.records import (
    ApprovalReview,
    EventRecord,
    RunSummary,
    ServiceThread,
    SnapshotRecord,
)

__all__ = [
    "AgentsResponse",
    "ApprovalDecisionRequest",
    "ApprovalReview",
    "ApprovalReviewsResponse",
    "ErrorResponse",
    "EventRecord",
    "HealthRecord",
    "Identifier",
    "InterruptResponse",
    "OpenThreadRequest",
    "PromptRequest",
    "ResourcePermission",
    "ResumeRequest",
    "RunSummary",
    "RunsResponse",
    "ServiceThread",
    "ShutdownRequest",
    "ShutdownResponse",
    "SnapshotRecord",
    "SteerRequest",
    "ThreadsResponse",
    "WorkspacePolicyRequest",
]

Identifier = Annotated[str, msgspec.Meta(min_length=1, max_length=512)]


class OpenThreadRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    agent_id: Identifier
    thread_id: Identifier | None = None
    cwd: str | None = None


class PromptRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    prompt: str
    request_id: Identifier


class SteerRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    content: str


class ResumeRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    pass


class ShutdownRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    expected_instance_id: Identifier


class ShutdownResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    instance_id: str
    accepted: bool = True


class ApprovalDecisionRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    approved: bool
    expected_revision: Annotated[int, msgspec.Meta(gt=0)]


class WorkspacePolicyRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    permissions: tuple[str, ...] | Literal["read-only", "full-access"] | None = None
    resources: tuple[ResourcePermission, ...] | None = None
    approval_policy: Literal["on-request", "never"] | None = None
    expected_revision: Annotated[int, msgspec.Meta(ge=0)] | None = None


class AgentsResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    agents: tuple[str, ...]


class ThreadsResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    threads: tuple[ServiceThread, ...]


class RunsResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    runs: tuple[RunSummary, ...]


class ApprovalReviewsResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    approvals: tuple[ApprovalReview, ...]


class InterruptResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    interrupted: bool


class ErrorResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    code: str
    message: str


class HealthRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    instance_id: str
    version: Literal[1] = 1
