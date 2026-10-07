"""Versioned native JSON/SSE contracts for AgentService clients."""

from typing import Annotated, Any, Literal

import msgspec

from msgflux.runtime.service.records import RunSummary, ServiceThread

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


class AgentsResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    agents: tuple[str, ...]


class ThreadsResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    threads: tuple[ServiceThread, ...]


class RunsResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    runs: tuple[RunSummary, ...]


class InterruptResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    interrupted: bool


class ErrorResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    code: str
    message: str


class SnapshotRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    thread_id: str
    namespace: str | None = None
    messages: tuple[dict[str, Any], ...] | None = None
    active_runs: tuple[dict[str, Any], ...] = ()
    running_tools: tuple[dict[str, Any], ...] = ()
    background_tasks: tuple[dict[str, Any], ...] = ()
    approvals: tuple[dict[str, Any], ...] = ()
    version: Literal[1] = 1


class EventRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    type: str
    timestamp: str
    data: dict[str, Any]
    run_id: str | None = None
    source_path: tuple[str, ...] = ()
    version: Literal[1] = 1


class HealthRecord(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    instance_id: str
    version: Literal[1] = 1
