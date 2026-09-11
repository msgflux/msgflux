"""Immutable, argument-free records for host-operated approval journals."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping

from msgflux.runtime.permissions import normalize_permissions


def require_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Approval identifiers must be non-empty strings")
    return value


def require_time(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Approval timestamps must be finite Unix seconds")
    if not math.isfinite(value):
        raise ValueError("Approval timestamps must be finite Unix seconds")
    return value


def _validate_json(value: Any) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Approval argument keys must be strings")
            _validate_json(item)
    elif type(value) is list:
        for item in value:
            _validate_json(item)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise TypeError("Approval bindings accept JSON values only")


def _digest(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping):
        raise TypeError("Approval arguments and resources must be mappings")
    payload = dict(value)
    _validate_json(payload)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, kw_only=True)
class ApprovalBinding:
    """Exact invocation identity, with digests instead of argument contents."""

    namespace: str
    thread_id: str
    run_id: str
    principal: str
    tool_call_id: str
    tool_name: str
    tool_revision: str
    policy_version: str
    arguments_digest: str
    resources_digest: str
    required_permissions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for value in (
            self.namespace,
            self.thread_id,
            self.run_id,
            self.principal,
            self.tool_call_id,
            self.tool_name,
            self.tool_revision,
            self.policy_version,
        ):
            require_name(value)
        for value in (self.arguments_digest, self.resources_digest):
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Approval digests must be lowercase SHA-256 hex")
        object.__setattr__(
            self,
            "required_permissions",
            normalize_permissions(self.required_permissions),
        )

    @classmethod
    def from_call(
        cls,
        *,
        arguments: Mapping[str, Any],
        resources: Mapping[str, Any],
        namespace: str,
        thread_id: str,
        run_id: str,
        principal: str,
        tool_call_id: str,
        tool_name: str,
        tool_revision: str,
        policy_version: str,
        required_permissions: tuple[str, ...] = (),
    ) -> ApprovalBinding:
        return cls(
            namespace=namespace,
            thread_id=thread_id,
            run_id=run_id,
            principal=principal,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_revision=tool_revision,
            policy_version=policy_version,
            arguments_digest=_digest(arguments),
            resources_digest=_digest(resources),
            required_permissions=required_permissions,
        )


@dataclass(frozen=True, kw_only=True)
class ApprovalRecord:
    request_id: str
    binding: ApprovalBinding
    expires_at: float
    created_at: float
    updated_at: float
    status: str = "pending"
    revision: int = 1
    decided_by: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        require_name(self.request_id)
        if not isinstance(self.binding, ApprovalBinding):
            raise TypeError("Approval records require an ApprovalBinding")
        for value in (self.expires_at, self.created_at, self.updated_at):
            require_time(value)
        if self.expires_at <= self.created_at or self.updated_at < self.created_at:
            raise ValueError("Invalid approval lifetime")
        if self.status not in {"pending", "approved", "denied", "consumed", "expired"}:
            raise ValueError("Invalid approval status")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("Approval revisions must be positive integers")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported approval schema version")
        if self.decided_by is not None:
            require_name(self.decided_by)
        if (
            self.status in {"approved", "denied", "consumed"}
            and self.decided_by is None
        ):
            raise ValueError("A decided approval must identify its reviewer")


@dataclass(frozen=True, kw_only=True)
class ApprovalEvent:
    namespace: str
    request_id: str
    revision: int
    status: str
    timestamp: float
    decided_by: str | None

    @classmethod
    def from_record(cls, record: ApprovalRecord) -> ApprovalEvent:
        return cls(
            namespace=record.binding.namespace,
            request_id=record.request_id,
            revision=record.revision,
            status=record.status,
            timestamp=record.updated_at,
            decided_by=record.decided_by,
        )
