"""Host-configured Agent approval rules and execution-local enforcement."""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import msgspec

from msgflux.exceptions import TaskPauseRequestedError
from msgflux.runtime.approvals.base import ApprovalConflictError, ApprovalStore
from msgflux.runtime.approvals.records import (
    ApprovalBinding,
    require_name,
    require_time,
)
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.events import EventType, emit_event
from msgflux.runtime.permissions import require_permissions
from msgflux.runtime.workspace_changes import PreparedFileChange
from msgflux.tools.runtime import ToolOutcome
from msgflux.tools.workspace_changes import (
    workspace_change_execution,
    workspace_change_tool,
)

_CURRENT_APPROVAL_BATCH = ContextVar("msgflux_approval_batch", default=None)


class ApprovalReconciliationRequiredError(TaskPauseRequestedError):
    """Do not overwrite a claimed batch: its worker may still be executing."""


@dataclass(frozen=True)
class AgentApprovals:
    """Approval rules for named foreground tools; versions are owned by the host."""

    store: ApprovalStore
    tools: Mapping[str, str]
    policy_version: str
    ttl_seconds: float = 300

    def __post_init__(self):
        if not isinstance(self.store, ApprovalStore):
            raise TypeError("AgentApprovals requires an ApprovalStore")
        require_name(self.policy_version)
        require_time(self.ttl_seconds)
        if self.ttl_seconds <= 0:
            raise ValueError("Approval TTL must be positive")
        if not isinstance(self.tools, Mapping) or not self.tools:
            raise ValueError(
                "Approval tools must map names to implementation revisions"
            )
        tools = {
            require_name(name): require_name(revision)
            for name, revision in self.tools.items()
        }
        object.__setattr__(self, "tools", MappingProxyType(tools))

    def binding(self, library, intent, *, arguments=None, change=None):
        scope = get_execution_scope()
        definition = library.get_tool_definition(intent.name)
        if (
            definition.dispatch.name != "foreground"
            or definition.feedback.name == "call_as_response"
        ):
            raise ValueError("Agent approvals only support executable foreground tools")
        resources = self._resource_binding(definition, scope)
        if change is not None:
            resources["prepared_change"] = change.digest
        return ApprovalBinding.from_call(
            namespace=scope.namespace,
            thread_id=scope.thread_id,
            run_id=scope.run_id,
            principal=scope.principal,
            tool_call_id=intent.id,
            tool_name=intent.name,
            tool_revision=self.tools[intent.name],
            policy_version=self.policy_version,
            arguments=intent.arguments if arguments is None else arguments,
            resources=resources,
            required_permissions=definition.required_permissions,
        )

    @staticmethod
    def _resource_binding(definition, scope):
        resources = {}
        if definition.required_resources:
            resources["required"] = [
                {"resource": item.resource, "action": item.action}
                for item in definition.required_resources
            ]
        if scope.environment is not None:
            resources["workspace_id"] = scope.environment.filesystem.workspace_id
            resources["isolation"] = sorted(scope.environment.requirements.mechanisms)
        return resources

    def prepare(self, library, intents, pending):
        records = {}
        changes = {}
        failures = {}
        for intent in intents:
            if intent.name not in self.tools:
                if intent.id in pending["requests"]:
                    raise TaskPauseRequestedError(
                        message="Approval rule removed from pending call"
                    )
                continue
            definition = library.get_tool_definition(intent.name)
            change = _prepare_review_change(definition, intent, pending)
            if isinstance(change, ToolOutcome):
                failures[intent.id] = change
                continue
            binding = self.binding(library, intent, change=change)
            require_permissions(
                binding.required_permissions,
                library.get_tool_definition(intent.name).required_resources,
            )
            key = json.dumps(
                [binding.namespace, binding.thread_id, binding.run_id, intent.id]
            )
            request_id = hashlib.sha256(key.encode()).hexdigest()
            record = self.store.get(binding.namespace, request_id)
            if record is None:
                record = self.store.request(
                    binding,
                    request_id=request_id,
                    expires_at=time.time() + self.ttl_seconds,
                )
            if record.binding != binding:
                raise TaskPauseRequestedError(
                    message="Approval binding changed; host reconciliation required"
                )
            records[intent.id] = record
            if change is not None:
                resources = self._resource_binding(definition, get_execution_scope())
                pending.setdefault("prepared_changes", {})[intent.id] = {
                    "change": msgspec.to_builtins(change),
                    "resources": resources,
                }
                changes[intent.id] = change
        return ApprovalBatch(self, library, records, changes=changes, failures=failures)


class ApprovalBatch:
    def __init__(self, policy, library, records, *, changes=None, failures=None):
        self.policy = policy
        self.library = library
        self.records = records
        self.changes = changes or {}
        self.failures = failures or {}
        self.consumed = set()

    def require_ready(self):
        if any(record.status == "consumed" for record in self.records.values()):
            raise TaskPauseRequestedError(
                message="Consumed approval needs reconciliation before retry"
            )
        if any(record.status == "pending" for record in self.records.values()):
            for record in self.records.values():
                if record.status == "pending":
                    emit_event(
                        EventType.TOOL_APPROVAL_REQUIRED,
                        {
                            "request_id": record.request_id,
                            "tool_call_id": record.binding.tool_call_id,
                            "tool_name": record.binding.tool_name,
                            "expires_at": record.expires_at,
                        },
                    )
            raise TaskPauseRequestedError(message="Agent is waiting for tool approval")

    @contextmanager
    def activate(self):
        token = _CURRENT_APPROVAL_BATCH.set(self)
        try:
            yield
        finally:
            _CURRENT_APPROVAL_BATCH.reset(token)

    def guard(self, plan, *, consume):
        if plan.intent.name not in self.policy.tools:
            return None
        if plan.intent.id in self.failures:
            return self.failures[plan.intent.id]
        try:
            record = self.records.get(plan.intent.id)
            if record is None:
                raise ApprovalConflictError("No approval for this tool call")
            if plan.dispatch.name != "foreground":
                raise ApprovalConflictError(
                    "Approved tools cannot change dispatch mode"
                )
            binding = self.policy.binding(
                self.library,
                plan.intent,
                arguments=plan.visible_arguments,
                change=prepare_workspace_change(
                    plan.definition, plan.visible_arguments
                ),
            )
            if (
                binding != record.binding
                or plan.definition != self.library.get_tool_definition(plan.intent.name)
            ):
                raise ApprovalConflictError(
                    "Final tool plan differs from the approved call"
                )
            if record.status in {"denied", "expired"}:
                raise ApprovalConflictError(f"Tool approval is {record.status}")
            if consume:
                self.policy.store.consume(record.request_id, binding=binding)
                self.consumed.add(plan.intent.id)
        except (
            ApprovalConflictError,
            PermissionError,
            ValueError,
            FileNotFoundError,
        ) as exc:
            emit_event(
                EventType.TOOL_BLOCKED,
                {
                    "tool_call_id": plan.intent.id,
                    "tool_name": plan.intent.name,
                    "code": "tool_approval_blocked",
                },
            )
            return ToolOutcome.failed(
                plan.intent,
                status="blocked",
                code="tool_approval_blocked",
                message=str(exc),
            )
        return None


def guard_approved_plan(plan: Any, *, consume: bool = False):
    batch = _CURRENT_APPROVAL_BATCH.get()
    return batch.guard(plan, consume=consume) if batch is not None else None


def approval_batch_active():
    return _CURRENT_APPROVAL_BATCH.get() is not None


def prepare_workspace_change(definition, arguments):
    tool = workspace_change_tool(definition)
    if tool is None:
        return None
    environment = get_execution_scope().environment
    if environment is None:
        raise PermissionError("Workspace changes require a live environment")
    change = tool.prepare_workspace_change(arguments, environment.filesystem)
    if not isinstance(change, PreparedFileChange):
        raise TypeError("Workspace change tools must return PreparedFileChange")
    return change


def _prepare_review_change(definition, intent, pending):
    stored = pending.get("prepared_changes", {}).get(intent.id)
    try:
        change = prepare_workspace_change(definition, intent.arguments)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        if stored is not None or intent.id in pending["requests"]:
            raise TaskPauseRequestedError(
                message="Prepared file change is invalid; host review required"
            ) from exc
        return ToolOutcome.failed(
            intent,
            status="blocked",
            code="workspace_change_invalid",
            message=f"Cannot prepare workspace change: {exc}",
        )
    if stored is not None:
        restored = msgspec.convert(stored["change"], type=PreparedFileChange)
        if restored != change:
            raise TaskPauseRequestedError(
                message="Prepared file change changed; a new review is required"
            )
    return change


@contextmanager
def approved_tool_execution(plan):
    batch = _CURRENT_APPROVAL_BATCH.get()
    change = batch.changes.get(plan.intent.id) if batch is not None else None
    if change is not None and plan.intent.id not in batch.consumed:
        raise ApprovalConflictError("Prepared change requires a consumed approval")
    with workspace_change_execution(workspace_change_tool(plan.definition), change):
        yield
