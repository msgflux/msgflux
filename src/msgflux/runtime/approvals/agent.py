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
from msgflux.runtime.agent_run import get_agent_run
from msgflux.runtime.approvals.base import ApprovalConflictError, ApprovalStore
from msgflux.runtime.approvals.records import (
    ApprovalBinding,
    require_name,
    require_time,
)
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.events import EventType, emit_event, event_source
from msgflux.runtime.permissions import require_permissions
from msgflux.runtime.workspace.changes import PreparedFileChange
from msgflux.tools.helpers import RUNTIME_BACKGROUND_PARAM, coerce_tool_params
from msgflux.tools.runtime import ToolOutcome
from msgflux.tools.workspace_changes import (
    workspace_change_execution,
    workspace_change_tool,
)
from msgflux.utils.errors import exception_message

_CURRENT_APPROVAL_BATCH = ContextVar("msgflux_approval_batch", default=None)


class ApprovalReconciliationRequiredError(TaskPauseRequestedError):
    """Do not overwrite a claimed batch: its worker may still be executing."""


def _approval_arguments(definition, arguments, *, dispatch_mode=None):
    """Bind dispatch controls separately from the executable arguments."""
    visible_arguments = coerce_tool_params(definition.name, arguments)
    selected_dispatch = definition.dispatch.name
    if selected_dispatch == "optional_background":
        argument = definition.dispatch.options.get("argument", RUNTIME_BACKGROUND_PARAM)
        if dispatch_mode is not None and argument in visible_arguments:
            raise ValueError(
                "Runtime dispatch controls cannot be added to approved arguments"
            )
        requested = visible_arguments.pop(argument, False)
        if requested is not None and type(requested) is not bool:
            raise TypeError(f"`{argument}` must be a bool or None")
        selected_dispatch = "background" if requested is True else "foreground"
    if dispatch_mode is not None:
        selected_dispatch = dispatch_mode
    if definition.dispatch.name == "foreground" and selected_dispatch != "foreground":
        raise ValueError("Approved tools cannot change dispatch mode")
    if selected_dispatch not in {"foreground", "background"}:
        raise ValueError("Unsupported approved dispatch mode")
    if (
        selected_dispatch == "background"
        and workspace_change_tool(definition) is not None
    ):
        raise ValueError("Approved file changes must execute in foreground")
    return visible_arguments, selected_dispatch


@dataclass(frozen=True)
class AgentApprovals:
    """Approval rules for executable tools; versions are owned by the host."""

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

    def binding(
        self, library, intent, *, arguments=None, change=None, dispatch_mode=None
    ):
        scope = get_execution_scope()
        definition = library.get_tool_definition(intent.name)
        if (
            definition.dispatch.name
            not in {"foreground", "optional_background", "background"}
            or definition.feedback.name == "call_as_response"
        ):
            raise ValueError(
                "Agent approvals require executable foreground or background tools"
            )
        visible_arguments, selected_dispatch = _approval_arguments(
            definition,
            intent.arguments if arguments is None else arguments,
            dispatch_mode=dispatch_mode,
        )
        resources = self._resource_binding(definition, scope)
        if definition.dispatch.name != "foreground":
            resources["dispatch_mode"] = selected_dispatch
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
            arguments=visible_arguments,
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
        if scope.workspace is not None:
            scope.workspace.require_active()
            resources["workspace_id"] = scope.workspace.workspace_id
            resources["workspace_identity"] = msgspec.to_builtins(
                scope.workspace.identity
            )
            resources["isolation"] = sorted(scope.workspace.requirements.mechanisms)
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

    @staticmethod
    def _emit_blocked_outcome(intent, outcome):
        error = outcome.error
        with event_source(intent.name, "tool"):
            emit_event(
                EventType.TOOL_BLOCKED,
                {
                    "tool_call_id": intent.id,
                    "tool_name": intent.name,
                    "code": error.code if error is not None else "tool_blocked",
                    "error": error,
                    "reason": error.message if error is not None else "Tool blocked",
                },
            )

    def guard(self, plan, *, consume):
        if plan.intent.name not in self.policy.tools:
            return None
        if plan.intent.id in self.failures:
            outcome = self.failures[plan.intent.id]
            self._emit_blocked_outcome(plan.intent, outcome)
            return outcome
        try:
            record = self.records.get(plan.intent.id)
            if record is None:
                raise ApprovalConflictError("No approval for this tool call")
            binding = self.policy.binding(
                self.library,
                plan.intent,
                arguments=plan.visible_arguments,
                dispatch_mode=plan.dispatch.name,
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
        except (ApprovalConflictError, ValueError, OSError) as exc:
            outcome = ToolOutcome.failed(
                plan.intent,
                status="blocked",
                code="tool_approval_blocked",
                message=exception_message(exc),
                details={"exception_type": type(exc).__name__},
            )
            self._emit_blocked_outcome(plan.intent, outcome)
            return outcome
        return None


def guard_approved_plan(plan: Any, *, consume: bool = False):
    outcome = _retired_tool_call_outcome(plan.intent)
    if outcome is not None:
        return outcome
    batch = _CURRENT_APPROVAL_BATCH.get()
    return batch.guard(plan, consume=consume) if batch is not None else None


def _retired_tool_call_outcome(intent):
    from msgflux.runtime.approvals.reconciliation import (  # noqa: PLC0415
        _retired_tool_call_ids,
    )

    run = get_agent_run()
    if run is None:
        return None
    retired_ids, fail_closed = _retired_tool_call_ids(run.extension_state)
    if not fail_closed and intent.id not in retired_ids:
        return None
    outcome = ToolOutcome.failed(
        intent,
        status="blocked",
        code=(
            "tool_approval_state_invalid" if fail_closed else "tool_approval_retired"
        ),
        message=(
            "Tool execution is blocked because approval retirement state "
            "is malformed or ambiguous"
            if fail_closed
            else "Tool call ID was retired by host reconciliation"
        ),
    )
    ApprovalBatch._emit_blocked_outcome(intent, outcome)
    return outcome


def approval_batch_active():
    return _CURRENT_APPROVAL_BATCH.get() is not None


def workspace_change_approved(change: PreparedFileChange) -> bool:
    """Check whether a consumed approval in this batch bound this exact preview."""
    if not isinstance(change, PreparedFileChange):
        return False
    batch = _CURRENT_APPROVAL_BATCH.get()
    if batch is None:
        return False
    return any(batch.changes.get(intent_id) == change for intent_id in batch.consumed)


def prepare_workspace_change(definition, arguments):
    tool = workspace_change_tool(definition)
    if tool is None:
        return None
    workspace = get_execution_scope().workspace
    if workspace is None:
        raise PermissionError("Workspace changes require a live workspace")
    workspace.require_active()
    change = tool.prepare_workspace_change(arguments, workspace)
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
            message=(f"Cannot prepare workspace change: {exception_message(exc)}"),
            details={"exception_type": type(exc).__name__},
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
