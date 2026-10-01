"""Read-only inspection and explicit recovery for durable background Agents."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

import msgspec

from msgflux.runtime.context import (
    execution_context,
    get_execution_context,
    get_execution_scope,
)
from msgflux.runtime.permissions import require_permissions

if TYPE_CHECKING:
    from msgflux.nn.modules.tool import ToolLibrary


RecoveryClassification = Literal[
    "completed", "terminal_result", "active", "recoverable", "uncertain", "blocked"
]


class TaskRecoveryReport(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Compact, versioned snapshot of one background Agent's recoverability."""

    version: Literal[1]
    task_id: str
    classification: RecoveryClassification
    task_status: str
    task_updated_at: str
    tool_name: str | None = None
    checkpoint_namespace: str | None = None
    checkpoint_thread_id: str | None = None
    checkpoint_run_id: str | None = None
    workspace_id: str | None = None
    checkpoint_status: str | None = None
    checkpoint_revision: int | None = None
    approval_phase: str | None = None
    lease_owner_id: str | None = None
    lease_expires_at: float | None = None
    workspace_status: str = "unchecked"
    inbox_status: str = "unchecked"
    permissions_status: str = "unchecked"
    reasons: tuple[str, ...] = ()


class AgentTaskRecovery:
    """Host-facing coordinator over the library's existing task/runtime stores.

    Inspection only reads the task, lease, checkpoint, and current bindings. It
    does not create a worker, claim a lease, or invoke an Agent/tool.
    """

    def __init__(self, library: ToolLibrary, *, workspace=None):
        self.library = library
        self.workspace = workspace

    def _host_scope(self):
        if self.workspace is None:
            return nullcontext()
        return execution_context(
            scope=replace(get_execution_scope(), workspace=self.workspace)
        )

    def inspect(self, task_id: str) -> TaskRecoveryReport:  # noqa: C901
        """Return a compact read-only status report for a durable Agent task."""
        handle = self.library.get_handle()
        task_store = self.library.get_task_store()
        task = task_store.get(task_id)
        if task is None:
            return TaskRecoveryReport(
                version=1,
                task_id=task_id,
                classification="blocked",
                task_status="missing",
                task_updated_at="",
                reasons=("task record was not found",),
            )
        if task.metadata.get("task_kind") != "agent":
            return TaskRecoveryReport(
                version=1,
                task_id=task_id,
                classification="blocked",
                task_status=task.status,
                task_updated_at=task.updated_at,
                tool_name=task.tool_name,
                reasons=("task is not a background Agent",),
            )

        reasons: list[str] = []
        dispatcher = self.library.get_background_dispatcher()
        checkpoint_status = None
        checkpoint_revision = None
        checkpoint = None
        approval_phase = None
        workspace_status = "unchecked"
        inbox_status = "unchecked"
        permissions_status = "unchecked"
        uncertain = False
        incompatible = False
        try:
            tool = handle.get_tool(task.tool_name)
            definition = handle.get_tool_definition(task.tool_name)
            resume_params = task.metadata.get("task_resume_params") or {}
            checkpoint_store = dispatcher._effective_checkpoint_store(
                tool=tool, resume_params=resume_params
            )
            dispatcher._validate_checkpoint_binding(task, checkpoint_store)
            namespace = task.metadata.get("checkpoint_namespace")
            thread_id = task.metadata.get("checkpoint_thread_id")
            run_id = task.metadata.get("checkpoint_run_id")
            if checkpoint_store is not None and all(
                isinstance(item, str) and item
                for item in (namespace, thread_id, run_id)
            ):
                checkpoint = checkpoint_store.load_state(namespace, thread_id, run_id)
                if checkpoint is not None:
                    checkpoint_status = checkpoint.get("status")
                    envelope = checkpoint.get("_checkpoint", {})
                    checkpoint_revision = envelope.get("revision")
                    extensions = checkpoint.get("runtime", {}).get("extensions", {})
                    pending = extensions.get("pending_approvals") or checkpoint.get(
                        "pending_approvals"
                    )
                    if isinstance(pending, dict):
                        if pending.get("schema_version") != 1:
                            uncertain = True
                            reasons.append("pending approval schema is unsupported")
                        else:
                            approval_phase = pending.get("phase")
                            if approval_phase is None:
                                approval_phase = "awaiting_decision"
                            if approval_phase not in {
                                "awaiting_decision",
                                "awaiting-decision",
                                "approved",
                            }:
                                uncertain = True
                                reasons.append(
                                    "approval execution outcome requires host "
                                    "reconciliation"
                                )
                    elif pending is not None:
                        uncertain = True
                        reasons.append("pending approval record is malformed")
            else:
                incompatible = True
                reasons.append("checkpoint store or route is unavailable")

            with self._host_scope():
                try:
                    dispatcher.validate_task_workspace(
                        task, tool=tool, resume_params=resume_params
                    )
                except Exception as exc:
                    workspace_status = "unverified"
                    incompatible = True
                    reasons.append(f"workspace incompatible: {exc}")
                else:
                    workspace_status = (
                        "compatible"
                        if task.metadata.get("workspace_reference") is not None
                        else "not_required"
                    )

                try:
                    require_permissions(
                        definition.required_permissions,
                        definition.required_resources,
                    )
                except Exception as exc:
                    permissions_status = "insufficient"
                    incompatible = True
                    reasons.append(f"required grants unavailable: {exc}")
                else:
                    permissions_status = "sufficient"

            inbox = get_execution_context().get("agent_inbox")
            if inbox is None:
                inbox = self.library._agent_inbox
            if inbox is None:
                inbox_status = "missing"
                incompatible = True
                reasons.append("Agent inbox binding is unavailable")
            else:
                try:
                    dispatcher._resolve_task_inbox(task, agent_inbox=inbox)
                except Exception as exc:
                    inbox_status = "incompatible"
                    incompatible = True
                    reasons.append(f"inbox incompatible: {exc}")
                else:
                    inbox_status = "compatible"

        except Exception as exc:
            incompatible = True
            reasons.append(f"runtime dependency unavailable: {exc}")

        lease = task_store.get_worker_lease(task_id)
        lease_active = lease is not None and lease.expires_at > time.time()
        lease_expired = lease is not None and lease.expires_at <= time.time()
        if lease is not None and lease_active:
            classification: RecoveryClassification = "active"
        elif incompatible:
            classification = "blocked"
        elif task.status in {"completed", "failed", "interrupted", "cancelled"}:
            classification = "completed" if task.status == "completed" else "blocked"
        elif checkpoint_status == "completed" and task.status == "running":
            result = checkpoint.get("task_result") if checkpoint is not None else None
            if isinstance(result, dict) and "value" in result:
                classification = "terminal_result"
            else:
                classification = "blocked"
                reasons.append("completed checkpoint has no recorded task result")
        elif uncertain:
            classification = "uncertain"
        elif (
            task.status == "queued"
            and not lease_active
            and isinstance(task.metadata.get("initial_call_params"), dict)
        ) or (
            task.status == "running"
            and lease_expired
            and (
                checkpoint_status is not None
                or isinstance(task.metadata.get("initial_call_params"), dict)
            )
        ):
            classification = "recoverable"
        else:
            classification = "blocked"
            if task.status == "running" and lease is None:
                reasons.append(
                    "worker lease is missing; recovery cannot claim ownership"
                )
            else:
                reasons.append("durable checkpoint or initial input is unavailable")

        return TaskRecoveryReport(
            version=1,
            task_id=task_id,
            classification=classification,
            task_status=task.status,
            task_updated_at=task.updated_at,
            tool_name=task.tool_name,
            checkpoint_namespace=task.metadata.get("checkpoint_namespace"),
            checkpoint_thread_id=task.metadata.get("checkpoint_thread_id"),
            checkpoint_run_id=task.metadata.get("checkpoint_run_id"),
            workspace_id=(
                task.metadata.get("workspace_reference", {}).get("workspace_id")
                if isinstance(task.metadata.get("workspace_reference"), dict)
                else None
            ),
            checkpoint_status=checkpoint_status,
            checkpoint_revision=checkpoint_revision,
            approval_phase=approval_phase,
            lease_owner_id=lease.owner_id if lease else None,
            lease_expires_at=lease.expires_at if lease else None,
            workspace_status=workspace_status,
            inbox_status=inbox_status,
            permissions_status=permissions_status,
            reasons=tuple(reasons),
        )

    def recover(
        self,
        task_id: str,
        message: str,
        *,
        worker_stopped: bool = False,
    ) -> str:
        """Reconcile a committed result or explicitly resume a stopped worker."""
        with self._host_scope():
            report = self.inspect(task_id)
        if report.classification == "terminal_result":
            with self._host_scope():
                return self.library.reconcile_agent_task(task_id)
        if report.classification == "uncertain":
            raise RuntimeError(
                "Task execution is uncertain; reconcile the pending approval with "
                "the host approval API before recovery."
            )
        if report.classification == "active":
            raise RuntimeError(
                "Task has an active worker lease and cannot be recovered."
            )
        if report.classification != "recoverable":
            detail = "; ".join(report.reasons) or report.classification
            raise RuntimeError(f"Task is not recoverable: {detail}")
        if worker_stopped is not True:
            raise RuntimeError(
                "Confirm that the previous worker has stopped before recovering "
                "this task."
            )

        task = self.library.get_task_store().get(task_id)
        dispatcher = self.library.get_background_dispatcher()
        tool = self.library.get_handle().get_tool(task.tool_name)
        with (
            self._host_scope(),
            execution_context(
                task_store=self.library.get_task_store(),
                agent_inbox=self.library.get_agent_inbox(),
            ),
        ):
            dispatcher.validate_task_workspace(
                task,
                tool=tool,
                resume_params=task.metadata.get("task_resume_params") or {},
            )
            # Dispatcher re-reads state after claiming the lease and enforces the
            # current tool's permissions before submitting execution.
            return dispatcher.resume_agent_task(
                task=task,
                message=message,
                recover_expired=True,
                worker_stopped=True,
            )
