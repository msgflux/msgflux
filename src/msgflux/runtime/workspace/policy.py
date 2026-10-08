"""Durable runtime policy snapshots and their trusted live binding."""

from __future__ import annotations

from threading import RLock
from typing import Literal

import msgspec

from msgflux.runtime.permissions import (
    PermissionSet,
    ResourcePermission,
    intersect_permissions,
    normalize_permissions,
    normalize_resources,
)


class WorkspacePolicy(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Serializable, host-validated policy selected for one durable thread."""

    thread_id: str
    permissions: tuple[str, ...]
    resources: tuple[ResourcePermission, ...] = ()
    approval_policy: Literal["on-request", "never"] = "never"
    revision: int = 0
    updated_at: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.thread_id, str) or not self.thread_id.strip():
            raise ValueError("thread_id must be non-empty text")
        try:
            permissions = normalize_permissions(self.permissions)
            resources = normalize_resources(self.resources)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid workspace policy grants: {error}") from error
        msgspec.structs.force_setattr(self, "permissions", permissions)
        msgspec.structs.force_setattr(self, "resources", resources)
        if self.approval_policy not in ("on-request", "never"):
            raise ValueError("approval_policy must be 'on-request' or 'never'")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision must be a non-negative integer")
        if self.updated_at is not None and (
            not isinstance(self.updated_at, str) or not self.updated_at.strip()
        ):
            raise ValueError("updated_at must be non-empty text or None")

    def permission_set(self) -> PermissionSet:
        """Return this snapshot's normalized grants as an immutable set."""
        return PermissionSet(frozenset(self.permissions), frozenset(self.resources))


class WorkspacePolicyState:
    """Trusted shared pointer to the current policy for one live thread.

    The service persists a proposed policy before replacing ``current``. Calls
    hold no stale permission snapshot: they ask this state for the current
    immutable value whenever execution scope is read.
    """

    def __init__(self, workspace, current: WorkspacePolicy) -> None:
        from msgflux.runtime.workspace.api import AgentWorkspace  # noqa: PLC0415

        if workspace is not None and not isinstance(workspace, AgentWorkspace):
            raise TypeError("workspace must be AgentWorkspace or None")
        if not isinstance(current, WorkspacePolicy):
            raise TypeError("current must be WorkspacePolicy")
        self.workspace = workspace
        self._current = current
        self._lock = RLock()

    @property
    def current(self) -> WorkspacePolicy:
        with self._lock:
            return self._current

    @current.setter
    def current(self, policy: WorkspacePolicy) -> None:
        if not isinstance(policy, WorkspacePolicy):
            raise TypeError("current must be WorkspacePolicy")
        with self._lock:
            if policy.thread_id != self._current.thread_id:
                raise ValueError("Workspace policy cannot change thread identity")
            if policy.revision <= self._current.revision:
                raise ValueError("Workspace policy revision must increase")
            self._current = policy

    def effective_permissions(self, scope) -> PermissionSet:
        """Clip a raw call ceiling against the latest policy and workspace."""
        policy = self.current
        if scope.thread_id != policy.thread_id:
            raise ValueError("Workspace policy does not belong to this thread")
        workspace = scope.workspace
        if self.workspace is not None:
            if workspace is None or not workspace.shares_environment(self.workspace):
                raise PermissionError("Execution is not bound to the policy workspace")
        elif workspace is not None:
            raise PermissionError(
                "Policy without a workspace cannot grant workspace access"
            )
        ceiling = scope.permissions
        if ceiling is None:
            ceiling = (
                workspace.permissions if workspace is not None else PermissionSet()
            )
        requested = policy.permission_set()
        if workspace is None:
            selected = intersect_permissions(ceiling, requested)
        else:
            # A workspace mode governs workspace filesystem/process authority.
            # Trusted host capabilities and exact resources for independent
            # services remain governed by the immutable per-call scope ceiling.
            workspace_grants = frozenset(
                grant
                for grant in ceiling.grants
                if grant.startswith(("filesystem.", "process."))
            )
            policy_grants = frozenset(
                grant
                for grant in requested.grants
                if grant.startswith(("filesystem.", "process."))
            )
            prefix = f"workspace:{workspace.workspace_id}:"
            workspace_resources = frozenset(
                resource
                for resource in ceiling.resources
                if resource.resource.startswith(prefix)
                and resource.action.startswith(("filesystem.", "process."))
            )
            policy_resources = frozenset(
                resource
                for resource in requested.resources
                if resource.resource.startswith(prefix)
                and resource.action.startswith(("filesystem.", "process."))
            )
            clipped = intersect_permissions(
                PermissionSet(workspace_grants, workspace_resources),
                PermissionSet(policy_grants, policy_resources),
                workspace_id=workspace.workspace_id,
            )
            independent_grants = ceiling.grants - workspace_grants
            independent_resources = frozenset(
                resource
                for resource in ceiling.resources
                if not resource.resource.startswith("workspace:")
            )
            selected = PermissionSet(
                independent_grants | clipped.grants,
                independent_resources | clipped.resources,
            )
        if workspace is not None:
            selected = workspace.effective_permissions(selected)
        return selected


__all__ = ["WorkspacePolicy", "WorkspacePolicyState"]
