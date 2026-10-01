"""Live capability grants; never infer authority from serialized model data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping

_CAPABILITY = re.compile(r"[a-z][a-z0-9_]*(?:[.:][a-z][a-z0-9_]*)*\Z")


def normalize_permissions(values: Iterable[str]) -> tuple[str, ...]:
    """Validate exact capability names, excluding wildcard/resource patterns."""
    if isinstance(values, (str, bytes, dict)):
        raise TypeError("Permissions must be a collection of capability names")
    names = tuple(values)
    if any(
        not isinstance(name, str) or not _CAPABILITY.fullmatch(name) for name in names
    ):
        raise ValueError("Permissions must contain exact capability names")
    return tuple(sorted(set(names)))


@dataclass(frozen=True)
class ResourcePermission:
    """One exact host-owned resource ID and operation; no wildcard semantics."""

    resource: str
    action: str

    def __post_init__(self):
        if not isinstance(self.resource, str) or not self.resource.strip():
            raise ValueError("Resource ID must be a non-empty string")
        normalize_permissions((self.action,))


def normalize_resources(values) -> tuple[ResourcePermission, ...]:
    if isinstance(values, (str, bytes, Mapping)):
        raise TypeError("Resources must be a collection of resource permissions")
    result = []
    for value in values:
        item = ResourcePermission(**value) if isinstance(value, Mapping) else value
        if not isinstance(item, ResourcePermission):
            raise TypeError("Expected ResourcePermission or its serialized mapping")
        result.append(item)
    return tuple(sorted(set(result), key=lambda item: (item.resource, item.action)))


@dataclass(frozen=True)
class PermissionSet:
    """Immutable grants. Intersection is the only delegation operation."""

    grants: frozenset[str] = field(default_factory=frozenset)
    resources: frozenset[ResourcePermission] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "grants", frozenset(normalize_permissions(self.grants))
        )
        object.__setattr__(
            self, "resources", frozenset(normalize_resources(self.resources))
        )

    def intersect(self, delegated: PermissionSet) -> PermissionSet:
        if not isinstance(delegated, PermissionSet):
            raise TypeError("Delegated grants must be a PermissionSet")
        return PermissionSet(
            self.grants & delegated.grants, self.resources & delegated.resources
        )

    def missing_resources(self, required) -> tuple[ResourcePermission, ...]:
        return tuple(
            item for item in normalize_resources(required) if item not in self.resources
        )

    def missing(self, required: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(set(normalize_permissions(required)) - self.grants))

    def allows(self, required: Iterable[str]) -> bool:
        return not self.missing(required)


def intersect_permissions(
    parent: PermissionSet,
    child: PermissionSet,
    *,
    workspace_id: str | None = None,
) -> PermissionSet:
    """Intersect delegated grants, allowing broad/exact workspace narrowing.

    Exact resource grants remain exact. For resources belonging to the active
    workspace, a broad operation grant on either side can admit the exact
    resource selected by the other side. Resource IDs outside that workspace
    continue to require an exact grant on both sides.
    """
    if not isinstance(parent, PermissionSet) or not isinstance(child, PermissionSet):
        raise TypeError("Permission intersection requires PermissionSet values")

    if workspace_id is None:
        resources = set(parent.resources & child.resources)
    else:
        if not isinstance(workspace_id, str) or not workspace_id:
            raise ValueError("workspace_id must be non-empty text or None")
        prefix = f"workspace:{workspace_id}:"
        resources = {
            item
            for item in parent.resources & child.resources
            if not item.resource.startswith("workspace:")
            or item.resource.startswith(prefix)
        }
        for item in child.resources:
            if (
                item.resource.startswith(prefix)
                and item.action.startswith(("filesystem.", "process."))
                and item.action in parent.grants
            ):
                resources.add(item)
        for item in parent.resources:
            if (
                item.resource.startswith(prefix)
                and item.action.startswith(("filesystem.", "process."))
                and item.action in child.grants
            ):
                resources.add(item)
    return PermissionSet(parent.grants & child.grants, frozenset(resources))


def require_permissions(required: Iterable[str], resources=()) -> None:
    """Check live authority, never arguments or persisted execution identity."""
    # Context imports PermissionSet; resolve the live reader only at invocation.
    from msgflux.runtime.context import get_execution_scope  # noqa: PLC0415

    scope = get_execution_scope()
    permissions = scope.permissions or PermissionSet()
    missing = permissions.missing(required)
    if missing:
        raise PermissionError(f"Missing tool permissions: {', '.join(missing)}")
    missing_resources = []
    workspace = scope.workspace
    for resource in normalize_resources(resources):
        if resource in permissions.resources:
            continue
        # Workspace-wide grants admit the backend's exact path check only for
        # the currently bound workspace. Other resource identities stay exact.
        workspace_prefix = (
            f"workspace:{workspace.workspace_id}:" if workspace is not None else None
        )
        if (
            workspace_prefix is not None
            and resource.resource.startswith(workspace_prefix)
            and resource.action.startswith(("filesystem.", "process."))
            and permissions.allows((resource.action,))
        ):
            continue
        missing_resources.append(resource)
    if missing_resources:
        # Resource IDs can contain private paths; do not put them in events/errors.
        raise PermissionError("Missing tool resource permissions")


__all__ = ["PermissionSet", "ResourcePermission", "intersect_permissions"]
