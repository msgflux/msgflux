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


def require_permissions(required: Iterable[str], resources=()) -> None:
    """Check live authority, never arguments or persisted execution identity."""
    # Context imports PermissionSet; resolve the live reader only at invocation.
    from msgflux.runtime.context import get_execution_scope  # noqa: PLC0415

    permissions = get_execution_scope().permissions or PermissionSet()
    missing = permissions.missing(required)
    if missing:
        raise PermissionError(f"Missing tool permissions: {', '.join(missing)}")
    if permissions.missing_resources(resources):
        # Resource IDs can contain private paths; do not put them in events/errors.
        raise PermissionError("Missing tool resource permissions")


__all__ = ["PermissionSet", "ResourcePermission"]
