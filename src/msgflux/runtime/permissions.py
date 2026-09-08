"""Live capability grants; never infer authority from serialized model data."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

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
class PermissionSet:
    """Immutable grants. Intersection is the only delegation operation."""

    grants: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "grants", frozenset(normalize_permissions(self.grants))
        )

    def intersect(self, delegated: PermissionSet) -> PermissionSet:
        if not isinstance(delegated, PermissionSet):
            raise TypeError("Delegated grants must be a PermissionSet")
        return PermissionSet(self.grants & delegated.grants)

    def missing(self, required: Iterable[str]) -> tuple[str, ...]:
        return tuple(sorted(set(normalize_permissions(required)) - self.grants))

    def allows(self, required: Iterable[str]) -> bool:
        return not self.missing(required)


def require_permissions(required: Iterable[str]) -> None:
    """Check live authority, never arguments or persisted execution identity."""
    # Context imports PermissionSet; resolve the live reader only at invocation.
    from msgflux.runtime.context import get_execution_scope  # noqa: PLC0415

    permissions = get_execution_scope().permissions or PermissionSet()
    missing = permissions.missing(required)
    if missing:
        raise PermissionError(f"Missing tool permissions: {', '.join(missing)}")


__all__ = ["PermissionSet"]
