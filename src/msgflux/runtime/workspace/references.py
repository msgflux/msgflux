"""Small, authority-free workspace references for durable Agent metadata."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import msgspec

from msgflux.runtime.workspace.contracts import WorkspaceIdentity


class WorkspaceReference(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Versioned checkpoint reference with no live authority."""

    version: Literal[1]
    workspace_id: str
    identity: WorkspaceIdentity
    cwd: str


class WorkspaceCwdMismatchError(ValueError):
    """The same resource was supplied with a different execution cwd."""


def encode_workspace_reference(workspace) -> dict | None:
    """Return the v1 safe reference for a live workspace, if configured."""
    if workspace is None:
        return None
    identity = workspace.identity
    if not isinstance(identity, WorkspaceIdentity):
        raise TypeError("workspace identity must be a WorkspaceIdentity")
    record = WorkspaceReference(
        version=1,
        workspace_id=workspace.workspace_id,
        identity=identity,
        cwd=workspace.cwd,
    )
    return msgspec.to_builtins(record)


def validate_workspace_reference(
    reference, workspace, *, match_cwd: bool = True
) -> None:
    """Require a live workspace to match a durable reference; legacy is allowed."""
    if reference is None:
        return
    if not isinstance(reference, Mapping):
        raise ValueError("Checkpoint has an invalid workspace reference")
    try:
        reference = msgspec.convert(reference, type=WorkspaceReference)
    except (msgspec.ValidationError, TypeError) as error:
        raise ValueError("Checkpoint has an unsupported workspace reference") from error
    expected = encode_workspace_reference(workspace)
    if expected is None:
        raise ValueError("Checkpoint requires a compatible live workspace")
    actual = msgspec.to_builtins(reference)
    if any(
        actual.get(field) != expected.get(field)
        for field in ("version", "workspace_id", "identity")
    ):
        raise ValueError("Checkpoint workspace reference does not match live workspace")
    if match_cwd and actual.get("cwd") != expected.get("cwd"):
        raise WorkspaceCwdMismatchError(
            "Checkpoint workspace cwd does not match the live workspace"
        )
