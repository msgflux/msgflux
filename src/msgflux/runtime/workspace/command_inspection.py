"""Host inspection results and ownership checks for durable commands."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

import msgspec

from msgflux.runtime.workspace.contracts import WorkspaceIdentity
from msgflux.runtime.workspace.receipts import CommandReceipt, decode_command_receipt
from msgflux.runtime.workspace.references import WorkspaceReference


class CommandInspection(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Compact host observation of a command resource and its known outcome."""

    version: Literal[1]
    execution_id: str
    classification: Literal["running", "completed", "unknown", "blocked"]
    resource_status: Literal[
        "present", "missing", "mismatch", "unavailable", "unchecked"
    ] = "unchecked"
    returncode: int | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.execution_id, str) or not self.execution_id:
            raise ValueError("execution_id must be non-empty")
        if self.returncode is not None and type(self.returncode) is not int:
            raise TypeError("returncode must be an integer or None")
        if not isinstance(self.stdout, bytes) or not isinstance(self.stderr, bytes):
            raise TypeError("command output must be bytes")


def receipt_mapping(receipt) -> Mapping:
    """Return a safe builtins view of a receipt snapshot."""
    if isinstance(receipt, CommandReceipt):
        return msgspec.to_builtins(receipt)
    if isinstance(receipt, Mapping):
        return msgspec.to_builtins(decode_command_receipt(receipt))
    raise TypeError("receipt must be a CommandReceipt or mapping snapshot")


def receipt_identity_error(receipt, filesystem, *, backend: str) -> str | None:
    """Check receipt version, workspace identity and backend ownership."""
    record = receipt_mapping(receipt)
    if record.get("version") != 1:
        return "command receipt version is unsupported"
    execution_id = record.get("execution_id")
    if not isinstance(execution_id, str) or not execution_id:
        return "command receipt has no execution ID"
    if record.get("backend") != backend:
        return "command receipt belongs to another backend"
    try:
        reference = msgspec.convert(
            record.get("workspace_reference"), type=WorkspaceReference
        )
    except (msgspec.ValidationError, TypeError, ValueError):
        return "command receipt workspace reference is invalid"

    identity = getattr(filesystem, "identity", None)
    workspace_id = getattr(filesystem, "workspace_id", None)
    if (
        not isinstance(identity, WorkspaceIdentity)
        or reference.workspace_id != workspace_id
        or reference.identity != identity
    ):
        return "command receipt belongs to another workspace identity"
    return None


def receipt_resource(receipt) -> Mapping | None:
    try:
        resource = receipt_mapping(receipt).get("resource")
    except TypeError:
        return None
    return resource if isinstance(resource, Mapping) else None


def command_inspection(
    receipt,
    classification,
    resource_status,
    reason=None,
    *,
    stdout=None,
    stderr=None,
    returncode=None,
) -> CommandInspection:
    """Build a consistently encoded host observation from a receipt snapshot."""
    record = receipt_mapping(receipt)

    def as_bytes(value):
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        return b""

    return CommandInspection(
        version=1,
        execution_id=record["execution_id"],
        classification=classification,
        resource_status=resource_status,
        returncode=returncode,
        stdout=as_bytes(stdout),
        stderr=as_bytes(stderr),
        reasons=(reason,) if reason else (),
    )


def local_process_identity(pid: int) -> dict[str, int | str] | None:
    """Read Linux boot and process-start identity; PID alone is never enough."""
    if type(pid) is not int or pid <= 0 or not os.path.isdir("/proc"):
        return None
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
            boot_id = stream.read(128).strip()
        with open(f"/proc/{pid}/stat", encoding="ascii") as stream:
            stat = stream.read(8192)
    except (OSError, UnicodeError):
        return None
    # `comm` is parenthesized and may itself contain spaces or parentheses.
    close = stat.rfind(")")
    if close < 0:
        return None
    fields = stat[close + 1 :].split()
    if len(fields) <= 19 or not boot_id:
        return None
    try:
        start_ticks = int(fields[19])
    except ValueError:
        return None
    return {
        "pid": pid,
        "boot_id": boot_id,
        "start_ticks": start_ticks,
        "state": fields[0],
    }


def local_identity_matches(resource: Mapping) -> tuple[bool | None, str, str]:
    """Return identity match, resource status, and a human-readable reason."""
    pid = resource.get("pid")
    boot_id = resource.get("boot_id")
    start_ticks = resource.get("start_ticks")
    if (
        type(pid) is not int
        or not isinstance(boot_id, str)
        or type(start_ticks) is not int
    ):
        return (
            None,
            "unavailable",
            "local process receipt lacks boot/PID-start identity",
        )
    current = local_process_identity(pid)
    if current is None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False, "missing", "recorded local process is no longer present"
        except PermissionError:
            return None, "unavailable", "local process identity cannot be read"
        return None, "unavailable", "local process identity is unavailable on this host"
    if current["boot_id"] != boot_id or current["start_ticks"] != start_ticks:
        return False, "mismatch", "PID now identifies a different process or host boot"
    if current.get("state") in {"Z", "X"}:
        return False, "missing", "recorded local process has exited"
    matches = True
    return (
        matches,
        "present",
        (
            "recorded local process identity matches"
            if matches
            else "PID now identifies a different process or host boot"
        ),
    )


__all__ = [
    "CommandInspection",
    "command_inspection",
    "local_identity_matches",
    "local_process_identity",
    "receipt_identity_error",
    "receipt_mapping",
    "receipt_resource",
]
