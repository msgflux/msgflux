"""Bounded durable receipts for workspace command executions."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

import msgspec

ReceiptState = Literal[
    "intent", "launched", "completed", "failed", "terminated", "unknown", "reconciled"
]

MAX_RECEIPT_OUTPUT_BYTES = 8192
MAX_TERMINAL_RECEIPTS = 64


class CommandReceipt(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Versioned evidence for one command; it contains no command arguments."""

    version: Literal[1]
    execution_id: str
    state: ReceiptState
    workspace_reference: dict[str, Any] | None
    backend: str
    created_at: str
    updated_at: str
    owner_id: str | None = None
    run_id: str | None = None
    task_id: str | None = None
    tool_call_id: str | None = None
    message_offset: int | None = None
    resource: dict[str, Any] | None = None
    returncode: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    tool_output_recorded: bool = False

    def __post_init__(self) -> None:
        if not self.execution_id or not isinstance(self.execution_id, str):
            raise ValueError("execution_id must be a non-empty string")
        if not self.backend or not isinstance(self.backend, str):
            raise ValueError("backend must be a non-empty string")
        if self.returncode is not None and type(self.returncode) is not int:
            raise TypeError("returncode must be an integer or None")
        if type(self.tool_output_recorded) is not bool:
            raise TypeError("tool_output_recorded must be a boolean")
        if self.message_offset is not None and (
            type(self.message_offset) is not int or self.message_offset < 0
        ):
            raise ValueError("message_offset must be a non-negative integer or None")
        for name in ("workspace_reference", "resource"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping or None")
            if value is not None:
                msgspec.structs.force_setattr(self, name, copy.deepcopy(dict(value)))

    @property
    def unresolved(self) -> bool:
        """Whether recovery still needs host action before the Agent can resume."""
        terminal = self.state in {"completed", "failed", "terminated", "reconciled"}
        return not (
            terminal and (self.tool_output_recorded or self.state == "reconciled")
        )

    def to_dict(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)


ReceiptPersist = Callable[[CommandReceipt], Awaitable[None] | None]


class CommandExecution:
    """Context-local handle exposed to process adapters during one execution."""

    def __init__(self, receipt: CommandReceipt, persist: ReceiptPersist):
        if not isinstance(receipt, CommandReceipt):
            raise TypeError("receipt must be CommandReceipt")
        if not callable(persist):
            raise TypeError("persist must be callable")
        self._receipt = receipt
        self._persist = persist
        self._lock = asyncio.Lock()

    @property
    def receipt(self) -> CommandReceipt:
        return self._receipt

    @property
    def execution_id(self) -> str:
        return self._receipt.execution_id

    async def update(
        self,
        state: ReceiptState,
        *,
        resource: Mapping[str, Any] | None = None,
        returncode: int | None = None,
        stdout: bytes | str | None = None,
        stderr: bytes | str | None = None,
        tool_output_recorded: bool | None = None,
    ) -> CommandReceipt:
        """Persist a monotonic state update before the caller advances."""
        if state not in {
            "intent",
            "launched",
            "completed",
            "failed",
            "terminated",
            "unknown",
            "reconciled",
        }:
            raise ValueError(f"Unsupported command receipt state: {state!r}")
        if resource is not None and not isinstance(resource, Mapping):
            raise TypeError("resource must be a mapping or None")
        if returncode is not None and type(returncode) is not int:
            raise TypeError("returncode must be an integer or None")
        async with self._lock:
            current = self._receipt
            if not _valid_transition(current.state, state):
                raise ValueError(
                    f"Invalid command receipt transition {current.state!r} -> {state!r}"
                )
            updated = msgspec.structs.replace(
                current,
                state=state,
                updated_at=_utc_now(),
                resource=(
                    copy.deepcopy(dict(resource))
                    if resource is not None
                    else current.resource
                ),
                returncode=(
                    returncode if returncode is not None else current.returncode
                ),
                stdout=(
                    _bounded_output(stdout) if stdout is not None else current.stdout
                ),
                stderr=(
                    _bounded_output(stderr) if stderr is not None else current.stderr
                ),
                tool_output_recorded=(
                    tool_output_recorded
                    if tool_output_recorded is not None
                    else current.tool_output_recorded
                ),
            )
            result = self._persist(updated)
            if inspect.isawaitable(result):
                await result
            self._receipt = updated
            return updated


_CURRENT_COMMAND_EXECUTION: contextvars.ContextVar[CommandExecution | None] = (
    contextvars.ContextVar("msgflux_command_execution", default=None)
)
_CURRENT_RECEIPT_PERSIST: contextvars.ContextVar[ReceiptPersist | None] = (
    contextvars.ContextVar("msgflux_command_receipt_persist", default=None)
)


def get_command_execution() -> CommandExecution | None:
    """Return the active command receipt session for a process adapter."""
    return _CURRENT_COMMAND_EXECUTION.get()


def get_command_receipt_persist() -> ReceiptPersist | None:
    """Return the active Agent checkpoint callback, if one is installed."""
    return _CURRENT_RECEIPT_PERSIST.get()


@contextmanager
def bind_command_receipt_persist(persist: ReceiptPersist):
    if not callable(persist):
        raise TypeError("persist must be callable")
    token = _CURRENT_RECEIPT_PERSIST.set(persist)
    try:
        yield
    finally:
        _CURRENT_RECEIPT_PERSIST.reset(token)


@contextmanager
def bind_command_execution(execution: CommandExecution):
    """Internal context manager that exposes one receipt session to an adapter."""
    if not isinstance(execution, CommandExecution):
        raise TypeError("execution must be CommandExecution")

    token = _CURRENT_COMMAND_EXECUTION.set(execution)
    try:
        yield execution
    finally:
        _CURRENT_COMMAND_EXECUTION.reset(token)


def new_command_receipt(
    *,
    workspace_reference: Mapping[str, Any] | None,
    backend: str,
    owner_id: str | None = None,
    run_id: str | None = None,
    task_id: str | None = None,
    tool_call_id: str | None = None,
    message_offset: int | None = None,
) -> CommandReceipt:
    """Create the initial intent receipt with a host-generated stable ID."""
    now = _utc_now()
    return CommandReceipt(
        version=1,
        execution_id=uuid4().hex,
        state="intent",
        workspace_reference=(
            dict(workspace_reference) if workspace_reference is not None else None
        ),
        backend=backend,
        created_at=now,
        updated_at=now,
        owner_id=owner_id,
        run_id=run_id,
        task_id=task_id,
        tool_call_id=tool_call_id,
        message_offset=message_offset,
    )


def serialize_command_receipt(receipt: CommandReceipt) -> dict[str, Any]:
    if not isinstance(receipt, CommandReceipt):
        raise TypeError("receipt must be CommandReceipt")
    return receipt.to_dict()


def decode_command_receipt(value: Mapping[str, Any]) -> CommandReceipt:
    """Validate a persisted v1 receipt before recovery or backend inspection."""
    if not isinstance(value, Mapping):
        raise TypeError("command receipt must be a mapping")
    try:
        receipt = msgspec.convert(value, type=CommandReceipt)
    except (msgspec.ValidationError, TypeError, ValueError) as error:
        raise ValueError("Invalid or unsupported command receipt") from error
    if receipt.version != 1:
        raise ValueError("Unsupported command receipt version")
    return receipt


def retain_command_receipts(
    current: Sequence[Mapping[str, Any] | CommandReceipt],
    receipt: CommandReceipt,
    *,
    terminal_limit: int = MAX_TERMINAL_RECEIPTS,
) -> list[dict[str, Any]]:
    """Upsert a receipt, retaining unresolved records and the newest terminals."""
    if not isinstance(receipt, CommandReceipt):
        raise TypeError("receipt must be CommandReceipt")
    if type(terminal_limit) is not int or terminal_limit < 0:
        raise ValueError("terminal_limit must be a non-negative integer")
    entries = [
        item if isinstance(item, CommandReceipt) else decode_command_receipt(item)
        for item in current
    ]
    matches = [item for item in entries if item.execution_id == receipt.execution_id]
    if matches:
        original = matches[-1]
        if _immutable_identity(original) != _immutable_identity(receipt):
            raise ValueError("Command execution ID was reused with another identity")
        entries = [
            item for item in entries if item.execution_id != receipt.execution_id
        ]
    entries.append(receipt)
    unresolved = [item for item in entries if item.unresolved]
    terminals = [item for item in entries if not item.unresolved]
    terminals = terminals[-terminal_limit:] if terminal_limit else []
    return [serialize_command_receipt(item) for item in (*unresolved, *terminals)]


def unresolved_command_receipts(
    values: Sequence[Mapping[str, Any] | CommandReceipt],
) -> tuple[CommandReceipt, ...]:
    entries = tuple(
        item if isinstance(item, CommandReceipt) else decode_command_receipt(item)
        for item in values
    )
    return tuple(item for item in entries if item.unresolved)


def task_command_receipts(task_store: Any, task_id: str) -> tuple[CommandReceipt, ...]:
    """Return the latest durable receipt snapshot for each background command."""
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be a non-empty string")
    activities = task_store.list_activity(task_id)
    latest: dict[str, CommandReceipt] = {}
    for activity in activities:
        if getattr(activity, "kind", None) != "command_receipt":
            continue
        metadata = getattr(activity, "metadata", None)
        value = metadata.get("receipt") if isinstance(metadata, Mapping) else None
        receipt = decode_command_receipt(value)
        if receipt.task_id != task_id:
            raise ValueError("Background command receipt task identity mismatch")
        previous = latest.get(receipt.execution_id)
        if previous is not None and _immutable_identity(
            previous
        ) != _immutable_identity(receipt):
            raise ValueError("Command execution ID was reused with another identity")
        latest[receipt.execution_id] = receipt
    return tuple(latest.values())


def resolved_command_execution_ids(  # noqa: C901
    extensions: Mapping[str, Any],
    *,
    receipts: Sequence[Mapping[str, Any] | CommandReceipt] = (),
) -> frozenset[str]:
    """Validate explicit host reconciliation decisions against receipt identity."""
    if not isinstance(extensions, Mapping):
        raise ValueError("Command reconciliation extensions are malformed")
    record = extensions.get("command_reconciliations")
    if record is None:
        return frozenset()
    if not isinstance(record, Mapping) or record.get("schema_version") != 1:
        raise ValueError("Command reconciliation schema is unsupported")
    decisions = record.get("decisions")
    if not isinstance(decisions, Mapping):
        raise ValueError("Command reconciliation decisions are malformed")
    decoded_receipts = tuple(
        receipt
        if isinstance(receipt, CommandReceipt)
        else decode_command_receipt(receipt)
        for receipt in receipts
    )
    receipt_map = {receipt.execution_id: receipt for receipt in decoded_receipts}
    ledger_ids: set[str] = set()
    resolved: set[str] = set()
    for decision_id, decision in decisions.items():
        if not isinstance(decision_id, str) or not decision_id:
            raise ValueError("Command reconciliation decision ID is invalid")
        if not isinstance(decision, Mapping):
            raise ValueError("Command reconciliation decision is malformed")
        revision = decision.get("expected_revision")
        decided_by = decision.get("decided_by")
        reason = decision.get("reason")
        execution_ids = decision.get("execution_ids")
        workspace_reference = decision.get("workspace_reference")
        results = decision.get("results")
        if (
            type(revision) is not int
            or revision < 0
            or not isinstance(decided_by, str)
            or not decided_by
            or not isinstance(reason, str)
            or not reason
            or not isinstance(execution_ids, list)
            or not execution_ids
            or not all(isinstance(item, str) and item for item in execution_ids)
            or len(set(execution_ids)) != len(execution_ids)
            or not isinstance(results, Mapping)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in results.items()
            )
            or not isinstance(workspace_reference, Mapping)
        ):
            raise ValueError("Command reconciliation decision is malformed")
        for execution_id in execution_ids:
            if execution_id in ledger_ids:
                raise ValueError("Command execution was reconciled more than once")
            ledger_ids.add(execution_id)
            receipt = receipt_map.get(execution_id)
            if receipt is not None:
                if receipt.workspace_reference is None or any(
                    receipt.workspace_reference.get(field)
                    != workspace_reference.get(field)
                    for field in ("version", "workspace_id", "identity")
                ):
                    raise ValueError("Reconciliation workspace does not match command")
                if (
                    not isinstance(receipt.tool_call_id, str)
                    or receipt.tool_call_id not in results
                ):
                    raise ValueError(
                        "Reconciliation result does not match command tool call"
                    )
                resolved.add(execution_id)
    return frozenset(resolved)


def mark_tool_outputs_recorded(
    values: Sequence[Mapping[str, Any] | CommandReceipt],
    messages: Sequence[Mapping[str, Any]],
    *,
    terminal_limit: int = MAX_TERMINAL_RECEIPTS,
) -> list[dict[str, Any]]:
    """Mark receipts delivered only when output occurs after its intent offset."""
    if not isinstance(messages, Sequence):
        raise TypeError("messages must be a sequence")
    outputs: dict[str, list[int]] = {}
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        call_id = message.get("tool_call_id")
        if message.get("type") == "function_call_output":
            call_id = message.get("call_id", call_id)
        if isinstance(call_id, str):
            outputs.setdefault(call_id, []).append(index)
    receipts = [
        item if isinstance(item, CommandReceipt) else decode_command_receipt(item)
        for item in values
    ]
    updated = []
    for receipt in receipts:
        if (
            receipt.tool_call_id in outputs
            and any(
                receipt.message_offset is None or index >= receipt.message_offset
                for index in outputs[receipt.tool_call_id]
            )
            and receipt.state in {"completed", "failed", "terminated"}
            and not receipt.tool_output_recorded
        ):
            updated_receipt = msgspec.structs.replace(
                receipt, tool_output_recorded=True, updated_at=_utc_now()
            )
            updated.append(updated_receipt)
            continue
        updated.append(receipt)
    if not updated:
        return []
    return retain_command_receipts(
        updated[:-1], updated[-1], terminal_limit=terminal_limit
    )


def _immutable_identity(receipt: CommandReceipt) -> tuple[Any, ...]:
    return (
        receipt.version,
        receipt.execution_id,
        receipt.workspace_reference,
        receipt.backend,
        receipt.owner_id,
        receipt.run_id,
        receipt.task_id,
        receipt.tool_call_id,
        receipt.message_offset,
        receipt.created_at,
    )


def _valid_transition(current: ReceiptState, target: ReceiptState) -> bool:
    allowed = {
        "intent": {"intent", "launched", "completed", "failed", "unknown"},
        "launched": {"launched", "completed", "terminated", "unknown"},
        "completed": {"completed"},
        "failed": {"failed"},
        "terminated": {"terminated", "reconciled"},
        "unknown": {"unknown", "completed", "terminated", "reconciled"},
        "reconciled": {"reconciled"},
    }
    return target in allowed[current]


def _bounded_output(value: bytes | str) -> str:
    if isinstance(value, bytes):
        encoded = value
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        raise TypeError("command output must be bytes or string")
    if len(encoded) > MAX_RECEIPT_OUTPUT_BYTES:
        encoded = encoded[:MAX_RECEIPT_OUTPUT_BYTES]
    return encoded.decode("utf-8", errors="ignore")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "CommandExecution",
    "CommandReceipt",
    "MAX_RECEIPT_OUTPUT_BYTES",
    "MAX_TERMINAL_RECEIPTS",
    "bind_command_execution",
    "bind_command_receipt_persist",
    "decode_command_receipt",
    "get_command_execution",
    "get_command_receipt_persist",
    "mark_tool_outputs_recorded",
    "new_command_receipt",
    "retain_command_receipts",
    "resolved_command_execution_ids",
    "serialize_command_receipt",
    "task_command_receipts",
    "unresolved_command_receipts",
]
