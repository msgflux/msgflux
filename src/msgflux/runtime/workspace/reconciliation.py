"""Host-only reconciliation of uncertain workspace command results."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

import msgspec

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores.base import CheckpointConflictError
from msgflux.models.tool_transport import (
    native_item_types,
    render_native_output,
)
from msgflux.runtime.approvals.reconciliation import (
    _validate_decision,
    inspect_batch,
)
from msgflux.runtime.workspace.receipts import (
    CommandReceipt,
    decode_command_receipt,
    mark_tool_outputs_recorded,
    resolved_command_execution_ids,
    retain_command_receipts,
    task_command_receipts,
    unresolved_command_receipts,
)
from msgflux.runtime.workspace.references import (
    encode_workspace_reference,
    validate_workspace_reference,
)
from msgflux.utils.msgspec import msgspec_loads

COMMAND_RECONCILIATIONS_KEY = "command_reconciliations"


def _message_maps(message_items, boundary):
    recorded_outputs: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    call_items: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, item in enumerate(message_items):
        if index < boundary:
            continue
        if not isinstance(item, Mapping):
            continue
        item_type = item.get("type")
        call_id = item.get("call_id") or item.get("id")
        if item_type == "function_call" or item_type in native_item_types():
            if isinstance(call_id, str) and call_id:
                call_items.setdefault(call_id, []).append((index, item))
        output_id = item.get("call_id")
        if item_type == "function_call_output" or item_type in native_item_types(
            output=True
        ):
            if isinstance(output_id, str) and output_id:
                recorded_outputs.setdefault(output_id, []).append((index, item))

    return call_items, recorded_outputs


def _render_output(call_id, call_item, result):
    metadata = (call_item.get("metadata") or {}).get("tool_transport")
    if metadata is not None:
        return render_native_output(call_id, msgspec_loads(result), metadata)
    return {"type": "function_call_output", "call_id": call_id, "output": result}


def _select_receipt_calls(call_items, receipts):
    raw_ids = [receipt.tool_call_id for receipt in receipts]
    call_ids = {item for item in raw_ids if isinstance(item, str) and item}
    if any(not isinstance(item, str) or not item for item in raw_ids):
        raise ValueError("Every unresolved command receipt must identify its tool call")
    selected = {}
    offsets = {}
    for receipt in receipts:
        matching = [
            entry
            for entry in call_items.get(receipt.tool_call_id, ())
            if receipt.message_offset is None or entry[0] < receipt.message_offset
        ]
        if not matching:
            raise ValueError("An unresolved command receipt has no matching tool call")
        selected[receipt.tool_call_id] = max(matching, key=lambda entry: entry[0])
        old_offset = offsets.get(receipt.tool_call_id)
        offset = receipt.message_offset
        if old_offset is None or (offset is not None and offset > old_offset):
            offsets[receipt.tool_call_id] = offset
    return selected, call_ids, offsets


def _is_paired(call_id, call_index, offset, recorded_outputs):
    threshold = call_index if offset is None else max(call_index, offset)
    return any(
        index >= threshold for index, _output in recorded_outputs.get(call_id, ())
    )


def _unresolved_outputs(message_items, receipts, results, turn_start):
    call_items, recorded_outputs = _message_maps(message_items, turn_start)
    selected, call_ids, receipt_offsets = _select_receipt_calls(call_items, receipts)
    call_items.update({call_id: [entry] for call_id, entry in selected.items()})
    if not call_ids.issubset(call_items):
        raise ValueError("An unresolved command receipt has no matching tool call")

    unpaired = set()
    for call_id, entries in call_items.items():
        call_index, _item = entries[-1]
        if not _is_paired(
            call_id, call_index, receipt_offsets.get(call_id), recorded_outputs
        ):
            unpaired.add(call_id)
    required_ids = unpaired | call_ids
    if set(results) != required_ids:
        raise ValueError("Results must cover every pending tool call and no other call")
    if not receipts:
        raise ValueError("No unresolved command receipts require reconciliation")

    outputs = []
    for call_id in sorted(required_ids):
        _call_index, item = call_items[call_id][-1]
        rendered = _render_output(call_id, item, results[call_id])
        threshold = max(_call_index, receipt_offsets.get(call_id) or _call_index)
        existing = [
            output
            for output_index, output in recorded_outputs.get(call_id, ())
            if output_index >= threshold
        ]
        if not existing:
            outputs.append(rendered)
        elif _stable_output(existing[-1]) != _stable_output(rendered):
            raise ValueError("Existing tool output conflicts with host confirmation")
    return outputs, tuple(sorted(required_ids))


def _stable_output(item):
    return {key: value for key, value in item.items() if key not in {"id", "item_id"}}


def _current_turn_start(messages):
    active = messages.get_active_turn()
    if active is None:
        return 0
    value = active.get("start_item_index", 0)
    return value if type(value) is int and value >= 0 else 0


def _mark_foreground_receipts(extensions, execution_ids, updated_at):
    current = extensions.get("command_receipts", [])
    retained = []
    for value in current:
        receipt = (
            value
            if isinstance(value, CommandReceipt)
            else decode_command_receipt(value)
        )
        if receipt.execution_id in execution_ids:
            receipt = msgspec.structs.replace(
                receipt,
                state="reconciled",
                tool_output_recorded=True,
                updated_at=updated_at,
            )
        retained = retain_command_receipts(retained, receipt)
    extensions["command_receipts"] = retained


def _validate_receipt_workspace(reference, workspace):
    if not isinstance(reference, Mapping):
        raise ValueError("Command receipt has no valid recorded workspace reference")
    validate_workspace_reference(reference, workspace, match_cwd=False)


def _validate_task_route(
    task_store, task_id, checkpoint_store, namespace, thread_id, run_id
):
    task = task_store.get(task_id)
    if task is None:
        raise ValueError("Background task was not found")
    metadata = task.metadata
    route = (
        metadata.get("checkpoint_namespace"),
        metadata.get("checkpoint_thread_id"),
        metadata.get("checkpoint_run_id"),
        metadata.get("checkpoint_store_id"),
    )
    expected = (namespace, thread_id, run_id, checkpoint_store.routing_id)
    if route != expected or metadata.get("task_kind") != "agent":
        raise ValueError("Background task checkpoint routing does not match")
    return task


def _load_pending_receipts(extensions, activity_receipts, message_items, workspace):
    activity_data = mark_tool_outputs_recorded(activity_receipts, message_items)
    activity = tuple(decode_command_receipt(item) for item in activity_data)
    foreground_data = mark_tool_outputs_recorded(
        extensions.get("command_receipts", []), message_items
    )
    foreground = tuple(unresolved_command_receipts(foreground_data))
    receipts = {}
    for receipt in (*activity, *foreground):
        previous = receipts.get(receipt.execution_id)
        if previous is not None and (
            previous.workspace_reference != receipt.workspace_reference
            or previous.tool_call_id != receipt.tool_call_id
        ):
            raise ValueError("Command execution ID has conflicting receipt identity")
        receipts[receipt.execution_id] = receipt
    resolved = resolved_command_execution_ids(
        extensions, receipts=tuple(receipts.values())
    )
    pending = tuple(
        receipt
        for execution_id, receipt in receipts.items()
        if receipt.unresolved and execution_id not in resolved
    )
    for receipt in pending:
        if receipt.workspace_reference is None:
            raise ValueError("Command receipt has no recorded workspace identity")
        _validate_receipt_workspace(receipt.workspace_reference, workspace)
    foreground_ids = {receipt.execution_id for receipt in foreground}
    return tuple(
        receipt for receipt in pending if receipt.execution_id in foreground_ids
    ), pending


def _check_previous_decision(
    previous, decision, current_workspace_reference, task_id, current_revision
):
    if previous is None:
        return None
    if not isinstance(previous, Mapping):
        raise ValueError("Stored command reconciliation decision is malformed")
    for key, value in decision.items():
        if previous.get(key) != value:
            raise CheckpointConflictError(
                "Command reconciliation decision ID was reused"
            )
    if previous.get("workspace_reference") != current_workspace_reference:
        raise ValueError("Decision workspace does not match the live workspace")
    if previous.get("task_id") != task_id:
        raise CheckpointConflictError("Decision belongs to a different task")
    return CommandReconciliationReport(
        version=1,
        decision_id=previous["decision_id"],
        checkpoint_revision=current_revision,
        execution_ids=tuple(previous.get("execution_ids", ())),
        tool_call_ids=tuple(sorted(previous.get("results", {}))),
        task_id=task_id,
        idempotent=True,
    )


def _reject_pending_approval(extensions, state):
    pending = extensions.get("pending_approvals") or state.get("pending_approvals")
    if pending is None:
        return
    if (
        isinstance(pending, Mapping)
        and pending.get("schema_version") == 1
        and pending.get("phase") == "executing"
    ):
        raise ValueError(
            "Protected tool batch is executing; use Agent.reconcile_approvals()"
        )
    raise ValueError("Resolve the pending approval with the Agent approval API first")


def _load_checkpoint_context(checkpoint_store, namespace, thread_id, run_id, workspace):
    state = inspect_batch(checkpoint_store, namespace, thread_id, run_id)
    revision = state.get("_checkpoint", {}).get("revision", 0)
    extensions = state.get("runtime", {}).get("extensions", {})
    if not isinstance(extensions, dict):
        raise ValueError("Checkpoint runtime extensions are malformed")
    reference = extensions.get("workspace_reference")
    if reference is not None:
        validate_workspace_reference(reference, workspace)
    messages = state.get("messages")
    if not isinstance(messages, Mapping):
        raise ValueError("Checkpoint messages are unavailable")
    items = messages.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Checkpoint message items are malformed")
    return state, revision, extensions, messages, items


def _validate_request(task_store, task_id, worker_stopped, workspace):
    _validate_request(task_store, task_id, worker_stopped, workspace)


class CommandReconciliationReport(
    msgspec.Struct, frozen=True, forbid_unknown_fields=True
):
    """Compact v1 record of a completed host reconciliation decision."""

    version: Literal[1]
    decision_id: str
    checkpoint_revision: int
    execution_ids: tuple[str, ...]
    tool_call_ids: tuple[str, ...]
    task_id: str | None = None
    idempotent: bool = False


def reconcile_command_results(
    checkpoint_store,
    namespace: str,
    thread_id: str,
    run_id: str,
    *,
    expected_revision: int,
    decision_id: str,
    decided_by: str,
    reason: str,
    worker_stopped: bool,
    results: Mapping[str, str],
    workspace,
    task_store=None,
    task_id: str | None = None,
) -> CommandReconciliationReport:
    """Commit host-confirmed command outputs without executing model or tools.

    For background work, the checkpoint commit is the durable resolution marker
    for immutable command receipts stored in the separate task database.
    """
    if (task_store is None) != (task_id is None):
        raise ValueError("task_store and task_id must be provided together")
    if worker_stopped is not True:
        raise ValueError(
            "Host must confirm the worker and command processes have stopped"
        )
    workspace.require_active()
    decision = _validate_decision(
        expected_revision,
        decision_id,
        decided_by,
        reason,
        worker_stopped,
        results,
        False,
    )
    (
        state,
        current_revision,
        extensions,
        messages_state,
        message_items,
    ) = _load_checkpoint_context(
        checkpoint_store, namespace, thread_id, run_id, workspace
    )
    current_workspace_reference = encode_workspace_reference(workspace)

    activity_receipts: tuple[CommandReceipt, ...] = ()
    if task_store is not None:
        _validate_task_route(
            task_store, task_id, checkpoint_store, namespace, thread_id, run_id
        )
        activity_receipts = task_command_receipts(task_store, task_id)

    messages = ChatMessages()
    messages._hydrate_state(messages_state)
    turn_start = _current_turn_start(messages)
    foreground_receipts, pending_receipts = _load_pending_receipts(
        extensions, activity_receipts, message_items, workspace
    )

    previous = (
        extensions.get(COMMAND_RECONCILIATIONS_KEY, {})
        .get("decisions", {})
        .get(decision_id)
    )
    replay = _check_previous_decision(
        previous,
        decision,
        current_workspace_reference,
        task_id,
        current_revision,
    )
    if replay is not None:
        return replay

    if current_revision != expected_revision:
        raise CheckpointConflictError(
            "Command reconciliation checkpoint revision changed"
        )
    _reject_pending_approval(extensions, state)

    output_items, receipt_call_ids = _unresolved_outputs(
        message_items, pending_receipts, results, turn_start
    )
    if messages.get_active_turn() is None:
        messages.resume_turn(run_id, metadata={"source": "command_reconciliation"})
    for item in output_items:
        messages.append(item)
    messages.end_turn(event="pause")

    execution_ids = tuple(sorted(receipt.execution_id for receipt in pending_receipts))
    now = datetime.now(UTC).isoformat()
    reconciliation_record = {
        **decision,
        "workspace_reference": current_workspace_reference,
        "execution_ids": list(execution_ids),
        "task_id": task_id,
        "resolved_at": now,
    }
    ledger = extensions.setdefault(
        COMMAND_RECONCILIATIONS_KEY,
        {"schema_version": 1, "decisions": {}},
    )
    if (
        not isinstance(ledger, dict)
        or ledger.get("schema_version") != 1
        or not isinstance(ledger.get("decisions"), dict)
    ):
        raise ValueError("Command reconciliation ledger is malformed")
    ledger["decisions"][decision_id] = reconciliation_record

    if foreground_receipts:
        foreground_ids = {receipt.execution_id for receipt in foreground_receipts}
        _mark_foreground_receipts(
            extensions, foreground_ids.intersection(execution_ids), now
        )

    state["messages"] = messages._to_state()
    state["status"] = "paused"
    committed = checkpoint_store.commit_state(
        namespace,
        thread_id,
        run_id,
        state,
        expected_revision=expected_revision,
        extension_state=extensions,
        head_item_id=messages[-1]["item_id"],
        event={
            "event_type": "command.reconciled",
            "decision_id": decision_id,
            "decided_by": decided_by,
            "execution_ids": list(execution_ids),
            "tool_call_ids": list(receipt_call_ids),
        },
    )
    return CommandReconciliationReport(
        version=1,
        decision_id=decision_id,
        checkpoint_revision=committed.revision,
        execution_ids=execution_ids,
        tool_call_ids=receipt_call_ids,
        task_id=task_id,
    )


__all__ = [
    "CommandReconciliationReport",
    "reconcile_command_results",
]
