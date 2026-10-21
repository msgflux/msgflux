"""SQLite coverage for explicit host reconciliation of command receipts."""

from __future__ import annotations

import asyncio
import json

import msgspec
import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.data.stores.base import CheckpointConflictError
from msgflux.runtime import AgentWorkspace, PermissionSet
from msgflux.runtime.workspace.local import LocalWorkspaceBackend
from msgflux.runtime.workspace.receipts import new_command_receipt
from msgflux.runtime.workspace.references import encode_workspace_reference
from msgflux.runtime.workspace.reconciliation import reconcile_command_results
from msgflux.runtime.workspace.receipts import (
    resolved_command_execution_ids,
)
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.tasks import SQLiteTaskStore


NAMESPACE = "worker"
THREAD_ID = "thread-1"
RUN_ID = "run-1"
WORKSPACE_ID = "reconcile-workspace"


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    registry = SQLiteWorkspaceRegistry(tmp_path / "workspace-registry.sqlite")
    workspace = asyncio.run(
        AgentWorkspace.open(
            LocalWorkspaceBackend(root, registry=registry),
            WORKSPACE_ID,
            permissions=PermissionSet({"filesystem.read", "filesystem.write"}),
        )
    )
    checkpoints = SQLiteCheckpointStore(tmp_path / "checkpoints.sqlite")
    tasks = SQLiteTaskStore(tmp_path / "tasks.sqlite")
    try:
        yield root, registry, workspace, checkpoints, tasks
    finally:
        asyncio.run(workspace.aclose())
        registry.close()
        checkpoints.close()
        tasks.close()


def _receipt(workspace, call_id, *, execution_id=None, message_offset=2, cwd=None):
    reference = encode_workspace_reference(workspace)
    if cwd is not None:
        reference["cwd"] = cwd
    receipt = new_command_receipt(
        workspace_reference=reference,
        backend=workspace.identity.backend,
        task_id=None,
        tool_call_id=call_id,
        message_offset=message_offset,
    )
    return msgspec.structs.replace(
        receipt,
        execution_id=execution_id or receipt.execution_id,
        state="unknown",
    )


def _checkpoint(checkpoints, workspace, *, calls, receipts, pending=None):
    messages = ChatMessages()
    messages.configure_thread(thread_id=THREAD_ID, namespace=NAMESPACE)
    messages.begin_turn(namespace=NAMESPACE, turn_id=RUN_ID)
    for call in calls:
        messages.append(call)
    extensions = {
        "workspace_reference": encode_workspace_reference(workspace),
        "command_receipts": [receipt.to_dict() for receipt in receipts],
    }
    if pending is not None:
        extensions["pending_approvals"] = pending
    state = {
        "schema_version": 1,
        "status": "running",
        "messages": messages._to_state(),
        "runtime": {"extensions": extensions},
    }
    commit = checkpoints.commit_state(
        NAMESPACE,
        THREAD_ID,
        RUN_ID,
        state,
        expected_revision=0,
        head_item_id=messages[-1]["item_id"],
    )
    return commit.revision


def _reconcile(checkpoints, workspace, revision, results, **kwargs):
    return reconcile_command_results(
        checkpoints,
        NAMESPACE,
        THREAD_ID,
        RUN_ID,
        expected_revision=revision,
        decision_id="host-decision-1",
        decided_by="operator@example.test",
        reason="verified process outcome after restart",
        worker_stopped=True,
        results=results,
        workspace=workspace,
        **kwargs,
    )


def test_reconciles_unknown_foreground_receipt_atomically_and_idempotently(
    setup,
):
    _root, _registry, workspace, checkpoints, _tasks = setup
    receipt = _receipt(workspace, "call-1")
    revision = _checkpoint(
        checkpoints,
        workspace,
        calls=[
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "bash",
                "arguments": "{}",
            }
        ],
        receipts=[receipt],
    )

    report = _reconcile(
        checkpoints, workspace, revision, {"call-1": "confirmed output"}
    )
    state = checkpoints.load_state(NAMESPACE, THREAD_ID, RUN_ID)
    output = [
        item
        for item in state["messages"]["items"]
        if item.get("type") == "function_call_output"
    ]
    saved_receipt = state["runtime"]["extensions"]["command_receipts"][0]
    assert report.checkpoint_revision == revision + 1
    assert report.execution_ids == (receipt.execution_id,)
    assert [item["output"] for item in output] == ["confirmed output"]
    assert saved_receipt["state"] == "reconciled"
    assert saved_receipt["tool_output_recorded"] is True
    assert state["status"] == "paused"

    repeated = _reconcile(
        checkpoints, workspace, revision, {"call-1": "confirmed output"}
    )
    assert repeated.idempotent is True
    assert (
        len(
            [
                item
                for item in checkpoints.load_state(NAMESPACE, THREAD_ID, RUN_ID)[
                    "messages"
                ]["items"]
                if item.get("type") == "function_call_output"
            ]
        )
        == 1
    )


def test_stale_revision_and_decision_id_parameter_reuse_conflict(setup):
    _root, _registry, workspace, checkpoints, _tasks = setup
    receipt = _receipt(workspace, "call-1")
    revision = _checkpoint(
        checkpoints,
        workspace,
        calls=[{"type": "function_call", "call_id": "call-1", "name": "bash"}],
        receipts=[receipt],
    )
    _reconcile(checkpoints, workspace, revision, {"call-1": "result"})
    with pytest.raises(CheckpointConflictError):
        reconcile_command_results(
            checkpoints,
            NAMESPACE,
            THREAD_ID,
            RUN_ID,
            expected_revision=revision,
            decision_id="other-decision",
            decided_by="operator@example.test",
            reason="stale",
            worker_stopped=True,
            results={"call-1": "result"},
            workspace=workspace,
        )
    with pytest.raises(CheckpointConflictError, match="reused"):
        _reconcile(checkpoints, workspace, revision, {"call-1": "different result"})


def test_requires_explicit_worker_quiescence_and_exact_workspace(setup, tmp_path):
    _root, registry, workspace, checkpoints, _tasks = setup
    receipt = _receipt(workspace, "call-1")
    revision = _checkpoint(
        checkpoints,
        workspace,
        calls=[{"type": "function_call", "call_id": "call-1", "name": "bash"}],
        receipts=[receipt],
    )
    with pytest.raises(ValueError, match="confirm"):
        reconcile_command_results(
            checkpoints,
            NAMESPACE,
            THREAD_ID,
            RUN_ID,
            expected_revision=revision,
            decision_id="host-decision-1",
            decided_by="operator@example.test",
            reason="verified",
            worker_stopped=1,
            results={"call-1": "result"},
            workspace=workspace,
        )

    other_root = tmp_path / "other"
    other_root.mkdir()
    other = asyncio.run(
        AgentWorkspace.open(
            LocalWorkspaceBackend(other_root, registry=registry),
            "other-workspace",
            permissions=PermissionSet({"filesystem.read", "filesystem.write"}),
        )
    )
    try:
        with pytest.raises(ValueError, match="does not match"):
            reconcile_command_results(
                checkpoints,
                NAMESPACE,
                THREAD_ID,
                RUN_ID,
                expected_revision=revision,
                decision_id="host-decision-1",
                decided_by="operator@example.test",
                reason="verified",
                worker_stopped=True,
                results={"call-1": "result"},
                workspace=other,
            )
    finally:
        asyncio.run(other.aclose())


def test_native_output_rendering_and_latest_call_offset(setup):
    _root, _registry, workspace, checkpoints, _tasks = setup
    native = {
        "codec": "openai.responses.shell",
        "version": 1,
        "name": "bash",
        "command_count": 1,
        "max_output_length": None,
    }
    receipt = _receipt(workspace, "same-call", message_offset=4)
    receipt2 = _receipt(workspace, "same-call", message_offset=4)
    calls = [
        {
            "type": "function_call",
            "call_id": "same-call",
            "name": "old",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "same-call",
            "output": "old result",
        },
        {
            "type": "shell_call",
            "call_id": "same-call",
            "action": {"commands": ["echo hi"]},
            "metadata": {"tool_transport": native},
        },
    ]
    revision = _checkpoint(
        checkpoints, workspace, calls=calls, receipts=[receipt, receipt2]
    )
    result = json.dumps(
        {
            "results": [
                {"status": "exited", "returncode": 0, "stdout": "ok", "stderr": ""}
            ]
        }
    )
    report = _reconcile(checkpoints, workspace, revision, {"same-call": result})
    state = checkpoints.load_state(NAMESPACE, THREAD_ID, RUN_ID)
    outputs = [
        item
        for item in state["messages"]["items"]
        if item.get("type") == "shell_call_output"
    ]
    assert report.tool_call_ids == ("same-call",)
    assert set(report.execution_ids) == {receipt.execution_id, receipt2.execution_id}
    assert len(outputs) == 1
    assert outputs[0]["call_id"] == "same-call"


def test_background_receipt_resolution_crosses_db_only_via_checkpoint_decision(setup):
    _root, _registry, workspace, checkpoints, tasks = setup
    task_id = "task-background-1"
    tasks.create(
        "worker",
        task_id=task_id,
        metadata={
            "task_kind": "agent",
            "checkpoint_namespace": NAMESPACE,
            "checkpoint_thread_id": THREAD_ID,
            "checkpoint_run_id": RUN_ID,
            "checkpoint_store_id": checkpoints.routing_id,
        },
    )
    receipt = msgspec.structs.replace(
        _receipt(workspace, "call-background", cwd="/nested"), task_id=task_id
    )
    tasks.add_activity(
        task_id,
        kind="command_receipt",
        summary="command may have run",
        metadata={"receipt": receipt.to_dict()},
    )
    revision = _checkpoint(
        checkpoints,
        workspace,
        calls=[
            {
                "type": "function_call",
                "call_id": "call-background",
                "name": "bash",
                "arguments": "{}",
            }
        ],
        receipts=[],
    )
    report = _reconcile(
        checkpoints,
        workspace,
        revision,
        {"call-background": "operator confirmed"},
        task_store=tasks,
        task_id=task_id,
    )
    state = checkpoints.load_state(NAMESPACE, THREAD_ID, RUN_ID)
    extensions = state["runtime"]["extensions"]
    decision_ids = resolved_command_execution_ids(extensions, receipts=(receipt,))
    assert report.execution_ids == (receipt.execution_id,)
    assert decision_ids == frozenset({receipt.execution_id})
    persisted_receipt = next(
        activity.metadata["receipt"]
        for activity in tasks.list_activity(task_id)
        if activity.kind == "command_receipt"
    )
    assert persisted_receipt["state"] == "unknown"
    assert persisted_receipt["tool_output_recorded"] is False


def test_executing_protected_batch_uses_existing_approval_reconciliation(setup):
    _root, _registry, workspace, checkpoints, _tasks = setup
    receipt = _receipt(workspace, "call-protected")
    revision = _checkpoint(
        checkpoints,
        workspace,
        calls=[
            {
                "type": "function_call",
                "call_id": "call-protected",
                "name": "bash",
                "arguments": "{}",
            }
        ],
        receipts=[receipt],
        pending={"schema_version": 1, "phase": "executing"},
    )
    with pytest.raises(ValueError, match=r"Agent\.reconcile_approvals"):
        _reconcile(checkpoints, workspace, revision, {"call-protected": "result"})
    assert (
        checkpoints.load_state(NAMESPACE, THREAD_ID, RUN_ID)["_checkpoint"]["revision"]
        == revision
    )


def test_existing_output_is_not_duplicated_and_mismatch_is_rejected(setup):
    _root, _registry, workspace, checkpoints, _tasks = setup
    receipt = _receipt(workspace, "call-existing", message_offset=2)
    call = {
        "type": "function_call",
        "call_id": "call-existing",
        "name": "bash",
        "arguments": "{}",
    }
    output = {
        "type": "function_call_output",
        "call_id": "call-existing",
        "output": "already confirmed",
    }
    revision = _checkpoint(
        checkpoints, workspace, calls=[call, output], receipts=[receipt]
    )
    report = _reconcile(
        checkpoints,
        workspace,
        revision,
        {"call-existing": "already confirmed"},
    )
    state = checkpoints.load_state(NAMESPACE, THREAD_ID, RUN_ID)
    assert report.checkpoint_revision == revision + 1
    assert (
        sum(
            item.get("type") == "function_call_output"
            and item.get("call_id") == "call-existing"
            for item in state["messages"]["items"]
        )
        == 1
    )

    other_run = "run-conflict"
    messages = ChatMessages()
    messages.configure_thread(thread_id=THREAD_ID, namespace=NAMESPACE)
    messages.begin_turn(namespace=NAMESPACE, turn_id=other_run)
    messages.append(call)
    messages.append(output)
    other_receipt = _receipt(workspace, "call-existing", message_offset=2)
    other_state = {
        "schema_version": 1,
        "status": "running",
        "messages": messages._to_state(),
        "runtime": {
            "extensions": {
                "workspace_reference": encode_workspace_reference(workspace),
                "command_receipts": [other_receipt.to_dict()],
            }
        },
    }
    checkpoints.commit_state(
        NAMESPACE,
        THREAD_ID,
        other_run,
        other_state,
        expected_revision=0,
        head_item_id=messages[-1]["item_id"],
    )
    with pytest.raises(ValueError, match="conflicts"):
        reconcile_command_results(
            checkpoints,
            NAMESPACE,
            THREAD_ID,
            other_run,
            expected_revision=1,
            decision_id="mismatch",
            decided_by="operator@example.test",
            reason="wrong confirmation",
            worker_stopped=True,
            results={"call-existing": "different"},
            workspace=workspace,
        )
