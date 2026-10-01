"""Approval-bound workspace edits survive a host process restart."""

import asyncio
import multiprocessing
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import (
    AgentApprovals,
    AgentWorkspace,
    ExecutionScope,
    PermissionSet,
    SQLiteApprovalStore,
)
from msgflux.runtime.workspace.local import LocalWorkspaceBackend
from msgflux.runtime.workspace.registry import SQLiteWorkspaceRegistry
from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.tools.builtin import WriteTool
from msgflux.utils.msgspec import msgspec_dumps


WORKSPACE_ID = "approval-restart"
NAMESPACE = "editor"
THREAD_ID = "thread"
RUN_ID = "run"
TOOL_REVISION = "write-v1"
POLICY_VERSION = "policy-v1"
FILE_PATH = "/state.txt"


def _permissions(*actions):
    from msgflux.runtime.permissions import ResourcePermission

    return PermissionSet(
        {f"filesystem.{action}" for action in actions},
        {
            ResourcePermission(
                f"workspace:{WORKSPACE_ID}:{FILE_PATH}",
                f"filesystem.{action}",
            )
            for action in actions
        },
    )


def _response(*, tool_call):
    response = ModelResponse()
    if tool_call:
        calls = ToolCallAggregator()
        calls.process(
            0,
            "write-call",
            "write",
            msgspec_dumps({"path": "state.txt", "content": "new"}),
        )
        response.set_response_type("tool_call")
        response.add(calls)
    else:
        response.set_response_type("text_generation")
        response.add("done")
    return response


def _agent(
    checkpoints, journal, *, policy_version=POLICY_VERSION, tool_revision=TOOL_REVISION
):
    model = Mock()
    model.model_type = "chat_completion"
    return Agent(
        name=NAMESPACE,
        model=model,
        tools=[WriteTool()],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(journal, {"write": tool_revision}, policy_version),
    )


def _scope(workspace):
    return ExecutionScope(
        namespace=NAMESPACE,
        thread_id=THREAD_ID,
        run_id=RUN_ID,
        principal="host-user",
        workspace=workspace,
    )


def _open_stores(checkpoint_path, approval_path):
    return SQLiteCheckpointStore(checkpoint_path), SQLiteApprovalStore(approval_path)


def _first_process(registry_path, root, checkpoint_path, approval_path, child_pipe):
    checkpoints, journal = _open_stores(checkpoint_path, approval_path)
    backend = LocalWorkspaceBackend(
        root, registry=SQLiteWorkspaceRegistry(registry_path)
    )
    workspace = asyncio.run(
        AgentWorkspace.open(
            backend,
            WORKSPACE_ID,
            permissions=_permissions("read", "write"),
            write_guarantee="cooperative_compare",
        )
    )
    agent = _agent(checkpoints, journal)
    agent.generator.forward = Mock(return_value=_response(tool_call=True))
    try:
        agent("change file", scope=_scope(workspace))
    except TaskPauseRequestedError:
        pass
    else:
        raise AssertionError("protected write did not pause for host approval")
    pending = journal.pending(NAMESPACE, THREAD_ID, RUN_ID)
    assert len(pending) == 1
    child_pipe.send(
        (
            workspace.identity.backend,
            workspace.identity.resource_id,
            workspace.identity.generation,
            workspace.identity.config_revision,
            pending[0].request_id,
        )
    )
    child_pipe.close()
    os._exit(0)


def _second_process(
    registry_path,
    root,
    checkpoint_path,
    approval_path,
    identity_fields,
    request_id,
    case,
    child_pipe,
):
    from msgflux.runtime.workspace.contracts import WorkspaceIdentity

    identity = WorkspaceIdentity(
        backend=identity_fields[0],
        resource_id=identity_fields[1],
        generation=identity_fields[2],
        config_revision=identity_fields[3],
    )
    checkpoints, journal = _open_stores(checkpoint_path, approval_path)
    backend = LocalWorkspaceBackend(
        root, registry=SQLiteWorkspaceRegistry(registry_path)
    )
    permission_set = (
        _permissions("read")
        if case == "reduced_grant"
        else _permissions("read", "write")
    )
    workspace = asyncio.run(
        AgentWorkspace.reconnect(
            backend,
            WORKSPACE_ID,
            identity,
            permissions=permission_set,
            write_guarantee="cooperative_compare",
        )
    )
    policy_version = "policy-v2" if case == "policy_revision" else POLICY_VERSION
    tool_revision = "write-v2" if case == "tool_revision" else TOOL_REVISION
    agent = _agent(
        checkpoints,
        journal,
        policy_version=policy_version,
        tool_revision=tool_revision,
    )
    agent.generator.forward = Mock(return_value=_response(tool_call=False))
    agent.decide_approval(request_id, approved=True, decided_by="host")

    error_type = None
    result = None
    try:
        result = agent("continue", scope=_scope(workspace))
    except BaseException as exc:
        error_type = type(exc).__name__

    state = checkpoints.load_state(NAMESPACE, THREAD_ID, RUN_ID)
    call_count = sum(
        item.get("type") == "function_call" for item in state["messages"]["items"]
    )
    record = journal.get(NAMESPACE, request_id)
    try:
        content = Path(root, "state.txt").read_text()
    except FileNotFoundError:
        content = None
    child_pipe.send(
        {
            "result": result,
            "error": error_type,
            "content": content,
            "approval_status": record.status,
            "events": [event.status for event in journal.events(NAMESPACE, request_id)],
            "function_call_count": call_count,
            "resume_model_calls": agent.generator.forward.call_count,
        }
    )
    child_pipe.close()
    os._exit(0)


def _spawn_and_read(target, args, *, timeout=15):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(*args, sender))
    process.start()
    sender.close()
    try:
        assert receiver.poll(timeout), "approval recovery worker returned no result"
        result = receiver.recv()
        process.join(timeout)
        assert not process.is_alive(), "approval recovery worker did not exit"
        assert process.exitcode == 0
        return result
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
            process.join(5)
        if process.is_alive():
            process.kill()
            process.join(5)


@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
@pytest.mark.parametrize(
    "case",
    ["success", "changed_file", "reduced_grant", "policy_revision", "tool_revision"],
)
def test_pending_workspace_approval_reconnects_with_live_authority(tmp_path, case):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "state.txt").write_text("old")
    registry_path = str(tmp_path / "host" / "workspace.sqlite")
    checkpoint_path = str(tmp_path / "host" / "checkpoint.sqlite")
    approval_path = str(tmp_path / "host" / "approval.sqlite")

    fields = _spawn_and_read(
        _first_process,
        (registry_path, str(root), checkpoint_path, approval_path),
    )
    identity_fields, request_id = fields[:4], fields[4]
    assert (root / "state.txt").read_text() == "old"
    if case == "changed_file":
        (root / "state.txt").write_text("concurrent host edit")

    result = _spawn_and_read(
        _second_process,
        (
            registry_path,
            str(root),
            checkpoint_path,
            approval_path,
            identity_fields,
            request_id,
            case,
        ),
    )
    assert result["function_call_count"] == 1
    if case == "success":
        assert result["result"] == "done"
        assert result["error"] is None
        assert result["content"] == "new"
        assert result["approval_status"] == "consumed"
        assert result["events"] == ["pending", "approved", "consumed"]
        assert result["resume_model_calls"] == 1
    else:
        assert result["error"] in {
            "TaskPauseRequestedError",
            "PermissionError",
        }
        assert result["content"] == (
            "concurrent host edit" if case == "changed_file" else "old"
        )
        assert result["approval_status"] != "consumed"
        assert result["events"][:2] == ["pending", "approved"]
