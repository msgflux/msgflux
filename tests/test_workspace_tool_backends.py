import os
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.exceptions import TaskPauseRequestedError
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent, ToolLibrary
from msgflux.runtime import (
    AgentApprovals,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryApprovalStore,
    InMemoryWorkspace,
    InMemoryWorkspaceBackend,
    LocalWorkspaceBackend,
    PermissionSet,
    execution_context,
)
from msgflux.runtime.workspace_local import LocalWorkspace
from msgflux.tools.builtin import ApplyPatchTool, EditTool, ReadFileTool, WriteTool
from msgflux.tools.workspace_changes import workspace_change_execution
from msgflux.utils.msgspec import msgspec_dumps


def _scope(environment, filesystem, *, read=True, write=True, delete=True):
    actions = []
    if read:
        actions.append("read")
    if write:
        actions.append("write")
    if delete:
        actions.append("delete")
    return ExecutionScope(
        namespace="workspace",
        thread_id="thread",
        run_id="run",
        principal="user",
        environment=environment,
        permissions=PermissionSet(
            resources=[
                filesystem.permission("/a", f"filesystem.{action}")
                for action in actions
            ]
        ),
    )


@pytest.fixture(params=[("memory", "atomic_compare"), ("local", "cooperative_compare")])
def backend_scope(request, tmp_path):
    kind, guarantee = request.param
    if kind == "local":
        if os.name != "posix":
            pytest.skip("POSIX local backend")
        (tmp_path / "a").write_bytes(b"one\ntwo\nthree\n")
    filesystem = (
        InMemoryWorkspace("files", {"/a": b"one\ntwo\nthree\n"})
        if kind == "memory"
        else LocalWorkspace("files", tmp_path)
    )
    environment = ExecutionEnvironment(filesystem, write_guarantee=guarantee)
    return filesystem, environment, guarantee


def _tools():
    return [WriteTool(), EditTool(), ApplyPatchTool()]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_workspace_tools_are_backend_neutral_and_read_bounds_work(
    backend_scope, asynchronous
):
    filesystem, environment, _ = backend_scope
    scope = _scope(environment, filesystem)
    read = ReadFileTool()
    write, edit, patch = _tools()
    library = ToolLibrary("workspace", [read, write, edit, patch])
    for name, fields in (
        ("read", {"path", "offset", "limit"}),
        ("write", {"path", "content"}),
        ("edit", {"path", "old", "new"}),
        ("apply_patch", {"operation", "path", "diff"}),
    ):
        assert (
            set(library.get_tool_definition(name).input_schema["properties"]) == fields
        )
    with execution_context(scope=scope):
        if asynchronous:
            assert (
                await read.acall("a", offset=2, limit=1, filesystem=filesystem)
                == "two\n"
            )
        else:
            assert read("a", offset=2, limit=1, filesystem=filesystem) == "two\n"

        for tool, arguments in (
            (write, {"path": "a", "content": "changed"}),
            (edit, {"path": "a", "old": "changed", "new": "CHANGED"}),
            (
                patch,
                {"operation": "update", "path": "a", "diff": "@@\n-CHANGED\n+final"},
            ),
        ):
            if asynchronous:
                result = await tool.acall(**arguments, filesystem=filesystem)
            else:
                result = tool(**arguments, filesystem=filesystem)
            assert result == {"status": "completed"}
        assert filesystem.read_text("/a") == "final"
        if asynchronous:
            await patch.acall("delete", "a", filesystem=filesystem)
            await patch.acall("create", "a", "+created", filesystem=filesystem)
        else:
            patch("delete", "a", filesystem=filesystem)
            patch("create", "a", "+created", filesystem=filesystem)
        assert filesystem.read_text("/a") == "created"


@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
def test_local_default_strict_guarantee_rejects_mutation_but_allows_read(tmp_path):
    (tmp_path / "a").write_bytes(b"one\ntwo\nthree\n")
    filesystem = LocalWorkspace("files", tmp_path)
    environment = ExecutionEnvironment(filesystem)
    scope = _scope(environment, filesystem)
    with execution_context(scope=scope):
        assert ReadFileTool()("a", filesystem=filesystem) == "one\ntwo\nthree\n"
        with pytest.raises(NotImplementedError):
            WriteTool()("a", "new", filesystem=filesystem)


def test_workspace_tool_permission_denial_is_live(backend_scope):
    filesystem, environment, _ = backend_scope
    scope = _scope(environment, filesystem, write=False)
    with execution_context(scope=scope):
        with pytest.raises(PermissionError):
            WriteTool()("a", "denied", filesystem=filesystem)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
async def test_same_library_reused_with_different_live_bindings(tmp_path):
    library = ToolLibrary("workspace", [WriteTool(), ReadFileTool()])
    for backend, guarantee in (
        (InMemoryWorkspaceBackend(), "atomic_compare"),
        (LocalWorkspaceBackend(tmp_path), "cooperative_compare"),
    ):
        async with await backend.open("files") as binding:
            environment = ExecutionEnvironment.from_binding(
                binding, write_guarantee=guarantee
            )
            with execution_context(scope=_scope(environment, binding.filesystem)):
                assert await library.arun(
                    "write", {"path": "a", "content": guarantee}
                ) == {"status": "completed"}
                assert await library.arun("read", {"path": "a"}) == guarantee
    assert (tmp_path / "a").read_text() == "cooperative_compare"


def test_workspace_tool_cannot_use_filesystem_outside_live_environment():
    first, other = InMemoryWorkspace("files"), InMemoryWorkspace("files")
    with execution_context(scope=_scope(ExecutionEnvironment(first), first)):
        with pytest.raises(PermissionError, match="live environment"):
            WriteTool()("a", "wrong resource", filesystem=other)


def test_selected_guarantee_is_recorded_in_preview(backend_scope):
    filesystem, environment, guarantee = backend_scope
    with execution_context(scope=_scope(environment, filesystem)):
        change = environment.workspace_editor().prepare_write("/a", "preview")
    assert change.write_guarantee == guarantee


def test_saved_proposal_cannot_apply_after_guarantee_changes(backend_scope):
    filesystem, environment, guarantee = backend_scope
    with execution_context(scope=_scope(environment, filesystem)):
        change = environment.workspace_editor(require_approval=False).prepare_write(
            "/a", "preview"
        )
    changed_environment = replace(
        environment,
        write_guarantee=(
            "cooperative_compare" if guarantee == "atomic_compare" else "atomic_compare"
        ),
    )
    tool = WriteTool()
    with execution_context(scope=_scope(changed_environment, filesystem)):
        with workspace_change_execution(tool, change):
            with pytest.raises(
                (PermissionError, NotImplementedError),
                match=r"guarantee|atomic_compare",
            ):
                tool("a", "preview", filesystem=filesystem)


def _response(name=None, arguments=None):
    response = ModelResponse()
    if name is None:
        response.set_response_type("text_generation")
        response.add("done")
        return response
    calls = ToolCallAggregator()
    calls.process(0, "call", name, msgspec_dumps(arguments))
    response.set_response_type("tool_call")
    response.add(calls)
    return response


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX local backend")
async def test_local_agent_approval_preview_decide_and_resume(tmp_path):
    binding = await LocalWorkspaceBackend(tmp_path).open("files")
    filesystem = binding.filesystem
    environment = ExecutionEnvironment.from_binding(
        binding, write_guarantee="cooperative_compare"
    )
    scope = _scope(environment, filesystem)
    checkpoints, approvals = InMemoryCheckpointStore(), InMemoryApprovalStore()
    current = Agent(
        name="editor",
        model=Mock(model_type="chat_completion"),
        tools=[WriteTool()],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(approvals, {"write": "v1"}, "p1"),
    )
    current.generator.forward = Mock(
        side_effect=[
            _response("write", {"path": "a", "content": "approved"}),
            _response(),
        ]
    )
    current.generator.aforward = AsyncMock(
        side_effect=[
            _response("write", {"path": "a", "content": "approved"}),
            _response(),
        ]
    )
    with pytest.raises(TaskPauseRequestedError):
        await current.acall("change", scope=scope)
    record = approvals.pending("editor", "thread", "run")[0]
    preview = await current.ainspect_approval_preview(
        "thread", "run", record.request_id
    )
    assert preview.write_guarantee == "cooperative_compare"
    assert "+approved" in preview.diff
    assert not (tmp_path / "a").exists()
    current.decide_approval(record.request_id, approved=True, decided_by="host")
    assert await current.acall("resume", scope=scope) == "done"
    with execution_context(scope=scope):
        assert filesystem.read_text("/a") == "approved"
    assert approvals.get("editor", record.request_id).status == "consumed"
    await binding.aclose()


@pytest.mark.parametrize("guarantee", ["atomic_compare", "cooperative_compare"])
def test_memory_pending_approval_invalidated_by_guarantee_change(guarantee):
    filesystem = InMemoryWorkspace("files", {"/a": b"old"})
    environment = ExecutionEnvironment(filesystem, write_guarantee=guarantee)
    scope = _scope(environment, filesystem)
    checkpoints, approvals = InMemoryCheckpointStore(), InMemoryApprovalStore()
    current = Agent(
        name="editor",
        model=Mock(model_type="chat_completion"),
        tools=[WriteTool()],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(approvals, {"write": "v1"}, "p1"),
    )
    current.generator.forward = Mock(
        return_value=_response("write", {"path": "a", "content": "new"})
    )
    with pytest.raises(TaskPauseRequestedError):
        current("change", scope=scope)
    record = approvals.pending("editor", "thread", "run")[0]
    current.decide_approval(record.request_id, approved=True, decided_by="host")
    changed = replace(
        environment,
        write_guarantee=(
            "cooperative_compare" if guarantee == "atomic_compare" else "atomic_compare"
        ),
    )
    with pytest.raises(TaskPauseRequestedError):
        current("resume", scope=replace(scope, environment=changed))
    with execution_context(scope=scope):
        assert filesystem.read_text("/a") == "old"
    assert approvals.get("editor", record.request_id).status == "approved"
