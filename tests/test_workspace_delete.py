import os

import pytest

from msgflux.nn import ToolLibrary
from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    LocalWorkspace,
    PermissionSet,
    WorkspaceConflictError,
    execution_context,
)
from msgflux.tools.builtin import DeleteTool
from msgflux.tools.workspace_changes import workspace_change_execution


@pytest.fixture(params=["memory", "local"])
def workspace(request, tmp_path):
    files = {"/dir/a": b"old\n", "/dir/binary": b"\xff"}
    if request.param == "local":
        if os.name != "posix":
            pytest.skip("POSIX backend")
        (tmp_path / "dir").mkdir()
        for path, data in files.items():
            (tmp_path / path.lstrip("/")).write_bytes(data)
        fs = LocalWorkspace("delete", tmp_path)
        guarantee = "cooperative_compare"
    else:
        fs = InMemoryWorkspace("delete", files)
        guarantee = "atomic_compare"
    return fs, ExecutionEnvironment(fs, write_guarantee=guarantee)


def scope(workspace, *, delete=True):
    fs, environment = workspace
    actions = ["read", "write"] + (["delete"] if delete else [])
    return ExecutionScope(
        environment=environment,
        permissions=PermissionSet(
            resources=[
                fs.permission(path, f"filesystem.{action}")
                for path in ("/dir", "/dir/a", "/dir/binary")
                for action in actions
            ]
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_delete_is_backend_neutral(workspace, asynchronous):
    fs, _ = workspace
    tool = DeleteTool(cwd="/dir")
    definition = ToolLibrary("files", [tool]).get_tool_definition("delete")
    assert set(definition.input_schema["properties"]) == {"path"}
    with execution_context(scope=scope(workspace)):
        preview = tool.prepare_workspace_change({"path": "a"}, fs)
        assert preview.after is None and "-old" in preview.diff
        assert fs.read_text("/dir/a") == "old\n"
        result = (
            await tool.acall("a", filesystem=fs)
            if asynchronous
            else tool("a", filesystem=fs)
        )
        assert result == {"status": "completed"}
        with pytest.raises(FileNotFoundError):
            fs.read_text("/dir/a")


def test_delete_denied_and_stale_proposals_leave_file_intact(workspace):
    fs, _ = workspace
    tool = DeleteTool()
    with execution_context(scope=scope(workspace, delete=False)):
        with pytest.raises(PermissionError):
            tool("/dir/a", filesystem=fs)
        assert fs.read_text("/dir/a") == "old\n"
    with execution_context(scope=scope(workspace)):
        preview = tool.prepare_workspace_change({"path": "/dir/a"}, fs)
        fs.write_text("/dir/a", "concurrent")
        with workspace_change_execution(tool, preview):
            with pytest.raises(WorkspaceConflictError):
                tool("/dir/a", filesystem=fs)
        assert fs.read_text("/dir/a") == "concurrent"


def test_delete_rejects_directories_binary_and_traversal(workspace):
    fs, _ = workspace
    tool = DeleteTool()
    with execution_context(scope=scope(workspace)):
        with pytest.raises((IsADirectoryError, PermissionError)):
            tool("/dir", filesystem=fs)
        with pytest.raises(UnicodeDecodeError):
            tool("/dir/binary", filesystem=fs)
        with pytest.raises(ValueError):
            tool("/dir/../a", filesystem=fs)
        assert fs.read_bytes("/dir/binary") == b"\xff"
