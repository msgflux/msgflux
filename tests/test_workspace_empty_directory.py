from msgflux.runtime import AgentWorkspace
import os

import msgspec
import pytest

from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    LocalWorkspace,
    PermissionSet,
    PreparedFileChange,
    WorkspaceConflictError,
    execution_context,
    get_execution_scope,
)
from msgflux.tools.builtin import ApplyPatchTool, DeleteTool
from msgflux.tools.workspace_changes import workspace_change_execution


@pytest.fixture(params=["memory", "local"])
def workspace(request, tmp_path):
    if request.param == "local":
        if os.name != "posix":
            pytest.skip("POSIX backend")
        fs = LocalWorkspace("directory", tmp_path)
        guarantee = "cooperative_compare"
    else:
        fs = InMemoryWorkspace("directory")
        guarantee = "atomic_compare"
    environment = ExecutionEnvironment(fs, write_guarantee=guarantee)
    scope = ExecutionScope(
        workspace=AgentWorkspace.from_environment(
            environment,
            permissions=PermissionSet(
                resources=[
                    fs.permission(path, f"filesystem.{action}")
                    for path in ("/", "/empty", "/empty/item")
                    for action in ("list", "mkdir", "read", "write", "delete")
                ]
            ),
        ),
    )
    with execution_context(scope=scope):
        fs.mkdir("/empty")
    return fs, scope


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_empty_directory_review_and_delete(workspace, asynchronous):
    fs, scope = workspace
    tool = DeleteTool()
    with execution_context(scope=scope):
        preview = tool.prepare_workspace_change(
            {"path": "/empty"}, get_execution_scope().workspace
        )
        assert preview.target_kind == "empty_directory"
        assert preview.operation == "delete"
        assert preview.before is None and preview.after is None
        assert preview.directory_token and "/empty/" in preview.diff
        assert (
            msgspec.json.decode(msgspec.json.encode(preview), type=PreparedFileChange)
            == preview
        )
        with workspace_change_execution(tool, preview):
            result = await tool.acall("/empty") if asynchronous else tool("/empty")
        assert result == {"status": "completed"}
        assert fs.listdir("/") == ()


@pytest.mark.parametrize("change", ["content", "replacement", "file"])
def test_directory_review_rejects_changed_target(workspace, change):
    fs, scope = workspace
    tool = DeleteTool()
    with execution_context(scope=scope):
        preview = tool.prepare_workspace_change(
            {"path": "/empty"}, get_execution_scope().workspace
        )
        if change == "content":
            fs.write_text("/empty/item", "keep")
        else:
            fs.checked_rmdir(
                "/empty",
                expected=preview.directory_token,
                guarantee=scope.workspace._environment.write_guarantee,
            )
            if change == "replacement":
                fs.mkdir("/empty")
            else:
                fs.write_text("/empty", "keep")
        with workspace_change_execution(tool, preview):
            with pytest.raises((WorkspaceConflictError, OSError)):
                tool("/empty")
        assert "empty" in fs.listdir("/")


def test_empty_directory_requires_list_and_delete_not_read(workspace):
    fs, original = workspace
    for actions in [("delete",), ("list",), ("delete", "list")]:
        scope = ExecutionScope(
            workspace=AgentWorkspace.from_environment(
                original.workspace._environment,
                permissions=PermissionSet(
                    resources=[
                        fs.permission("/empty", f"filesystem.{action}")
                        for action in actions
                    ]
                ),
            ),
        )
        with execution_context(scope=scope):
            if len(actions) == 1:
                with pytest.raises(PermissionError):
                    DeleteTool()("/empty")
            else:
                assert DeleteTool()("/empty")["status"] == "completed"


def test_root_and_apply_patch_directory_deletion_are_rejected(workspace):
    fs, scope = workspace
    with execution_context(scope=scope):
        with pytest.raises(PermissionError, match="root"):
            DeleteTool()("/")
        with pytest.raises((IsADirectoryError, PermissionError)):
            ApplyPatchTool().prepare_workspace_change(
                {"operation": "delete", "path": "/empty", "diff": None},
                get_execution_scope().workspace,
            )
        assert fs.listdir("/empty") == ()


def test_local_symlink_directory_is_not_followed(tmp_path):
    (tmp_path / "target").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "target", target_is_directory=True)
    fs = LocalWorkspace("links", tmp_path)
    scope = ExecutionScope(
        workspace=AgentWorkspace.from_environment(
            ExecutionEnvironment(fs, write_guarantee="cooperative_compare"),
            permissions=PermissionSet(
                resources=[
                    fs.permission("/alias", f"filesystem.{action}")
                    for action in ("delete", "list")
                ]
            ),
        ),
    )
    with execution_context(scope=scope), pytest.raises(PermissionError):
        DeleteTool()("/alias")
    assert (tmp_path / "target").is_dir()
    assert (tmp_path / "alias").is_symlink()


def test_local_rmdir_refuses_content_added_after_last_check(tmp_path, monkeypatch):
    (tmp_path / "empty").mkdir()
    fs = LocalWorkspace("race", tmp_path)
    scope = ExecutionScope(
        workspace=AgentWorkspace.from_environment(
            ExecutionEnvironment(fs, write_guarantee="cooperative_compare"),
            permissions=PermissionSet(
                resources=[
                    fs.permission("/empty", f"filesystem.{action}")
                    for action in ("delete", "list")
                ]
            ),
        ),
    )
    original = os.rmdir

    def insert_then_remove(path, *, dir_fd):
        (tmp_path / "empty" / "new").write_text("keep")
        return original(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "rmdir", insert_then_remove)
    with execution_context(scope=scope), pytest.raises(OSError):
        DeleteTool()("/empty")
    assert (tmp_path / "empty" / "new").read_text() == "keep"
