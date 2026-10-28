"""Missing workspace targets retain useful OS error details."""

import errno

import pytest

from msgflux.nn import ToolLibrary
from msgflux.runtime import (
    AgentWorkspace,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    PermissionSet,
    WorkspaceEditor,
    execution_context,
)
from msgflux.tools.builtin import ApplyPatchTool, TextEdit
from msgflux.tools.runtime import ToolIntent


@pytest.mark.parametrize(
    "prepare",
    [
        lambda editor: editor.prepare_edit("/missing.txt", "old", "new"),
        lambda editor: editor.prepare_edits(
            "/missing.txt", [TextEdit(old="old", new="new", replace_all=True)]
        ),
        lambda editor: editor.prepare_transform("/missing.txt", str.upper),
        lambda editor: editor.prepare_delete("/missing.txt"),
    ],
    ids=["edit", "edits", "transform", "delete"],
)
def test_missing_change_target_has_enoent_details(prepare):
    filesystem = InMemoryWorkspace("missing-target", {})
    editor = WorkspaceEditor(filesystem)
    scope = ExecutionScope(
        namespace="workspace",
        thread_id="thread",
        run_id="run",
        principal="user",
        workspace=AgentWorkspace.from_environment(
            ExecutionEnvironment(filesystem),
            permissions=PermissionSet(
                resources=[
                    filesystem.permission("/missing.txt", f"filesystem.{action}")
                    for action in ("read", "write", "delete")
                ]
            ),
        ),
    )

    with execution_context(scope=scope), pytest.raises(FileNotFoundError) as raised:
        prepare(editor)

    assert raised.value.errno == errno.ENOENT
    assert raised.value.filename == "/missing.txt"
    assert "No such file or directory" in str(raised.value)
    with execution_context(scope=scope), pytest.raises(FileNotFoundError):
        filesystem.read_bytes("/missing.txt")


@pytest.mark.asyncio
async def test_no_approval_apply_patch_error_keeps_standard_missing_path(tmp_path):
    workspace = AgentWorkspace.local(tmp_path, approval_policy="never")
    library = ToolLibrary("patch-audit", [ApplyPatchTool()])
    intent = ToolIntent(
        id="missing-update",
        name="apply_patch",
        arguments={
            "operation": "update",
            "path": "missing.txt",
            "diff": "@@\n-old\n+new",
        },
    )
    try:
        with execution_context(scope=ExecutionScope(workspace=workspace)):
            (outcome,) = await library.aexecute_intents((intent,))

        assert outcome.status == "execution_failed"
        assert outcome.error.code == "tool_execution_failed"
        assert outcome.error.message == (
            "[Errno 2] No such file or directory: '/missing.txt'"
        )
        assert outcome.error.details["exception_type"] == "FileNotFoundError"
    finally:
        await workspace.aclose()
