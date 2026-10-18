"""Missing paths have useful ENOENT diagnostics across workspace backends."""

import asyncio
import errno
import os

import pytest

from msgflux.nn import ToolLibrary
from msgflux.runtime import (
    AgentWorkspace,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    PermissionSet,
    execution_context,
)
from msgflux.tools.builtin import DeleteTool, GlobTool, LsTool, ReadFileTool
from msgflux.tools.runtime import ToolIntent


def _workspace(backend, tmp_path, *, files=None):
    if backend == "local":
        for path, data in (files or {}).items():
            host_path = tmp_path / path.lstrip("/")
            host_path.parent.mkdir(parents=True, exist_ok=True)
            host_path.write_bytes(data)
        return AgentWorkspace.local(tmp_path)
    filesystem = InMemoryWorkspace("missing-paths", files or {})
    permissions = PermissionSet(
        resources=[
            filesystem.permission(path, f"filesystem.{action}")
            for path in (
                "/",
                "/missing.txt",
                "/missing-dir",
                "/parent/missing",
                "/parent/missing/child.txt",
                "/file.txt",
                "/directory",
            )
            for action in ("read", "list", "delete", "write")
        ]
    )
    return AgentWorkspace.from_environment(
        ExecutionEnvironment(filesystem), permissions=permissions
    )


@pytest.mark.parametrize("backend", ["memory", "local"])
@pytest.mark.parametrize(
    "operation",
    ["read", "read_prefix", "read_lines", "list", "scandir", "unlink", "delete_token"],
)
@pytest.mark.asyncio
async def test_missing_targets_use_canonical_enoent(backend, operation, tmp_path):
    workspace = _workspace(backend, tmp_path)
    filesystem = workspace._environment.filesystem
    target = "/missing-dir" if operation in {"list", "scandir"} else "/missing.txt"
    try:
        with execution_context(scope=ExecutionScope(workspace=workspace)):
            with pytest.raises(FileNotFoundError) as raised:
                if operation == "read":
                    filesystem.read_bytes(target)
                elif operation == "read_prefix":
                    filesystem.read_prefix(target, max_bytes=10)
                elif operation == "read_lines":
                    filesystem.read_lines(target)
                elif operation == "list":
                    filesystem.listdir(target)
                elif operation == "scandir":
                    filesystem.scandir(target)
                elif operation == "unlink":
                    filesystem.unlink(target)
                else:
                    filesystem.deletion_directory_token(target)

        assert raised.value.errno == errno.ENOENT
        assert raised.value.filename
        assert "No such file or directory" in str(raised.value)
        assert target.rsplit("/", 1)[-1] in str(raised.value)
    finally:
        await workspace.aclose()


@pytest.mark.parametrize("backend", ["memory", "local"])
@pytest.mark.asyncio
async def test_missing_parent_path_is_reported_as_missing_target(backend, tmp_path):
    workspace = _workspace(backend, tmp_path)
    filesystem = workspace._environment.filesystem
    try:
        with execution_context(scope=ExecutionScope(workspace=workspace)):
            with pytest.raises(FileNotFoundError) as raised:
                filesystem.read_bytes("/parent/missing/child.txt")

        assert raised.value.errno == errno.ENOENT
        assert raised.value.filename
        assert raised.value.filename in str(raised.value)
    finally:
        await workspace.aclose()


@pytest.mark.parametrize("backend", ["memory", "local"])
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("read", {"path": "missing.txt"}),
        ("ls", {"path": "missing-dir"}),
        ("glob", {"pattern": "*", "path": "missing-dir"}),
        ("delete", {"path": "missing.txt"}),
    ],
)
@pytest.mark.asyncio
async def test_navigation_and_delete_tools_preserve_missing_path_diagnostics(
    backend, tool_name, arguments, tmp_path
):
    workspace = _workspace(backend, tmp_path)
    tools = [ReadFileTool(), LsTool(), GlobTool(), DeleteTool()]
    library = ToolLibrary("workspace-missing", tools)
    intent = ToolIntent(id="missing-target", name=tool_name, arguments=arguments)
    try:
        with execution_context(scope=ExecutionScope(workspace=workspace)):
            (outcome,) = await library.aexecute_intents((intent,))

        expected_path = (
            "/missing-dir" if tool_name in {"ls", "glob"} else "/missing.txt"
        )
        assert outcome.status == "execution_failed"
        assert outcome.error.code == "tool_execution_failed"
        assert outcome.error.details["exception_type"] == "FileNotFoundError"
        assert "No such file or directory" in outcome.error.message
        assert expected_path.rsplit("/", 1)[-1] in outcome.error.message
    finally:
        await workspace.aclose()


@pytest.mark.parametrize("backend", ["memory", "local"])
@pytest.mark.parametrize(
    ("operation", "path", "error_type", "error_number", "reason"),
    [
        (
            "read_directory",
            "/directory",
            IsADirectoryError,
            errno.EISDIR,
            "Is a directory",
        ),
        (
            "list_file",
            "/file.txt",
            NotADirectoryError,
            errno.ENOTDIR,
            "Not a directory",
        ),
        (
            "scandir_file",
            "/file.txt",
            NotADirectoryError,
            errno.ENOTDIR,
            "Not a directory",
        ),
    ],
)
@pytest.mark.asyncio
async def test_wrong_kind_errors_include_errno_and_path(
    backend, operation, path, error_type, error_number, reason, tmp_path
):
    workspace = _workspace(
        backend,
        tmp_path,
        files={"/directory/child.txt": b"child", "/file.txt": b"file"},
    )
    filesystem = workspace._environment.filesystem
    try:
        with execution_context(scope=ExecutionScope(workspace=workspace)):
            with pytest.raises(error_type) as raised:
                if operation == "read_directory":
                    filesystem.read_bytes(path)
                elif operation == "list_file":
                    filesystem.listdir(path)
                else:
                    filesystem.scandir(path)

        assert raised.value.errno == error_number
        assert raised.value.filename == path
        assert reason in str(raised.value)
    finally:
        await workspace.aclose()


@pytest.mark.parametrize("backend", ["memory", "local"])
@pytest.mark.asyncio
async def test_tool_errors_explain_existing_wrong_kind_targets(backend, tmp_path):
    workspace = _workspace(
        backend,
        tmp_path,
        files={"/directory/child.txt": b"child", "/file.txt": b"file"},
    )
    library = ToolLibrary(
        "wrong-kind",
        [ReadFileTool(), LsTool(), GlobTool()],
    )
    intents = (
        ToolIntent(id="read-directory", name="read", arguments={"path": "directory"}),
        ToolIntent(id="list-file", name="ls", arguments={"path": "file.txt"}),
        ToolIntent(
            id="glob-file",
            name="glob",
            arguments={"pattern": "*", "path": "file.txt"},
        ),
    )
    try:
        with execution_context(scope=ExecutionScope(workspace=workspace)):
            outcomes = await library.aexecute_intents(intents)

        assert [outcome.status for outcome in outcomes] == [
            "execution_failed",
            "execution_failed",
            "execution_failed",
        ]
        assert [outcome.error.details["exception_type"] for outcome in outcomes] == [
            "IsADirectoryError",
            "NotADirectoryError",
            "NotADirectoryError",
        ]
        assert "Is a directory" in outcomes[0].error.message
        assert "Not a directory" in outcomes[1].error.message
        assert "Not a directory" in outcomes[2].error.message
        assert "/directory" in outcomes[0].error.message
        assert "/file.txt" in outcomes[1].error.message
        assert "/file.txt" in outcomes[2].error.message
    finally:
        await workspace.aclose()


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "mkfifo"), reason="FIFO requires POSIX"
)
@pytest.mark.asyncio
async def test_read_tool_reports_fifo_as_unsupported_special_file(tmp_path):
    os.mkfifo(tmp_path / "pipe")
    workspace = AgentWorkspace.local(tmp_path)
    library = ToolLibrary("fifo-read", [ReadFileTool()])
    intent = ToolIntent(id="read-fifo", name="read", arguments={"path": "pipe"})
    try:
        with execution_context(scope=ExecutionScope(workspace=workspace)):
            (outcome,) = await asyncio.wait_for(
                library.aexecute_intents((intent,)), timeout=5
            )

        assert outcome.status == "execution_failed"
        assert outcome.error.code == "tool_execution_failed"
        assert outcome.error.details["exception_type"] == "PermissionError"
        assert outcome.error.message == (
            "Non-regular workspace files are not supported: /pipe"
        )
    finally:
        await workspace.aclose()


@pytest.mark.asyncio
async def test_missing_path_diagnostics_do_not_swallow_cancellation(tmp_path):
    workspace = _workspace("memory", tmp_path)
    filesystem = workspace._environment.filesystem
    cancellation = asyncio.CancelledError("read cancelled")

    def cancelled(*_args):
        raise cancellation

    try:
        filesystem._operate = cancelled
        with execution_context(scope=ExecutionScope(workspace=workspace)):
            with pytest.raises(asyncio.CancelledError) as raised:
                filesystem.read_bytes("/missing.txt")
        assert raised.value is cancellation
    finally:
        await workspace.aclose()
