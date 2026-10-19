import asyncio
import os

import pytest

from msgflux.runtime import (
    AgentWorkspace,
    ExecutionEnvironment,
    ExecutionScope,
    PermissionSet,
    WorkspaceEditor,
    execution_context,
)
from msgflux.runtime.workspace.local import LocalWorkspace


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX backend")


def _scope(fs):
    workspace = AgentWorkspace.from_environment(
        ExecutionEnvironment(fs),
        permissions=PermissionSet(
            resources=[
                fs.permission(path, f"filesystem.{action}")
                for path in ("/a", "/dir", "/dir/a", "/new")
                for action in ("read", "write", "delete", "mkdir", "list")
            ]
        ),
    )
    return ExecutionScope(
        namespace="local",
        thread_id="thread",
        run_id="run",
        principal="user",
        workspace=workspace,
    )


def _fail_root_close_after_replace(fs, monkeypatch, replace_impl):
    state = {"replace_attempted": False, "raised": False}
    real_close = os.close

    def replace(*args, **kwargs):
        state["replace_attempted"] = True
        return replace_impl(*args, **kwargs)

    def close(fd):
        try:
            descriptor_stat = os.fstat(fd)
        except OSError:
            raise
        if (
            state["replace_attempted"]
            and not state["raised"]
            and (descriptor_stat.st_dev, descriptor_stat.st_ino)
            == (fs._root_stat.st_dev, fs._root_stat.st_ino)
        ):
            state["raised"] = True
            real_close(fd)
            raise OSError("injected descriptor cleanup failure")
        real_close(fd)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "close", close)
    return state


def _fail_root_close_after_syscall(fs, monkeypatch, syscall_name):
    state = {"committed": False, "raised": False}
    real_syscall = getattr(os, syscall_name)
    real_close = os.close

    def syscall(*args, **kwargs):
        result = real_syscall(*args, **kwargs)
        state["committed"] = True
        return result

    def close(fd):
        descriptor_stat = os.fstat(fd)
        if (
            state["committed"]
            and not state["raised"]
            and (descriptor_stat.st_dev, descriptor_stat.st_ino)
            == (fs._root_stat.st_dev, fs._root_stat.st_ino)
        ):
            state["raised"] = True
            real_close(fd)
            raise OSError("injected descriptor cleanup failure")
        real_close(fd)

    monkeypatch.setattr(os, syscall_name, syscall)
    monkeypatch.setattr(os, "close", close)
    return state


def _fail_nested_parent_close_after_replace(fs, monkeypatch):
    state = {"committed": False, "parent_close_failed": False, "root_closed": False}
    real_replace = os.replace
    real_close = os.close
    parent_stat = os.stat(os.path.join(fs.host_root, "dir"))

    def replace(*args, **kwargs):
        result = real_replace(*args, **kwargs)
        state["committed"] = True
        return result

    def close(fd):
        descriptor_stat = os.fstat(fd)
        identity = descriptor_stat.st_dev, descriptor_stat.st_ino
        if state["committed"] and identity == (parent_stat.st_dev, parent_stat.st_ino):
            state["parent_close_failed"] = True
            real_close(fd)
            raise OSError("injected nested parent close failure")
        if state["committed"] and identity == (
            fs._root_stat.st_dev,
            fs._root_stat.st_ino,
        ):
            state["root_closed"] = True
        real_close(fd)

    monkeypatch.setattr(os, "replace", replace)
    monkeypatch.setattr(os, "close", close)
    return state


def test_committed_replace_reports_success_when_root_close_fails(
    tmp_path, monkeypatch, caplog
):
    (tmp_path / "a").write_text("before")
    fs = LocalWorkspace("project", tmp_path)
    state = _fail_root_close_after_replace(fs, monkeypatch, os.replace)
    editor = WorkspaceEditor(
        fs, require_approval=False, write_guarantee="cooperative_compare"
    )

    with execution_context(scope=_scope(fs)):
        change = editor.prepare_write("/a", "after")
        editor.apply(change)

    assert state["raised"]
    assert (tmp_path / "a").read_text() == "after"
    assert "cleanup failed after a committed mutation" in caplog.text


def test_replace_failure_remains_primary_when_root_close_also_fails(
    tmp_path, monkeypatch, caplog
):
    (tmp_path / "a").write_text("before")
    fs = LocalWorkspace("project", tmp_path)
    primary = OSError("injected replace failure")

    def fail_replace(*args, **kwargs):
        raise primary

    state = _fail_root_close_after_replace(fs, monkeypatch, fail_replace)
    editor = WorkspaceEditor(
        fs, require_approval=False, write_guarantee="cooperative_compare"
    )
    with execution_context(scope=_scope(fs)):
        change = editor.prepare_write("/a", "after")
        with pytest.raises(OSError) as raised:
            editor.apply(change)

    assert raised.value is primary
    assert state["raised"]
    assert (tmp_path / "a").read_text() == "before"
    assert any(
        "Workspace cleanup failed" in note for note in getattr(primary, "__notes__", ())
    )
    assert "Workspace cleanup failed" in caplog.text


def test_cancelled_replace_remains_primary_when_root_close_also_fails(
    tmp_path, monkeypatch
):
    (tmp_path / "a").write_text("before")
    fs = LocalWorkspace("project", tmp_path)
    primary = asyncio.CancelledError("injected cancellation")

    def cancel_replace(*args, **kwargs):
        raise primary

    state = _fail_root_close_after_replace(fs, monkeypatch, cancel_replace)
    editor = WorkspaceEditor(
        fs, require_approval=False, write_guarantee="cooperative_compare"
    )
    with execution_context(scope=_scope(fs)):
        change = editor.prepare_write("/a", "after")
        with pytest.raises(asyncio.CancelledError) as raised:
            editor.apply(change)

    assert raised.value is primary
    assert state["raised"]
    assert (tmp_path / "a").read_text() == "before"
    assert any(
        "Workspace cleanup failed" in note for note in getattr(primary, "__notes__", ())
    )


def test_nested_parent_close_failure_does_not_skip_root_close(tmp_path, monkeypatch):
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir/a").write_text("before")
    fs = LocalWorkspace("project", tmp_path)
    state = _fail_nested_parent_close_after_replace(fs, monkeypatch)
    editor = WorkspaceEditor(
        fs, require_approval=False, write_guarantee="cooperative_compare"
    )

    with execution_context(scope=_scope(fs)):
        change = editor.prepare_write("/dir/a", "after")
        editor.apply(change)

    assert state["committed"]
    assert state["parent_close_failed"]
    assert state["root_closed"]
    assert (tmp_path / "dir/a").read_text() == "after"


@pytest.mark.parametrize(
    ("path", "syscall", "operation"),
    [("/a", "unlink", "unlink"), ("/new", "mkdir", "mkdir")],
)
def test_direct_file_mutations_report_success_after_close_failure(
    tmp_path, monkeypatch, caplog, path, syscall, operation
):
    if operation == "unlink":
        (tmp_path / "a").write_text("before")
    fs = LocalWorkspace("project", tmp_path)
    state = _fail_root_close_after_syscall(fs, monkeypatch, syscall)

    with execution_context(scope=_scope(fs)):
        getattr(fs, operation)(path)

    assert state["raised"]
    assert (tmp_path / path.lstrip("/")).exists() is (operation == "mkdir")
    assert "cleanup failed after a committed mutation" in caplog.text


def test_empty_directory_removal_reports_success_after_close_failure(
    tmp_path, monkeypatch, caplog
):
    (tmp_path / "dir").mkdir()
    fs = LocalWorkspace("project", tmp_path)
    state = _fail_root_close_after_syscall(fs, monkeypatch, "rmdir")
    editor = WorkspaceEditor(
        fs, require_approval=False, write_guarantee="cooperative_compare"
    )

    with execution_context(scope=_scope(fs)):
        change = editor.prepare_delete_target("/dir")
        editor.apply(change)

    assert state["raised"]
    assert not (tmp_path / "dir").exists()
    assert "cleanup failed after a committed mutation" in caplog.text
