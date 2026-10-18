import asyncio
import errno
import os

import pytest

from msgflux.runtime import (
    AgentWorkspace,
    ExecutionEnvironment,
    ExecutionScope,
    PermissionSet,
    execution_context,
)
from msgflux.runtime.workspace.local import LocalWorkspace
from msgflux.tools.builtin import ReadFileTool


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX backend")


def _scope(fs):
    workspace = AgentWorkspace.from_environment(
        ExecutionEnvironment(fs),
        permissions=PermissionSet(
            resources=[
                fs.permission(path, "filesystem.read")
                for path in ("/", "/dir/missing/file", "/dir", "/oversized.txt")
            ]
            + [
                fs.permission("/", "filesystem.list"),
                fs.permission("/dir", "filesystem.list"),
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


def test_read_tool_preserves_byte_limit_guidance_when_close_fails(
    tmp_path, monkeypatch, caplog
):
    (tmp_path / "oversized.txt").write_text("too long\n")
    fs = LocalWorkspace("project", tmp_path)
    original_open, original_close = os.open, os.close
    state = {"file_open": False, "root_close_failed": False}

    def tracked_open(path, *args, **kwargs):
        fd = original_open(path, *args, **kwargs)
        if path == "oversized.txt":
            state["file_open"] = True
        return fd

    def tracked_close(fd):
        st = os.fstat(fd)
        if (
            state["file_open"]
            and not state["root_close_failed"]
            and (st.st_dev, st.st_ino) == (fs._root_stat.st_dev, fs._root_stat.st_ino)
        ):
            state["root_close_failed"] = True
            original_close(fd)
            raise OSError("injected root descriptor close failure")
        original_close(fd)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "close", tracked_close)

    with execution_context(scope=_scope(fs)):
        with pytest.raises(
            ValueError, match="single line exceeds the byte limit"
        ) as raised:
            ReadFileTool(max_text_bytes=1)("/oversized.txt")

    assert state["root_close_failed"]
    assert "No content was returned" in str(raised.value)
    assert any(
        "Workspace cleanup failed" in note
        for note in getattr(raised.value.__cause__, "__notes__", ())
    )
    assert "Workspace cleanup failed" in caplog.text


@pytest.mark.parametrize(
    "primary",
    [
        FileNotFoundError(errno.ENOENT, "No such file or directory", "missing"),
        asyncio.CancelledError("injected cancellation"),
    ],
    ids=("missing-parent", "cancellation"),
)
def test_nested_parent_open_failure_preserves_primary_and_closes_root(
    tmp_path, monkeypatch, primary
):
    (tmp_path / "dir").mkdir()
    fs = LocalWorkspace("project", tmp_path)
    directory_stat = os.stat(tmp_path / "dir")
    original_open, original_close = os.open, os.close
    state = {"primary": None, "parent_close_failed": False, "root_closed": False}

    def tracked_open(path, *args, **kwargs):
        if path == "missing":
            if isinstance(primary, FileNotFoundError):
                try:
                    return original_open(path, *args, **kwargs)
                except FileNotFoundError as error:
                    state["primary"] = error
                    raise
            state["primary"] = primary
            raise primary
        return original_open(path, *args, **kwargs)

    def tracked_close(fd):
        st = os.fstat(fd)
        identity = st.st_dev, st.st_ino
        if (
            state["primary"] is not None
            and not state["parent_close_failed"]
            and identity == (directory_stat.st_dev, directory_stat.st_ino)
        ):
            state["parent_close_failed"] = True
            original_close(fd)
            raise OSError("injected parent descriptor close failure")
        if state["primary"] is not None and identity == (
            fs._root_stat.st_dev,
            fs._root_stat.st_ino,
        ):
            state["root_closed"] = True
        original_close(fd)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "close", tracked_close)

    with execution_context(scope=_scope(fs)):
        with pytest.raises(type(primary)) as raised:
            fs.read_bytes("/dir/missing/file")

    assert state["primary"] is raised.value
    assert state["parent_close_failed"]
    assert state["root_closed"]
    assert any(
        "Workspace cleanup failed" in note
        for note in getattr(raised.value, "__notes__", ())
    )


def test_scandir_limit_error_survives_directory_close_failure(tmp_path, monkeypatch):
    directory = tmp_path / "dir"
    directory.mkdir()
    (directory / "a").write_text("a")
    (directory / "b").write_text("b")
    fs = LocalWorkspace("project", tmp_path)
    directory_stat = os.stat(directory)
    original_close = os.close
    state = {"directory_close_failed": False, "root_closed": False}

    def tracked_close(fd):
        st = os.fstat(fd)
        identity = st.st_dev, st.st_ino
        if not state["directory_close_failed"] and identity == (
            directory_stat.st_dev,
            directory_stat.st_ino,
        ):
            state["directory_close_failed"] = True
            original_close(fd)
            raise OSError("injected scan directory close failure")
        if state["directory_close_failed"] and identity == (
            fs._root_stat.st_dev,
            fs._root_stat.st_ino,
        ):
            state["root_closed"] = True
        original_close(fd)

    monkeypatch.setattr(os, "close", tracked_close)

    with execution_context(scope=_scope(fs)):
        with pytest.raises(ValueError, match="exceeds max_entries"):
            fs.scandir("/dir", max_entries=1)

    assert state["directory_close_failed"]
    assert state["root_closed"]


def test_duplicate_root_fd_in_cleanup_is_closed_once(tmp_path, monkeypatch):
    (tmp_path / "file").write_text("data")
    fs = LocalWorkspace("project", tmp_path)
    original_listdir, original_close = os.listdir, os.close
    state = {"after_list": False, "root_closes": 0}

    def tracked_listdir(fd):
        result = original_listdir(fd)
        state["after_list"] = True
        return result

    def tracked_close(fd):
        st = os.fstat(fd)
        if state["after_list"] and (st.st_dev, st.st_ino) == (
            fs._root_stat.st_dev,
            fs._root_stat.st_ino,
        ):
            state["root_closes"] += 1
        original_close(fd)

    monkeypatch.setattr(os, "listdir", tracked_listdir)
    monkeypatch.setattr(os, "close", tracked_close)

    with execution_context(scope=_scope(fs)):
        assert fs.listdir("/") == ("file",)

    assert state["root_closes"] == 1


def test_open_root_closes_duplicate_if_root_context_close_fails(tmp_path, monkeypatch):
    fs = LocalWorkspace("project", tmp_path)
    original_dup, original_close = os.dup, os.close
    state = {"duplicate": None, "injected": False, "duplicate_closed": False}

    def tracked_dup(fd):
        state["duplicate"] = original_dup(fd)
        return state["duplicate"]

    def tracked_close(fd):
        descriptor_stat = os.fstat(fd)
        if fd == state["duplicate"]:
            state["duplicate_closed"] = True
        elif (
            state["duplicate"] is not None
            and not state["injected"]
            and (descriptor_stat.st_dev, descriptor_stat.st_ino)
            == (fs._root_stat.st_dev, fs._root_stat.st_ino)
        ):
            state["injected"] = True
            original_close(fd)
            raise OSError("injected root context close failure")
        original_close(fd)

    monkeypatch.setattr(os, "dup", tracked_dup)
    monkeypatch.setattr(os, "close", tracked_close)

    with pytest.raises(OSError, match="injected root context close failure"):
        fs._open_root()

    assert state["injected"]
    assert state["duplicate_closed"]
    with pytest.raises(OSError, match="Bad file descriptor"):
        os.fstat(state["duplicate"])
