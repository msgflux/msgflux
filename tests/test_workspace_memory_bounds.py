from msgflux.runtime import AgentWorkspace
import tracemalloc
from unittest.mock import Mock

import pytest

from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    LocalWorkspace,
    PermissionSet,
    WorkspaceEditor,
    execution_context,
)
from msgflux.runtime.workspace import changes as workspace_changes
from msgflux.tools.builtin import EditTool, ReadFileTool, TextEdit, WriteTool


@pytest.mark.parametrize("kind", ["image", "edit"])
def test_sparse_real_file_rejected_without_full_allocation(tmp_path, kind):
    name = "large.png" if kind == "image" else "large.txt"
    with (tmp_path / name).open("wb") as stream:
        stream.truncate(1024 * 1024 * 1024)
    fs = LocalWorkspace("bounded", tmp_path)
    scope = ExecutionScope(
        workspace=AgentWorkspace.from_environment(
            ExecutionEnvironment(
                fs, write_guarantee="cooperative_compare", max_edit_bytes=4096
            ),
            permissions=PermissionSet(
                resources=[
                    fs.permission("/" + name, "filesystem.read"),
                    fs.permission("/" + name, "filesystem.write"),
                ]
            ),
        ),
    )
    tool = (
        ReadFileTool(supports_vision=True, max_image_bytes=4096)
        if kind == "image"
        else WriteTool()
    )
    with execution_context(scope=scope):
        tracemalloc.start()
        try:
            with pytest.raises(ValueError, match=r"limit|max_edit_bytes"):
                if kind == "image":
                    tool(name)
                else:
                    tool(name, "replacement")
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert peak < 512 * 1024
    assert (tmp_path / name).stat().st_size == 1024 * 1024 * 1024


def test_disabled_vision_fails_before_reading():
    workspace = Mock()
    with pytest.raises(ValueError, match="disabled"):
        ReadFileTool()("image.png", workspace=workspace)
    workspace.read_prefix.assert_not_called()
    workspace.read_bytes.assert_not_called()


def test_unicode_new_content_rejected_before_filesystem_read(tmp_path):
    from msgflux.runtime import WorkspaceEditor

    fs = LocalWorkspace("bounded", tmp_path)
    editor = WorkspaceEditor(fs, max_edit_bytes=4)
    with pytest.raises(ValueError, match="max_edit_bytes"):
        editor.prepare_write("/new", "😀😀")
    assert not (tmp_path / "new").exists()


def _bounded_scope(fs, max_edit_bytes=4):
    return ExecutionScope(
        workspace=AgentWorkspace.from_environment(
            ExecutionEnvironment(fs, max_edit_bytes=max_edit_bytes),
            permissions=PermissionSet(
                resources=[
                    fs.permission("/a", f"filesystem.{action}")
                    for action in ("read", "write", "delete")
                ]
            ),
        )
    )


def test_legacy_replace_all_rejects_oversize_before_replacement(monkeypatch):
    def trap(*args, **kwargs):
        raise AssertionError("replacement allocated before the size check")

    monkeypatch.setattr(workspace_changes, "_replace_all_text", trap)
    fs = InMemoryWorkspace("bounded", {"/a": b"aaaa"})
    editor = WorkspaceEditor(fs, max_edit_bytes=4)
    current_scope = _bounded_scope(fs)
    with execution_context(scope=current_scope):
        with pytest.raises(ValueError, match="max_edit_bytes"):
            editor.prepare_edit("/a", "a", "bb", replace_all=True)
        assert fs.read_text("/a") == "aaaa"


def test_list_replace_all_rejects_intermediate_growth_before_shrink(monkeypatch):
    def trap(*args, **kwargs):
        raise AssertionError("replacement allocated before the size check")

    monkeypatch.setattr(workspace_changes, "_replace_all_text", trap)
    fs = InMemoryWorkspace("bounded", {"/a": b"aaaa"})
    editor = WorkspaceEditor(fs, max_edit_bytes=4)
    current_scope = _bounded_scope(fs)
    with execution_context(scope=current_scope):
        with pytest.raises(ValueError, match="max_edit_bytes"):
            editor.prepare_edits(
                "/a",
                (
                    TextEdit(old="a", new="bb", replace_all=True),
                    TextEdit(old="bb", new="c", replace_all=True),
                ),
            )
        assert fs.read_text("/a") == "aaaa"


def test_replace_all_bounds_utf8_bytes_not_character_count(monkeypatch):
    def trap(*args, **kwargs):
        raise AssertionError("replacement allocated before the size check")

    monkeypatch.setattr(workspace_changes, "_replace_all_text", trap)
    fs = InMemoryWorkspace("bounded", {"/a": b"aaaa"})
    editor = WorkspaceEditor(fs, max_edit_bytes=4)
    current_scope = _bounded_scope(fs)
    with execution_context(scope=current_scope):
        # Four characters fit, but four two-byte replacements produce 8 bytes.
        with pytest.raises(ValueError, match="max_edit_bytes"):
            editor.prepare_edit("/a", "a", "é", replace_all=True)
        assert fs.read_text("/a") == "aaaa"


def test_single_edit_rejects_oversize_before_replacement(monkeypatch):
    def trap(*args, **kwargs):
        raise AssertionError("replacement allocated before the size check")

    monkeypatch.setattr(workspace_changes, "_replace_once_text", trap)
    fs = InMemoryWorkspace("bounded", {"/a": b"aXa"})
    editor = WorkspaceEditor(fs, max_edit_bytes=4)
    current_scope = _bounded_scope(fs)
    with execution_context(scope=current_scope):
        with pytest.raises(ValueError, match="max_edit_bytes"):
            editor.prepare_edit("/a", "X", "YYYY")
        assert fs.read_text("/a") == "aXa"


def test_each_intermediate_edit_is_bounded_before_later_shrink(monkeypatch):
    original = workspace_changes._replace_once_text

    def trap(text, start, old, new):
        if new == "abcdef":
            raise AssertionError("oversized intermediate allocated")
        return original(text, start, old, new)

    monkeypatch.setattr(workspace_changes, "_replace_once_text", trap)
    fs = InMemoryWorkspace("bounded", {"/a": b"a"})
    editor = WorkspaceEditor(fs, max_edit_bytes=4)
    current_scope = _bounded_scope(fs)
    with execution_context(scope=current_scope):
        with pytest.raises(ValueError, match="max_edit_bytes"):
            editor.prepare_edits(
                "/a",
                (
                    TextEdit(old="a", new="abcd"),
                    TextEdit(old="abcd", new="abcdef"),
                    TextEdit(old="abcdef", new="x"),
                ),
            )
        assert fs.read_text("/a") == "a"


@pytest.mark.parametrize(
    "edit",
    [
        TextEdit(old="ééé", new="x"),
        TextEdit(old="x", new="ééé"),
    ],
)
def test_each_edit_input_is_bounded_before_file_read(edit):
    fs = InMemoryWorkspace("bounded", {"/a": b"x"})
    editor = WorkspaceEditor(fs, max_edit_bytes=4)
    current_scope = _bounded_scope(fs)
    with execution_context(scope=current_scope):
        with pytest.raises(ValueError, match="max_edit_bytes"):
            editor.prepare_edits("/a", (edit,))
        assert fs.read_text("/a") == "x"


def test_edit_tool_rejects_oversize_intermediate_before_write():
    fs = InMemoryWorkspace("bounded", {"/a": b"aaaa"})
    current_scope = _bounded_scope(fs)
    with execution_context(scope=current_scope):
        with pytest.raises(ValueError, match="max_edit_bytes"):
            EditTool()(
                "a",
                edits=[
                    TextEdit(old="a", new="bb", replace_all=True),
                    TextEdit(old="bb", new="c", replace_all=True),
                ],
            )
        assert fs.read_text("/a") == "aaaa"
