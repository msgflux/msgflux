import tracemalloc
from unittest.mock import Mock

import pytest

from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    LocalWorkspace,
    PermissionSet,
    execution_context,
)
from msgflux.tools.builtin import ReadFileTool, WriteTool


@pytest.mark.parametrize("kind", ["image", "edit"])
def test_sparse_real_file_rejected_without_full_allocation(tmp_path, kind):
    name = "large.png" if kind == "image" else "large.txt"
    with (tmp_path / name).open("wb") as stream:
        stream.truncate(1024 * 1024 * 1024)
    fs = LocalWorkspace("bounded", tmp_path)
    scope = ExecutionScope(
        environment=ExecutionEnvironment(
            fs, write_guarantee="cooperative_compare", max_edit_bytes=4096
        ),
        permissions=PermissionSet(
            resources=[
                fs.permission("/" + name, "filesystem.read"),
                fs.permission("/" + name, "filesystem.write"),
            ]
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
                    tool(name, filesystem=fs)
                else:
                    tool(name, "replacement", filesystem=fs)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    assert peak < 512 * 1024
    assert (tmp_path / name).stat().st_size == 1024 * 1024 * 1024


def test_disabled_vision_fails_before_reading():
    fs = Mock()
    with pytest.raises(ValueError, match="disabled"):
        ReadFileTool()("image.png", filesystem=fs)
    fs.read_prefix.assert_not_called()
    fs.read_bytes.assert_not_called()


def test_unicode_new_content_rejected_before_filesystem_read(tmp_path):
    from msgflux.runtime import WorkspaceEditor

    fs = LocalWorkspace("bounded", tmp_path)
    editor = WorkspaceEditor(fs, max_edit_bytes=4)
    with pytest.raises(ValueError, match="max_edit_bytes"):
        editor.prepare_write("/new", "😀😀")
    assert not (tmp_path / "new").exists()
