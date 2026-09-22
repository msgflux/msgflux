"""Workspace navigation tools across authorized filesystem backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from msgflux.nn import ToolLibrary
from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    PermissionSet,
    execution_context,
)
from msgflux.runtime.workspace import InMemoryWorkspace
from msgflux.runtime.workspace_local import LocalWorkspace
from msgflux.tools.builtin.workspace_query import GlobTool, GrepTool, LsTool


def _scope(filesystem, paths):
    resources = [filesystem.permission(path, action) for path, action in paths]
    return execution_context(
        scope=ExecutionScope(
            environment=ExecutionEnvironment(filesystem),
            permissions=PermissionSet(resources=resources),
        )
    )


def _fixture(request, tmp_path: Path):
    files = {
        "/.gitignore": b"src/ignored/*\n",
        "/src/main.py": b"needle = 1\nplain\n",
        "/src/readme.txt": b"needle text\n",
        "/src/ignored/drop.py": b"needle hidden\n",
        "/src/ignored/.gitignore": b"!keep.py\n",
        "/src/ignored/keep.py": b"needle keep\n",
        "/.git/config": b"needle should not appear\n",
    }
    if request.param == "memory":
        return InMemoryWorkspace("query", files)
    root = tmp_path / "query"
    root.mkdir()
    for name, data in files.items():
        target = root / name.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return LocalWorkspace("query", root)


@pytest.fixture(params=["memory", "local"])
def filesystem(request, tmp_path):
    return _fixture(request, tmp_path)


def _all_grants(filesystem):
    return _scope(
        filesystem,
        [
            ("/", "filesystem.list"),
            ("/.git", "filesystem.list"),
            ("/.git/config", "filesystem.read"),
            ("/.gitignore", "filesystem.read"),
            ("/src", "filesystem.list"),
            ("/src/main.py", "filesystem.read"),
            ("/src/readme.txt", "filesystem.read"),
            ("/src/ignored", "filesystem.list"),
            ("/src/ignored/.gitignore", "filesystem.read"),
            ("/src/ignored/drop.py", "filesystem.read"),
            ("/src/ignored/keep.py", "filesystem.read"),
        ],
    )


def test_schemas_expose_only_visible_query_arguments():
    library = ToolLibrary("query", [LsTool(), GlobTool(), GrepTool()])
    assert set(library.get_tool_definition("ls").input_schema["properties"]) == {"path"}
    assert set(library.get_tool_definition("glob").input_schema["properties"]) == {
        "pattern",
        "path",
    }
    assert set(library.get_tool_definition("grep").input_schema["properties"]) == {
        "pattern",
        "path",
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("tool_cls", [LsTool])
@pytest.mark.asyncio
async def test_ls_is_structured_sorted_and_async(filesystem, tool_cls, asynchronous):
    tool = tool_cls()
    with _all_grants(filesystem):
        result = (
            await tool.acall("/src", filesystem=filesystem)
            if asynchronous
            else tool("/src", filesystem=filesystem)
        )
    assert [entry["name"] for entry in result["entries"]] == [
        "ignored",
        "main.py",
        "readme.txt",
    ]
    assert result["path"] == "/src"


@pytest.mark.asyncio
async def test_glob_supports_recursive_segments_and_prunes_git(filesystem):
    tool = GlobTool(max_depth=8, max_nodes=100)
    with _all_grants(filesystem):
        result = await tool.acall("**/*.py", "/", filesystem=filesystem)
    assert [item["path"] for item in result["matches"]] == [
        "/src/ignored/keep.py",
        "/src/main.py",
    ]
    assert all("/.git/" not in item["path"] for item in result["matches"])


@pytest.mark.asyncio
async def test_grep_honors_nested_ignore_negation_and_returns_bounded_matches(
    filesystem,
):
    tool = GrepTool(max_depth=8, max_nodes=100, max_results=10, max_file_bytes=200)
    with _all_grants(filesystem):
        result = await tool.acall("needle", "/", filesystem=filesystem)
    paths = [item["path"] for item in result["matches"]]
    assert paths == ["/src/ignored/keep.py", "/src/main.py", "/src/readme.txt"]
    assert "/src/ignored/drop.py" not in paths
    assert "/.git/config" not in paths


@pytest.mark.parametrize(
    "tool_cls, kwargs", [(GlobTool, {"max_nodes": 1}), (GrepTool, {"max_nodes": 1})]
)
def test_query_budget_exhaustion_fails_closed(filesystem, tool_cls, kwargs):
    tool = tool_cls(**kwargs)
    with _all_grants(filesystem), pytest.raises(ValueError):
        tool("needle" if tool_cls is GrepTool else "**", "/", filesystem=filesystem)


def test_grep_reports_binary_and_oversized_without_reading_unbounded_data(tmp_path):
    root = tmp_path / "bounded"
    root.mkdir()
    (root / "binary").write_bytes(b"needle\x00needle")
    (root / "large").write_bytes(b"needle\n" * 100)
    filesystem = LocalWorkspace("bounded", root)
    tool = GrepTool(max_file_bytes=16)
    with _scope(
        filesystem,
        [
            ("/", "filesystem.list"),
            ("/binary", "filesystem.read"),
            ("/large", "filesystem.read"),
        ],
    ):
        result = tool("needle", "/", filesystem=filesystem)
    assert result["matches"] == []
    assert {item["reason"] for item in result["skipped"]} == {"binary", "oversized"}


def test_query_denied_io_is_not_silently_ignored(filesystem):
    with _scope(filesystem, [("/", "filesystem.list")]), pytest.raises(PermissionError):
        GrepTool()("needle", "/", filesystem=filesystem)


@pytest.mark.parametrize(
    "pattern, expected",
    [("src/**/*.py", True), ("src/*.py", False), ("**/a.[pt][xy]", True)],
)
def test_glob_segments(pattern, expected):
    from msgflux.tools.builtin.workspace_query import _glob_match

    assert _glob_match(pattern, "src/nested/a.py") is expected
    assert _glob_match("src/**/*.py", "src/a.py")


@pytest.mark.parametrize("pattern", ["", "a\\b", "x" * 4097])
def test_invalid_glob_is_rejected_before_filesystem_access(pattern):
    with pytest.raises(ValueError):
        GlobTool()(pattern, filesystem=None)


def test_search_caps_records_including_skipped_metadata():
    fs = InMemoryWorkspace("caps", {f"/{index:03}": b"\x00" for index in range(100)})
    grants = [("/", "filesystem.list")] + [
        (f"/{index:03}", "filesystem.read") for index in range(100)
    ]
    with _scope(fs, grants):
        result = GrepTool(max_output_bytes=100)("a", filesystem=fs)
    assert result["truncated"]
    assert 0 < len(result["skipped"]) < 100


def test_grep_caps_unicode_match_bytes_and_marks_long_lines():
    fs = InMemoryWorkspace("caps", {"/a": ("á" * 5000 + "\n").encode()})
    with _scope(fs, [("/", "filesystem.list"), ("/a", "filesystem.read")]):
        result = GrepTool(max_output_bytes=100)("á", filesystem=fs)
        assert result["truncated"] and result["matches"] == []
        result = GrepTool()("á", filesystem=fs)
        assert len(result["matches"][0]["text"]) == 4096
        assert result["matches"][0]["truncated"]


def test_depth_and_elapsed_limits_are_not_silent(monkeypatch):
    fs = InMemoryWorkspace("limits", {"/a/b": b"text"})
    with _scope(fs, [("/", "filesystem.list")]):
        with pytest.raises(ValueError, match="max_depth"):
            GlobTool(max_depth=0)("**", filesystem=fs)
        ticks = iter([0.0, 10.0])
        monkeypatch.setattr(
            "msgflux.tools.builtin.workspace_query.time.monotonic", lambda: next(ticks)
        )
        with pytest.raises(TimeoutError):
            GlobTool(max_seconds=1)("**", filesystem=fs)


def test_ignored_directory_is_pruned_without_grants_to_its_contents():
    fs = InMemoryWorkspace(
        "ignore", {"/.gitignore": b"hidden/\n", "/hidden/secret": b"secret"}
    )
    with _scope(fs, [("/", "filesystem.list"), ("/.gitignore", "filesystem.read")]):
        assert GrepTool()("secret", filesystem=fs)["matches"] == []


def test_grep_regex_timeout_is_reported(monkeypatch):
    import regex

    class SlowPattern:
        def search(self, *args, **kwargs):
            assert kwargs["timeout"] > 0
            raise TimeoutError

    monkeypatch.setattr(regex, "compile", lambda pattern: SlowPattern())
    fs = InMemoryWorkspace("regex", {"/a": b"text"})
    with _scope(fs, [("/", "filesystem.list"), ("/a", "filesystem.read")]):
        with pytest.raises(ValueError, match="time limit"):
            GrepTool()("a", filesystem=fs)
