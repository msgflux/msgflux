"""Contract tests for bounded workspace enumeration and prefix reads."""

from __future__ import annotations

import os
from pathlib import Path

import msgspec
import pytest

from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    PermissionSet,
    execution_context,
)
from msgflux.runtime.workspace import InMemoryWorkspace
from msgflux.runtime.workspace_contracts import WorkspaceEntry
from msgflux.runtime.workspace_local import LocalWorkspace, LocalWorkspaceBackend


def _filesystem(request, tmp_path: Path):
    if request.param == "memory":
        return InMemoryWorkspace(
            "workspace",
            {"/alpha.txt": b"alpha", "/nested/beta.txt": b"beta"},
        )
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "alpha.txt").write_bytes(b"alpha")
    (root / "nested").mkdir()
    (root / "nested/beta.txt").write_bytes(b"beta")
    return LocalWorkspace("workspace", root)


@pytest.fixture(params=["memory", "local"])
def filesystem(request, tmp_path):
    return _filesystem(request, tmp_path)


def _authorized(filesystem, *permissions):
    environment = ExecutionEnvironment(filesystem)
    scope = ExecutionScope(
        environment=environment,
        permissions=PermissionSet(
            resources=[
                filesystem.permission(path, action) for path, action in permissions
            ]
        ),
    )
    return execution_context(scope=scope)


def test_scandir_is_sorted_typed_and_serializable(filesystem):
    with _authorized(filesystem, ("/", "filesystem.list")):
        entries = filesystem.scandir("/")
    assert entries == (
        WorkspaceEntry(name="alpha.txt", kind="file"),
        WorkspaceEntry(name="nested", kind="directory"),
    )
    assert (
        msgspec.json.decode(
            msgspec.json.encode(entries), type=tuple[WorkspaceEntry, ...]
        )
        == entries
    )


def test_read_prefix_is_bounded_and_preserves_prefix(filesystem):
    with _authorized(filesystem, ("/alpha.txt", "filesystem.read")):
        assert filesystem.read_prefix("/alpha.txt", max_bytes=3) == b"alp"


@pytest.mark.parametrize("backend", ["memory", "local"])
def test_read_prefix_large_file_is_still_bounded(tmp_path, backend):
    if backend == "memory":
        filesystem = InMemoryWorkspace("large", {"/large": b"x" * 2_000_000})
    else:
        root = tmp_path / "large"
        root.mkdir()
        (root / "large").write_bytes(b"x" * 2_000_000)
        filesystem = LocalWorkspace("large", root)
    with _authorized(filesystem, ("/large", "filesystem.read")):
        assert filesystem.read_prefix("/large", max_bytes=17) == b"x" * 17


def test_missing_grants_are_rejected(filesystem):
    with _authorized(filesystem):
        with pytest.raises(PermissionError):
            filesystem.scandir("/")
        with pytest.raises(PermissionError):
            filesystem.read_prefix("/alpha.txt", max_bytes=2)


@pytest.mark.parametrize("method", ["scandir", "read_prefix"])
def test_path_traversal_is_rejected(filesystem, method):
    with _authorized(
        filesystem, ("/", "filesystem.list"), ("/alpha.txt", "filesystem.read")
    ):
        with pytest.raises(ValueError):
            if method == "scandir":
                filesystem.scandir("/../")
            else:
                filesystem.read_prefix("/../alpha.txt", max_bytes=1)


def test_invalid_limits_are_rejected(filesystem):
    with _authorized(
        filesystem, ("/", "filesystem.list"), ("/alpha.txt", "filesystem.read")
    ):
        with pytest.raises(ValueError):
            filesystem.scandir("/", max_entries=0)
        with pytest.raises(ValueError):
            filesystem.read_prefix("/alpha.txt", max_bytes=0)


@pytest.mark.parametrize("backend", ["memory", "local"])
def test_scandir_rejects_excess_entries(tmp_path, backend):
    if backend == "memory":
        filesystem = InMemoryWorkspace(
            "many", {f"/{index}": b"x" for index in range(4)}
        )
    else:
        root = tmp_path / "many"
        root.mkdir()
        for index in range(4):
            (root / str(index)).write_bytes(b"x")
        filesystem = LocalWorkspace("many", root)
    with _authorized(filesystem, ("/", "filesystem.list")):
        with pytest.raises(ValueError, match="max_entries"):
            filesystem.scandir("/", max_entries=2)


def test_local_unsafe_entries_are_reported_as_other(tmp_path):
    root = tmp_path / "unsafe"
    root.mkdir()
    (root / "regular").write_bytes(b"x")
    (root / "hardlink").write_bytes(b"x")
    os.link(root / "hardlink", root / "hardlink-alias")
    (root / "alias").symlink_to(root / "regular")
    os.mkfifo(root / "fifo")
    filesystem = LocalWorkspace("unsafe", root)
    with _authorized(filesystem, ("/", "filesystem.list")):
        entries = {entry.name: entry.kind for entry in filesystem.scandir("/")}
    assert entries["regular"] == "file"
    assert entries["hardlink"] == entries["hardlink-alias"] == "other"
    assert entries["alias"] == entries["fifo"] == "other"


def test_local_enumeration_and_reads_do_not_leak_descriptors(tmp_path):
    descriptors = Path("/proc/self/fd")
    if not descriptors.exists():
        pytest.skip("Linux descriptor accounting required")
    root = tmp_path / "fds"
    root.mkdir()
    (root / "item").write_bytes(b"value")
    filesystem = LocalWorkspace("fds", root)
    with _authorized(
        filesystem, ("/", "filesystem.list"), ("/item", "filesystem.read")
    ):
        before = len(list(descriptors.iterdir()))
        for _ in range(30):
            filesystem.scandir("/")
            filesystem.read_prefix("/item", max_bytes=2)
            filesystem.read_lines("/item")
        assert len(list(descriptors.iterdir())) == before


@pytest.mark.asyncio
async def test_closed_local_binding_rejects_operations(tmp_path):
    root = tmp_path / "bound"
    root.mkdir()
    (root / "item").write_bytes(b"value")
    backend = LocalWorkspaceBackend(root)
    binding = await backend.open("bound")
    filesystem = binding.filesystem
    environment = ExecutionEnvironment.from_binding(binding)
    permissions = PermissionSet(
        resources=[
            filesystem.permission("/", "filesystem.list"),
            filesystem.permission("/item", "filesystem.read"),
        ]
    )
    await binding.aclose()
    with execution_context(
        scope=ExecutionScope(environment=environment, permissions=permissions)
    ):
        with pytest.raises(PermissionError):
            filesystem.scandir("/")
        with pytest.raises(PermissionError):
            filesystem.read_prefix("/item", max_bytes=2)
