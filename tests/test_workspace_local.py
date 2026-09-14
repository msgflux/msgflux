import os
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Event

import msgspec
import pytest

from msgflux.runtime import (
    AbortSignal,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryApprovalStore,
    PermissionSet,
    WorkspaceConflictError,
    WorkspaceEditor,
    execution_context,
)
from msgflux.exceptions import AbortRequestedError
from msgflux.runtime.workspace_local import LocalWorkspace, LocalWorkspaceBackend


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX backend")


def scope(fs, paths=("/a",), actions=("read", "write", "delete"), binding=None):
    return ExecutionScope(
        namespace="local",
        thread_id="thread",
        run_id="run",
        principal="user",
        environment=(
            ExecutionEnvironment(fs)
            if binding is None
            else ExecutionEnvironment.from_binding(binding)
        ),
        permissions=PermissionSet(
            resources=[
                fs.permission(path, f"filesystem.{action}")
                for path in paths
                for action in actions
            ]
        ),
    )


def test_local_operations_and_exact_permissions(tmp_path):
    fs = LocalWorkspace("project", tmp_path)
    with pytest.raises(PermissionError):
        fs.write_bytes("/a", b"denied")
    assert not (tmp_path / "a").exists()
    with execution_context(
        scope=scope(
            fs,
            ("/", "/a", "/dir", "/dir/b"),
            ("read", "write", "delete", "list", "mkdir"),
        )
    ):
        fs.write_bytes("/a", b"real")
        assert (tmp_path / "a").read_bytes() == b"real"
        (tmp_path / "a").write_bytes(b"external")
        assert fs.read_bytes("/a") == b"external"
        fs.mkdir("/dir")
        fs.write_bytes("/dir/b", b"nested")
        assert fs.listdir("/") == ("a", "dir")
        assert fs.listdir("/dir") == ("b",)
        fs.unlink("/dir/b")
        with pytest.raises(PermissionError):
            fs.write_bytes("/other", b"denied")
    assert not (tmp_path / "dir/b").exists()


@pytest.mark.parametrize("path", ["../a", "/../a", "//a", "/dir/../../a", "/a\\b"])
def test_rejects_virtual_path_escape(tmp_path, path):
    fs = LocalWorkspace("project", tmp_path)
    with execution_context(scope=scope(fs)):
        with pytest.raises(ValueError):
            fs.read_bytes(path)


@pytest.mark.parametrize("kind", ["symlink", "directory_symlink", "hardlink", "fifo"])
def test_rejects_links_and_special_files(tmp_path, kind):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "secret"
    target.write_bytes(b"secret")
    path = "/a"
    if kind == "symlink":
        (root / "a").symlink_to(target)
    elif kind == "directory_symlink":
        (root / "dir").symlink_to(outside, target_is_directory=True)
        path = "/dir/secret"
    elif kind == "hardlink":
        os.link(target, root / "a")
    else:
        os.mkfifo(root / "a")
    fs = LocalWorkspace("project", root)
    with execution_context(scope=scope(fs, (path,))):
        for operation in (
            lambda: fs.read_bytes(path),
            lambda: fs.write_bytes(path, b"changed"),
            lambda: fs.unlink(path),
        ):
            with pytest.raises(OSError):
                operation()
    assert target.read_bytes() == b"secret"


def test_root_replacement_and_no_implicit_creation(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalWorkspace("project", tmp_path / "missing")
    root = tmp_path / "root"
    root.mkdir()
    (root / "a").write_bytes(b"old")
    fs = LocalWorkspace("project", root)
    root.rename(tmp_path / "old_root")
    root.mkdir()
    (root / "a").write_bytes(b"replacement")
    with execution_context(scope=scope(fs)):
        with pytest.raises((PermissionError, WorkspaceConflictError)):
            fs.write_bytes("/a", b"must not write")
    assert (root / "a").read_bytes() == b"replacement"
    assert (tmp_path / "old_root/a").read_bytes() == b"old"


def test_rejects_symlink_root_and_ancestor(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "child").mkdir()
    (tmp_path / "alias").symlink_to(real, target_is_directory=True)
    for root in (tmp_path / "alias", tmp_path / "alias/child"):
        with pytest.raises(OSError):
            LocalWorkspace("project", root)


def test_missing_parent_and_directory_targets_are_not_created_or_removed(tmp_path):
    fs = LocalWorkspace("project", tmp_path)
    with execution_context(scope=scope(fs, ("/missing/a", "/"))):
        with pytest.raises(FileNotFoundError):
            fs.write_bytes("/missing/a", b"no")
        with pytest.raises(IsADirectoryError):
            fs.unlink("/")
    assert list(tmp_path.iterdir()) == []


def test_create_update_delete_preserve_ordinary_mode(tmp_path):
    fs = LocalWorkspace("project", tmp_path)
    with execution_context(scope=scope(fs)):
        fs.checked_replace(
            "/a", expected=None, replacement=b"new", guarantee="cooperative_compare"
        )
        (tmp_path / "a").chmod(0o4750)
        fs.checked_replace(
            "/a",
            expected=b"new",
            replacement=b"updated",
            guarantee="cooperative_compare",
        )
        assert (tmp_path / "a").stat().st_mode & 0o7777 == 0o750
        fs.checked_replace(
            "/a", expected=b"updated", replacement=None, guarantee="cooperative_compare"
        )
    assert not (tmp_path / "a").exists()


def test_local_editor_requires_explicit_guarantee_and_approval(tmp_path):
    (tmp_path / "a").write_text("old\n")
    fs = LocalWorkspace("project", tmp_path)
    assert fs.write_capabilities.atomic_replace
    assert fs.write_capabilities.cooperative_compare
    assert not fs.write_capabilities.atomic_compare
    editor = WorkspaceEditor(fs, write_guarantee="cooperative_compare")
    journal = InMemoryApprovalStore()
    with execution_context(scope=scope(fs)):
        with pytest.raises(NotImplementedError):
            WorkspaceEditor(fs).prepare_write("/a", "new\n")
        change = editor.prepare_edit("/a", "old", "new")
        assert "-old\n+new\n" in change.diff
        with pytest.raises(PermissionError):
            editor.apply(change)
        binding = editor.approval_binding(
            change,
            tool_call_id="call",
            tool_name="edit",
            tool_revision="1",
            policy_version="1",
        )
        record = journal.request(
            binding, request_id="review", expires_at=time.time() + 60
        )
        journal.decide("local", "review", approved=True, decided_by="reviewer")
        editor.apply(change, approval=record, approval_store=journal)
    assert (tmp_path / "a").read_text() == "new\n"


def test_two_cooperative_writers_and_external_conflict(tmp_path):
    (tmp_path / "a").write_bytes(b"old")
    fs = LocalWorkspace("project", tmp_path)
    barrier = Barrier(2)

    def write(value):
        with execution_context(scope=scope(fs)):
            barrier.wait(timeout=3)
            try:
                fs.checked_replace(
                    "/a",
                    expected=b"old",
                    replacement=value,
                    guarantee="cooperative_compare",
                )
                return "applied"
            except WorkspaceConflictError:
                return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(write, (b"one", b"two"))) == ["applied", "conflict"]
    with execution_context(scope=scope(fs)):
        editor = WorkspaceEditor(
            fs, require_approval=False, write_guarantee="cooperative_compare"
        )
        change = editor.prepare_write("/a", "new")
        (tmp_path / "a").write_bytes(b"external")
        with pytest.raises(WorkspaceConflictError):
            editor.apply(change)
    assert (tmp_path / "a").read_bytes() == b"external"


def test_replace_failure_preserves_original_and_cleans_temporary(tmp_path, monkeypatch):
    (tmp_path / "a").write_bytes(b"old")
    fs = LocalWorkspace("project", tmp_path)

    def fail(*args, **kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail)
    with execution_context(scope=scope(fs)):
        with pytest.raises(OSError, match="injected"):
            fs.write_bytes("/a", b"new")
    assert (tmp_path / "a").read_bytes() == b"old"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a"]


def test_temp_collision_never_removes_an_existing_file(tmp_path, monkeypatch):
    import msgflux.runtime.workspace_local as local

    fs = LocalWorkspace("project", tmp_path)
    temporary = tmp_path / ".msgflux-collision"
    temporary.write_bytes(b"not ours")
    monkeypatch.setattr(local.secrets, "token_hex", lambda size: "collision")
    with execution_context(scope=scope(fs)):
        with pytest.raises(FileExistsError):
            fs.write_bytes("/a", b"new")
    assert temporary.read_bytes() == b"not ours"
    assert not (tmp_path / "a").exists()


def test_missing_platform_primitives_fail_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "supports_dir_fd", set())
    with pytest.raises(NotImplementedError):
        LocalWorkspace("project", tmp_path)


def test_oversized_selected_line_is_rejected_before_reading_it_all():
    stream = BytesIO(b"x" * 200_000)
    with pytest.raises(ValueError, match="byte limit"):
        LocalWorkspace._select_lines(stream, offset=1, limit=1, max_bytes=9)
    assert stream.tell() == 10


def test_abort_rechecked_after_waiting_for_filesystem_lock(tmp_path, monkeypatch):
    fs = LocalWorkspace("project", tmp_path)
    abort = AbortSignal()
    authorized = Event()
    original = fs._authorize

    def signal_authorized(*args):
        result = original(*args)
        authorized.set()
        return result

    monkeypatch.setattr(fs, "_authorize", signal_authorized)

    def write():
        with execution_context(scope=replace(scope(fs), abort_signal=abort)):
            with pytest.raises(AbortRequestedError):
                fs.write_bytes("/a", b"no")

    with ThreadPoolExecutor(max_workers=1) as pool:
        with fs._lock:
            future = pool.submit(write)
            try:
                assert authorized.wait(timeout=3)
            finally:
                abort.abort("cancelled while waiting")
        future.result(timeout=3)
    assert not (tmp_path / "a").exists()


@pytest.mark.asyncio
async def test_pagination_and_async_operations(tmp_path):
    fs = LocalWorkspace("project", tmp_path)
    with execution_context(scope=scope(fs)):
        await fs.awrite_bytes("/a", b"first\nsecond\nthird")
        assert await fs.aread_lines("/a", offset=2, limit=1) == b"second\n"
        assert fs.read_lines("/a", offset=3, limit=10) == b"third"
        with pytest.raises(ValueError):
            fs.read_lines("/a", offset=4)
        with pytest.raises(ValueError):
            fs.read_lines("/a", limit=1, max_bytes=3)
        await fs.awrite_bytes("/a", b"")
        assert fs.read_lines("/a") == b""
        await fs.awrite_bytes("/a", b"x" * 200_000 + b"\nselected\n")
        assert fs.read_lines("/a", offset=2, limit=1, max_bytes=9) == b"selected\n"


@pytest.mark.asyncio
async def test_backend_reconnect_identity_and_close(tmp_path):
    backend = LocalWorkspaceBackend(tmp_path)
    first = await backend.open("project")
    second = await backend.reconnect("project", first.identity)
    fs = first.filesystem
    assert second.filesystem is fs
    environment = scope(fs, binding=first)
    await first.aclose()
    with execution_context(scope=environment):
        with pytest.raises(PermissionError):
            fs.write_bytes("/a", b"denied")
    with execution_context(scope=scope(fs, binding=second)):
        await fs.awrite_bytes("/a", b"retained")
    for changed in (msgspec.structs.replace(first.identity, generation="other"),):
        with pytest.raises(FileNotFoundError):
            await backend.reconnect("project", changed)
    await second.aclose()
    assert (tmp_path / "a").read_bytes() == b"retained"
    with pytest.raises(FileNotFoundError):
        await LocalWorkspaceBackend(tmp_path).reconnect("project", first.identity)


@pytest.mark.asyncio
async def test_aborted_open_and_operations_do_not_change_files(tmp_path):
    abort = AbortSignal()
    abort.abort("cancelled")
    backend = LocalWorkspaceBackend(tmp_path)
    with pytest.raises(AbortRequestedError):
        await backend.open("project", abort_signal=abort)
    binding = await backend.open("project")
    with execution_context(
        scope=replace(scope(binding.filesystem, binding=binding), abort_signal=abort)
    ):
        with pytest.raises(AbortRequestedError):
            await binding.filesystem.awrite_bytes("/a", b"no")
    await binding.aclose()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_local_open_reuses_resource_but_new_backend_changes_identity(tmp_path):
    backend = LocalWorkspaceBackend(tmp_path)
    first = await backend.open("project")
    again = await backend.open("project")
    other = await LocalWorkspaceBackend(tmp_path).open("project")
    assert first.filesystem is again.filesystem
    assert first.identity == again.identity
    assert other.identity != first.identity
    for binding in (first, again, other):
        await binding.aclose()


@pytest.mark.asyncio
async def test_backend_rejects_replaced_root_before_reconnect(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    backend = LocalWorkspaceBackend(root)
    binding = await backend.open("project")
    await binding.aclose()
    root.rename(tmp_path / "original")
    root.mkdir()
    with pytest.raises(PermissionError):
        await backend.open("project")
    with pytest.raises(PermissionError):
        await backend.reconnect("project", binding.identity)


def test_read_only_grants_do_not_allow_cooperative_changes(tmp_path):
    (tmp_path / "a").write_bytes(b"old")
    fs = LocalWorkspace("project", tmp_path)
    with execution_context(scope=scope(fs, actions=("read",))):
        assert fs.read_bytes("/a") == b"old"
        with pytest.raises(PermissionError):
            fs.checked_replace(
                "/a",
                expected=b"old",
                replacement=b"new",
                guarantee="cooperative_compare",
            )
    assert (tmp_path / "a").read_bytes() == b"old"
