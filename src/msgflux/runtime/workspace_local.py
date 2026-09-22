"""POSIX host-directory workspace backend.

This adapter provides path-safe access to a directory supplied by the host.  It
is deliberately not an OS sandbox: writers outside this backend can race it.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
from contextlib import contextmanager
from threading import RLock

from msgflux.runtime.abort import AbortSignal
from msgflux.runtime.workspace import WorkspaceConflictError, WorkspaceFilesystem
from msgflux.runtime.workspace_backend import WorkspaceBackend, WorkspaceBinding
from msgflux.runtime.workspace_contracts import (
    WorkspaceEntry,
    WorkspaceIdentity,
    WorkspacePromptInfo,
    WorkspaceWriteCapabilities,
)

_UNSET = object()


def _check_posix() -> None:
    if (
        os.name != "posix"
        or any(
            not hasattr(os, flag)
            for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
        )
        or not {os.open, os.stat, os.mkdir, os.rmdir, os.unlink, os.rename}
        <= os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
        or os.listdir not in os.supports_fd
    ):
        raise NotImplementedError(
            "LocalWorkspace requires POSIX descriptor-relative filesystem APIs"
        )


@contextmanager
def _root_directory(parts):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts:
            child = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


class LocalWorkspace(WorkspaceFilesystem):
    """A virtual workspace rooted at an existing absolute host directory."""

    prompt_info = WorkspacePromptInfo(
        storage="host files; changes persist after binding close",
        guidance=(
            "Changes affect real files. Parents must exist. Symlinks, hardlinked "
            "files, special files and cross-device traversal are rejected. "
            "External writers can race cooperative comparison. "
            "This backend is not an OS sandbox."
        ),
    )

    def __init__(
        self,
        workspace_id: str,
        root: str | os.PathLike[str],
    ):
        _check_posix()
        super().__init__(workspace_id)
        root_path = os.fspath(root)
        if (
            not isinstance(root_path, str)
            or not os.path.isabs(root_path)
            or ".." in root_path.split("/")
        ):
            raise ValueError("Local workspace root must be absolute without traversal")
        self._root_parts = tuple(
            part for part in root_path.split("/") if part and part != "."
        )
        self._lock = RLock()
        with _root_directory(self._root_parts) as fd:
            self._root_stat = os.fstat(fd)

    @property
    def write_capabilities(self) -> WorkspaceWriteCapabilities:
        return WorkspaceWriteCapabilities(
            atomic_replace=True, cooperative_compare=True, atomic_compare=False
        )

    def _open_root(self) -> int:
        with _root_directory(self._root_parts) as root:
            fd = os.dup(root)
        try:
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) != (
                self._root_stat.st_dev,
                self._root_stat.st_ino,
            ):
                raise PermissionError("Workspace root was replaced")
            return fd
        except BaseException:
            os.close(fd)
            raise

    @staticmethod
    def _parts(path: str) -> tuple[str, ...]:
        return tuple(part for part in path.split("/") if part)

    def _parent(self, fd: int, path: str) -> tuple[int, str]:
        parts = self._parts(path)
        if not parts:
            raise IsADirectoryError(path)
        current = fd
        try:
            for part in parts[:-1]:
                nxt = os.open(
                    part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current
                )
                if os.fstat(nxt).st_dev != self._root_stat.st_dev:
                    os.close(nxt)
                    raise PermissionError("Workspace mount crossing")
                if current != fd:
                    os.close(current)
                current = nxt
            return current, parts[-1]
        except BaseException:
            if current != fd:
                os.close(current)
            raise

    def _regular(self, st, path: str) -> None:
        if not stat.S_ISREG(st.st_mode):
            raise IsADirectoryError(path)
        if st.st_nlink != 1:
            raise PermissionError("Hard-linked workspace files are not supported")
        if st.st_dev != self._root_stat.st_dev:
            raise PermissionError("Workspace mount crossing")

    def _stat_at(self, parent: int, name: str, path: str):
        st = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISLNK(st.st_mode):
            raise PermissionError("Symlinks are not supported")
        self._regular(st, path)
        return st

    def _replace(  # noqa: C901
        self, path: str, replacement: bytes | None, *, expected=_UNSET
    ) -> None:
        root = self._open_root()
        parent = root
        temp_name = None
        temp_fd = None
        try:
            parent, name = self._parent(root, path)
            try:
                st = self._stat_at(parent, name, path)
                current = (
                    self._read_fd_at(parent, name, len(expected) + 1)
                    if isinstance(expected, bytes)
                    else b""
                )
            except FileNotFoundError:
                current = None
            if expected is not _UNSET:
                if current != expected:
                    raise WorkspaceConflictError("File changed since preparation")
            if replacement is None:
                if current is None:
                    raise FileNotFoundError(path)
                os.unlink(name, dir_fd=parent)
                return
            mode = 0o600 if current is None else st.st_mode & 0o777
            created = False
            for _ in range(10):
                candidate = f".msgflux-{secrets.token_hex(12)}"
                try:
                    temp_fd = os.open(
                        candidate,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=parent,
                    )
                    temp_name = candidate
                    created = True
                    break
                except FileExistsError:
                    continue
            if not created:
                raise FileExistsError("Unable to create temporary workspace file")
            with os.fdopen(temp_fd, "wb") as stream:
                temp_fd = None
                stream.write(replacement)
                stream.flush()
                os.fchmod(stream.fileno(), mode)
            os.replace(temp_name, name, src_dir_fd=parent, dst_dir_fd=parent)
            temp_name = None
        finally:
            if temp_fd is not None:
                os.close(temp_fd)
            if temp_name is not None:
                try:
                    os.unlink(temp_name, dir_fd=parent)
                except OSError:
                    pass
            if parent != root:
                os.close(parent)
            os.close(root)

    def _read_fd_at(self, parent: int, name: str, size: int = -1) -> bytes:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            st = os.fstat(fd)
            self._regular(st, name)
            with os.fdopen(fd, "rb") as stream:
                fd = -1
                return stream.read(size)
        finally:
            if fd >= 0:
                os.close(fd)

    def _operate(self, operation: str, path: str, data: bytes | None):  # noqa: C901
        with self._lock:
            self._authorize(operation, path)
            if operation == "write":
                self._replace(path, data)
                return None
            root = self._open_root()
            parent = root
            try:
                if operation == "list":
                    if path == "/":
                        fd = root
                    else:
                        parent, name = self._parent(root, path)
                        st = os.stat(name, dir_fd=parent, follow_symlinks=False)
                        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                            raise NotADirectoryError(path)
                        fd = os.open(
                            name,
                            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=parent,
                        )
                    try:
                        if os.fstat(fd).st_dev != self._root_stat.st_dev:
                            raise PermissionError("Workspace mount crossing")
                        return tuple(sorted(os.listdir(fd)))
                    finally:
                        if fd != root:
                            os.close(fd)
                parent, name = self._parent(root, path)
                if operation == "mkdir":
                    os.mkdir(name, dir_fd=parent)
                    return None
                self._stat_at(parent, name, path)
                if operation == "read":
                    return self._read_fd_at(parent, name)
                if operation == "delete":
                    os.unlink(name, dir_fd=parent)
                    return None
                raise ValueError("Unsupported workspace operation")
            finally:
                if parent != root:
                    os.close(parent)
                os.close(root)

    def _read_lines(self, path, offset, limit, max_bytes):
        with self._lock:
            self._authorize("read", path)
            root = self._open_root()
            parent = root
            fd = None
            try:
                parent, name = self._parent(root, path)
                self._stat_at(parent, name, path)
                fd = os.open(
                    name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent
                )
                opened_stat = os.fstat(fd)
                self._regular(opened_stat, path)
                stream = os.fdopen(fd, "rb")
                fd = -1
                try:
                    return self._select_lines(stream, offset, limit, max_bytes)
                finally:
                    stream.close()
            finally:
                if fd is not None and fd >= 0:
                    os.close(fd)
                if parent != root:
                    os.close(parent)
                os.close(root)

    def _read_prefix(self, path, max_bytes):
        with self._lock:
            self._authorize("read", path)
            root = self._open_root()
            parent = root
            try:
                parent, name = self._parent(root, path)
                self._stat_at(parent, name, path)
                return self._read_fd_at(parent, name, max_bytes)
            finally:
                if parent != root:
                    os.close(parent)
                os.close(root)

    def _directory_deletion(self, path, expected=None):
        with self._lock:
            self._authorize("delete", path)
            root = self._open_root()
            parent = root
            directory = None
            try:
                parent, name = self._parent(root, path)
                st = os.stat(name, dir_fd=parent, follow_symlinks=False)
                if stat.S_ISREG(st.st_mode):
                    self._regular(st, path)
                    if expected is not None:
                        raise WorkspaceConflictError("Directory replaced by a file")
                    return None
                self._authorize("list", path)
                if not stat.S_ISDIR(st.st_mode):
                    raise PermissionError("Deletion target is not a safe directory")
                directory = os.open(
                    name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                )
                opened = os.fstat(directory)
                if opened.st_dev != self._root_stat.st_dev:
                    raise PermissionError("Workspace mount crossing")
                token = f"{opened.st_dev}:{opened.st_ino}:{opened.st_ctime_ns}"
                with os.scandir(directory) as entries:
                    if next(entries, None) is not None:
                        raise OSError(errno.ENOTEMPTY, "Directory is not empty", path)
                if expected is not None:
                    current = os.stat(name, dir_fd=parent, follow_symlinks=False)
                    current_token = (
                        f"{current.st_dev}:{current.st_ino}:{current.st_ctime_ns}"
                    )
                    if token != expected or current_token != expected:
                        raise WorkspaceConflictError(
                            "Directory changed since preparation"
                        )
                    # POSIX rmdir refuses newly added contents. As with file
                    # writes, unrelated external writers can race comparison.
                    os.rmdir(name, dir_fd=parent)
                return token
            finally:
                if directory is not None:
                    os.close(directory)
                if parent != root:
                    os.close(parent)
                os.close(root)

    def _deletion_directory_token(self, path):
        return self._directory_deletion(path)

    def _checked_rmdir(self, path, expected):
        self._directory_deletion(path, expected)

    def _scandir(self, path, max_entries):  # noqa: C901
        with self._lock:
            self._authorize("list", path)
            root = self._open_root()
            parent = root
            directory = root
            try:
                if path != "/":
                    parent, name = self._parent(root, path)
                    st = os.stat(name, dir_fd=parent, follow_symlinks=False)
                    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                        raise NotADirectoryError(path)
                    if st.st_dev != self._root_stat.st_dev:
                        raise PermissionError("Workspace mount crossing")
                    directory = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=parent,
                    )
                    if os.fstat(directory).st_dev != self._root_stat.st_dev:
                        raise PermissionError("Workspace mount crossing")
                entries = []
                with os.scandir(directory) as iterator:
                    for item in iterator:
                        st = item.stat(follow_symlinks=False)
                        if stat.S_ISLNK(st.st_mode):
                            kind = "other"
                        elif stat.S_ISDIR(st.st_mode):
                            kind = (
                                "directory"
                                if st.st_dev == self._root_stat.st_dev
                                else "other"
                            )
                        elif (
                            stat.S_ISREG(st.st_mode)
                            and st.st_nlink == 1
                            and st.st_dev == self._root_stat.st_dev
                        ):
                            kind = "file"
                        else:
                            kind = "other"
                        entries.append(WorkspaceEntry(name=item.name, kind=kind))
                        if len(entries) > max_entries:
                            raise ValueError("Workspace directory exceeds max_entries")
                return tuple(sorted(entries, key=lambda entry: entry.name))
            finally:
                if directory != root:
                    os.close(directory)
                if parent != root:
                    os.close(parent)
                os.close(root)

    @staticmethod
    def _select_lines(stream, offset, limit, max_bytes):
        for _ in range(offset - 1):
            while True:
                chunk = stream.readline(65536)
                if not chunk:
                    raise ValueError("offset exceeds the number of lines")
                if chunk.endswith(b"\n"):
                    break
        result = bytearray()
        for _ in range(limit):
            line = stream.readline(max_bytes - len(result) + 1)
            if not line:
                if not result and offset != 1:
                    raise ValueError("offset exceeds the number of lines")
                break
            result.extend(line)
            if len(result) > max_bytes:
                raise ValueError("Selected lines exceed the read byte limit")
        return bytes(result)

    def _checked_replace(self, path, expected, replacement):
        with self._lock:
            self._authorize("read", path)
            self._authorize("delete" if replacement is None else "write", path)
            self._replace(path, replacement, expected=expected)


class LocalWorkspaceBackend(WorkspaceBackend):
    """One selected directory, with identities retained during this backend's life.

    Repeated opens for one id share files/identity but not grants or lifecycle.
    Different ids still address this same directory, not isolated copies.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
    ):
        _check_posix()
        self._root = os.fspath(root)
        self._resources: dict[str, LocalWorkspace] = {}
        self._registry_lock = RLock()
        self._filesystem_lock = RLock()

    async def open(self, workspace_id: str, *, abort_signal: AbortSignal | None = None):
        if abort_signal is not None:
            abort_signal.raise_if_aborted()
        with self._registry_lock:
            filesystem = self._resources.get(workspace_id)
            if filesystem is None:
                filesystem = LocalWorkspace(workspace_id, self._root)
                filesystem._lock = self._filesystem_lock
                self._resources[workspace_id] = filesystem
            os.close(filesystem._open_root())
            return self._bind(filesystem)

    async def reconnect(self, workspace_id, identity, *, abort_signal=None):
        if abort_signal is not None:
            abort_signal.raise_if_aborted()
        if not isinstance(identity, WorkspaceIdentity):
            raise TypeError("identity must be a WorkspaceIdentity")
        with self._registry_lock:
            filesystem = self._resources.get(workspace_id)
            if filesystem is None or filesystem.identity != identity:
                raise FileNotFoundError(
                    "Workspace resource is unavailable or identity changed"
                )
            os.close(filesystem._open_root())
            return self._bind(filesystem)

    def _bind(self, filesystem):
        return WorkspaceBinding(self, filesystem, ownership="borrowed")

    async def _release(self, binding):
        if binding.backend is not self:
            raise ValueError("Binding belongs to another backend")
