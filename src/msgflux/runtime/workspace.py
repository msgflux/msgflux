"""Virtual workspace operations authorized against live execution authority."""

from __future__ import annotations

import asyncio
import errno
import itertools
import re
from abc import ABC, abstractmethod
from pathlib import PurePosixPath
from threading import RLock
from typing import Mapping
from uuid import uuid4

from msgflux.runtime.permissions import ResourcePermission, require_permissions
from msgflux.runtime.workspace_contracts import (
    WorkspaceEntry,
    WorkspaceIdentity,
    WorkspacePromptInfo,
    WorkspaceWriteCapabilities,
    WriteGuarantee,
)


def workspace_path(path: str) -> str:
    """Canonical virtual path; never interpret this as a host filesystem path."""
    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or path.startswith("//")
        or "\\" in path
        or any(ord(char) < 32 for char in path)
        or ".." in path.split("/")
    ):
        raise ValueError("Expected an absolute virtual POSIX path without traversal")
    return str(PurePosixPath(path))


class WorkspaceFilesystem(ABC):
    """Trusted backend; public operations share resolution and authorization."""

    supports_atomic_changes = False
    prompt_info = WorkspacePromptInfo()

    def __init__(self, workspace_id: str, *, identity: WorkspaceIdentity | None = None):
        if not isinstance(workspace_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+", workspace_id
        ):
            raise ValueError("workspace_id must contain letters, digits, dots, _ or -")
        self._workspace_id = workspace_id
        self._requires_binding = False
        if identity is not None and not isinstance(identity, WorkspaceIdentity):
            raise TypeError("identity must be a WorkspaceIdentity")
        self._identity = identity or WorkspaceIdentity(
            backend=f"{type(self).__module__}.{type(self).__qualname__}",
            resource_id=workspace_id,
            generation=uuid4().hex,
        )

    @property
    def identity(self) -> WorkspaceIdentity:
        return self._identity

    @property
    def requires_binding(self) -> bool:
        return self._requires_binding

    def _require_binding(self) -> None:
        # One-way promotion: shared/reconnected resources may have multiple live
        # bindings, but must never regain unmanaged access through public APIs.
        self._requires_binding = True

    @property
    def write_capabilities(self) -> WorkspaceWriteCapabilities:
        # Preserve the existing strict backend contract. Cooperative backends
        # override this property; they must not claim supports_atomic_changes.
        return WorkspaceWriteCapabilities(atomic_compare=self.supports_atomic_changes)

    def require_write_guarantee(self, guarantee: WriteGuarantee) -> None:
        capabilities = self.write_capabilities
        if not isinstance(capabilities, WorkspaceWriteCapabilities):
            raise TypeError("Expected WorkspaceWriteCapabilities")
        capabilities.require(guarantee)
        if guarantee == "atomic_compare" and not self.supports_atomic_changes:
            raise NotImplementedError("Backend has no atomic compare implementation")

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    def permission(self, path: str, action: str) -> ResourcePermission:
        return ResourcePermission(
            f"workspace:{self.workspace_id}:{workspace_path(path)}", action
        )

    def _authorize(self, operation, path):
        from msgflux.runtime.context import get_execution_scope  # noqa: PLC0415

        canonical = workspace_path(path)
        scope = get_execution_scope()
        if scope.environment is None or scope.environment.filesystem is not self:
            raise PermissionError("Filesystem is not bound to the live environment")
        scope.environment.require_active()
        if scope.abort_signal is not None:
            scope.abort_signal.raise_if_aborted()
        require_permissions(
            (), (self.permission(canonical, f"filesystem.{operation}"),)
        )
        return canonical

    def _perform(self, operation, path, data=None):
        return self._operate(operation, self._authorize(operation, path), data)

    @abstractmethod
    def _operate(self, operation: str, path: str, data: bytes | None):
        """Perform exactly the authorized operation on the resolved virtual path."""
        raise NotImplementedError

    def read_bytes(self, path: str) -> bytes:
        return self._perform("read", path)

    def read_prefix(self, path: str, *, max_bytes: int) -> bytes:
        """Read at most ``max_bytes`` from one authorized regular file."""
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        canonical = self._authorize("read", path)
        data = self._read_prefix(canonical, max_bytes)
        if not isinstance(data, bytes) or len(data) > max_bytes:
            raise ValueError("Backend exceeded the requested read byte limit")
        return data

    @abstractmethod
    def _read_prefix(self, path: str, max_bytes: int) -> bytes:
        raise NotImplementedError

    def read_lines(
        self,
        path: str,
        *,
        offset: int = 1,
        limit: int = 2000,
        max_bytes: int = 1_000_000,
    ) -> bytes:
        """Read LF-delimited lines, preserving bytes, after live authorization.

        Backends may override _read_lines for bounded I/O. The compatibility
        implementation reads bytes once but never splits/decodes the whole file.
        """
        if any(
            type(value) is not int or value <= 0 for value in (offset, limit, max_bytes)
        ):
            raise ValueError("offset, limit and max_bytes must be positive integers")
        canonical = self._authorize("read", path)
        data = self._read_lines(canonical, offset, limit, max_bytes)
        if not isinstance(data, bytes) or len(data) > max_bytes:
            raise ValueError("Backend exceeded the requested read byte limit")
        if data.count(b"\n") + bool(data and not data.endswith(b"\n")) > limit:
            raise ValueError("Backend exceeded the requested read line limit")
        return data

    def _read_lines(self, path, offset, limit, max_bytes):
        data = self._operate("read", path, None)
        start = 0
        for _ in range(offset - 1):
            newline = data.find(b"\n", start)
            if newline < 0:
                raise ValueError("offset exceeds the number of lines")
            start = newline + 1
        if start >= len(data) and offset != 1:
            raise ValueError("offset exceeds the number of lines")
        end = start
        for _ in range(limit):
            newline = data.find(b"\n", end)
            end = len(data) if newline < 0 else newline + 1
            if end - start > max_bytes:
                raise ValueError("Selected lines exceed the read byte limit")
            if end == len(data):
                break
        return data[start:end]

    async def aread_lines(
        self,
        path: str,
        *,
        offset: int = 1,
        limit: int = 2000,
        max_bytes: int = 1_000_000,
    ) -> bytes:
        return await asyncio.to_thread(
            self.read_lines, path, offset=offset, limit=limit, max_bytes=max_bytes
        )

    def write_bytes(self, path: str, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise TypeError("File contents must be bytes")
        self._perform("write", path, data)

    def compare_exchange(
        self, path: str, *, expected: bytes | None, replacement: bytes | None
    ) -> None:
        """Atomically replace exactly the expected contents; None means absent.

        Backends must override _compare_exchange with a real atomic operation,
        including protection against concurrent writers and path substitution.
        There is deliberately no read-then-write compatibility implementation.
        """
        self._validate_change(expected, replacement)
        if not self.supports_atomic_changes:
            raise NotImplementedError(
                "Workspace backend does not support atomic changes"
            )
        canonical = self._authorize_change(path, replacement)
        self._compare_exchange(canonical, expected, replacement)

    def _compare_exchange(self, path, expected, replacement):
        raise NotImplementedError("Workspace backend does not support atomic changes")

    def checked_replace(
        self,
        path: str,
        *,
        expected: bytes | None,
        replacement: bytes | None,
        guarantee: WriteGuarantee = "atomic_compare",
    ) -> None:
        """Replace under an explicit guarantee; never downgrade atomic requests."""
        self.require_write_guarantee(guarantee)
        if guarantee == "atomic_compare" or self.supports_atomic_changes:
            return self.compare_exchange(
                path, expected=expected, replacement=replacement
            )
        self._validate_change(expected, replacement)
        canonical = self._authorize_change(path, replacement)
        self._checked_replace(canonical, expected, replacement)

    @staticmethod
    def _validate_change(expected, replacement):
        if any(
            value is not None and not isinstance(value, bytes)
            for value in (expected, replacement)
        ):
            raise TypeError("Expected contents and replacement must be bytes or None")
        if expected is None and replacement is None:
            raise ValueError("Cannot delete an absent file")

    def _authorize_change(self, path, replacement):
        canonical = self._authorize("read", path)
        self._authorize("delete" if replacement is None else "write", canonical)
        return canonical

    def _checked_replace(self, path, expected, replacement):
        """Cooperative backend hook; coordinate, recheck authority and compare."""
        raise NotImplementedError("Backend has no cooperative compare implementation")

    async def achecked_replace(self, path: str, **kwargs) -> None:
        await asyncio.to_thread(self.checked_replace, path, **kwargs)

    async def acompare_exchange(
        self, path: str, *, expected: bytes | None, replacement: bytes | None
    ) -> None:
        await asyncio.to_thread(
            self.compare_exchange, path, expected=expected, replacement=replacement
        )

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        return self.read_bytes(path).decode(encoding)

    def write_text(self, path: str, text: str, *, encoding: str = "utf-8") -> None:
        self.write_bytes(path, text.encode(encoding))

    def listdir(self, path: str) -> tuple[str, ...]:
        return self._perform("list", path)

    def scandir(
        self, path: str, *, max_entries: int = 10_000
    ) -> tuple[WorkspaceEntry, ...]:
        """Return a bounded, sorted description of one authorized directory."""
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        canonical = self._authorize("list", path)
        entries = self._scandir(canonical, max_entries)
        if not isinstance(entries, tuple) or len(entries) > max_entries:
            raise ValueError("Backend exceeded the requested directory entry limit")
        if not all(isinstance(entry, WorkspaceEntry) for entry in entries):
            raise TypeError("Backend returned an invalid workspace entry")
        return entries

    @abstractmethod
    def _scandir(self, path: str, max_entries: int) -> tuple[WorkspaceEntry, ...]:
        raise NotImplementedError

    def mkdir(self, path: str) -> None:
        self._perform("mkdir", path)

    def deletion_directory_token(self, path: str) -> str | None:
        """Inspect an authorized deletion target; None denotes a regular file.

        Directories additionally require list permission and must be empty.
        The opaque token binds a later deletion to this directory incarnation.
        """
        canonical = self._authorize("delete", path)
        if canonical == "/":
            raise PermissionError("Cannot delete the workspace root")
        return self._deletion_directory_token(canonical)

    def _deletion_directory_token(self, path: str) -> str | None:
        raise NotImplementedError("Backend does not support directory deletion")

    def checked_rmdir(
        self, path: str, *, expected: str, guarantee: WriteGuarantee = "atomic_compare"
    ) -> None:
        """Delete only the reviewed empty directory, never recursively."""
        canonical = self._authorize("delete", path)
        self._authorize("list", canonical)
        if canonical == "/":
            raise PermissionError("Cannot delete the workspace root")
        if not isinstance(expected, str) or not expected:
            raise ValueError("Expected a directory identity token")
        self.require_write_guarantee(guarantee)
        self._checked_rmdir(canonical, expected)

    def _checked_rmdir(self, path: str, expected: str) -> None:
        raise NotImplementedError("Backend does not support directory deletion")

    def unlink(self, path: str) -> None:
        self._perform("delete", path)

    async def aread_bytes(self, path: str) -> bytes:
        return await asyncio.to_thread(self.read_bytes, path)

    async def aread_prefix(self, path: str, *, max_bytes: int) -> bytes:
        return await asyncio.to_thread(self.read_prefix, path, max_bytes=max_bytes)

    async def awrite_bytes(self, path: str, data: bytes) -> None:
        await asyncio.to_thread(self.write_bytes, path, data)

    async def aread_text(self, path: str, *, encoding: str = "utf-8") -> str:
        return await asyncio.to_thread(self.read_text, path, encoding=encoding)

    async def awrite_text(
        self, path: str, text: str, *, encoding: str = "utf-8"
    ) -> None:
        await asyncio.to_thread(self.write_text, path, text, encoding=encoding)

    async def alistdir(self, path: str) -> tuple[str, ...]:
        return await asyncio.to_thread(self.listdir, path)

    async def ascandir(
        self, path: str, *, max_entries: int = 10_000
    ) -> tuple[WorkspaceEntry, ...]:
        return await asyncio.to_thread(self.scandir, path, max_entries=max_entries)

    async def amkdir(self, path: str) -> None:
        await asyncio.to_thread(self.mkdir, path)

    async def aunlink(self, path: str) -> None:
        await asyncio.to_thread(self.unlink, path)


class InMemoryWorkspace(WorkspaceFilesystem):
    """Process-local VFS with no symlinks, mounts or host filesystem access."""

    supports_atomic_changes = True
    prompt_info = WorkspacePromptInfo(
        storage="process-local memory",
        guidance=(
            "Files do not modify the host filesystem and are not durable across "
            "process restarts. No symlinks or mounts."
        ),
    )

    def __init__(self, workspace_id: str, files: Mapping[str, bytes] | None = None):
        super().__init__(workspace_id)
        self._lock = RLock()
        self._files = {}
        self._directories = {"/"}
        for path, data in (files or {}).items():
            canonical = workspace_path(path)
            if not isinstance(data, bytes):
                raise TypeError("Initial file contents must be bytes")
            if canonical in self._files or canonical in self._directories:
                raise ValueError("Conflicting initial workspace paths")
            self._files[canonical] = data
            self._directories.update(
                str(parent) for parent in PurePosixPath(canonical).parents
            )
        if self._directories & self._files.keys():
            raise ValueError("A workspace path cannot be both a file and a directory")
        self._directory_tokens = {path: uuid4().hex for path in self._directories}

    def _deletion_directory_token(self, path):
        with self._lock:
            self._authorize("delete", path)
            if path in self._files:
                return None
            self._authorize("list", path)
            if path not in self._directories:
                raise FileNotFoundError(path)
            if any(
                item != path and item.startswith(path.rstrip("/") + "/")
                for item in itertools.chain(self._files, self._directories)
            ):
                raise OSError(errno.ENOTEMPTY, "Directory is not empty", path)
            return self._directory_tokens[path]

    def _checked_rmdir(self, path, expected):
        with self._lock:
            self._authorize("list", path)
            if self.deletion_directory_token(path) != expected:
                raise WorkspaceConflictError("Directory changed since preparation")
            self._directories.remove(path)
            del self._directory_tokens[path]

    def _operate(self, operation, path, data):
        with self._lock:
            if operation == "list":
                if path not in self._directories:
                    raise NotADirectoryError(path)
                return tuple(
                    sorted(
                        PurePosixPath(item).name
                        for item in self._files.keys() | self._directories
                        if item != path and str(PurePosixPath(item).parent) == path
                    )
                )
            if path in self._directories:
                raise IsADirectoryError(path)
            if operation in {"write", "mkdir"}:
                return self._create(operation, path, data)
            if path not in self._files:
                raise FileNotFoundError(path)
            if operation == "read":
                return self._files[path]
            if operation == "delete":
                del self._files[path]
                return None
            raise ValueError("Unsupported workspace operation")

    def _read_prefix(self, path, max_bytes):
        with self._lock:
            self._authorize("read", path)
            if path in self._directories:
                raise IsADirectoryError(path)
            if path not in self._files:
                raise FileNotFoundError(path)
            return self._files[path][:max_bytes]

    def _scandir(self, path, max_entries):
        with self._lock:
            self._authorize("list", path)
            if path not in self._directories:
                raise NotADirectoryError(path)
            entries = []
            for item in itertools.chain(self._files, self._directories):
                if item == path or str(PurePosixPath(item).parent) != path:
                    continue
                entries.append(
                    WorkspaceEntry(
                        name=PurePosixPath(item).name,
                        kind="file" if item in self._files else "directory",
                    )
                )
                if len(entries) > max_entries:
                    raise ValueError("Workspace directory exceeds max_entries")
            return tuple(sorted(entries, key=lambda entry: entry.name))

    def _compare_exchange(self, path, expected, replacement):
        with self._lock:
            # Recheck live authority after waiting for the backend lock.
            self._authorize("read", path)
            self._authorize("delete" if replacement is None else "write", path)
            if path in self._directories:
                raise IsADirectoryError(path)
            if self._files.get(path) != expected:
                raise WorkspaceConflictError("File changed since preparation")
            if replacement is None:
                del self._files[path]
            else:
                self._create("write", path, replacement)

    def _create(self, operation, path, data):
        if str(PurePosixPath(path).parent) not in self._directories:
            raise FileNotFoundError("Parent directory does not exist")
        if operation == "mkdir":
            if path in self._files:
                raise FileExistsError(path)
            self._directories.add(path)
            self._directory_tokens[path] = uuid4().hex
        else:
            self._files[path] = data


class WorkspaceConflictError(RuntimeError):
    """The current file no longer matches the prepared change."""
