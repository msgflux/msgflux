"""Virtual workspace operations authorized against live execution authority."""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from pathlib import PurePosixPath
from threading import RLock
from typing import Mapping

from msgflux.runtime.permissions import ResourcePermission, require_permissions


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

    def __init__(self, workspace_id: str):
        if not isinstance(workspace_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+", workspace_id
        ):
            raise ValueError("workspace_id must contain letters, digits, dots, _ or -")
        self._workspace_id = workspace_id

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

    def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        return self.read_bytes(path).decode(encoding)

    def write_text(self, path: str, text: str, *, encoding: str = "utf-8") -> None:
        self.write_bytes(path, text.encode(encoding))

    def listdir(self, path: str) -> tuple[str, ...]:
        return self._perform("list", path)

    def mkdir(self, path: str) -> None:
        self._perform("mkdir", path)

    def unlink(self, path: str) -> None:
        self._perform("delete", path)

    async def aread_bytes(self, path: str) -> bytes:
        return await asyncio.to_thread(self.read_bytes, path)

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

    async def amkdir(self, path: str) -> None:
        await asyncio.to_thread(self.mkdir, path)

    async def aunlink(self, path: str) -> None:
        await asyncio.to_thread(self.unlink, path)


class InMemoryWorkspace(WorkspaceFilesystem):
    """Process-local VFS with no symlinks, mounts or host filesystem access."""

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

    def _create(self, operation, path, data):
        if str(PurePosixPath(path).parent) not in self._directories:
            raise FileNotFoundError("Parent directory does not exist")
        if operation == "mkdir":
            if path in self._files:
                raise FileExistsError(path)
            self._directories.add(path)
        else:
            self._files[path] = data
