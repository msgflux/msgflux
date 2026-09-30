"""Explicit tool dependency for a single filesystem and execution environment."""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from uuid import uuid4

from msgflux.runtime.environment import ExecutionEnvironment, ProcessRequest
from msgflux.runtime.isolation import SandboxRequirements
from msgflux.runtime.local_executor import LocalProcessExecutor
from msgflux.runtime.permissions import PermissionSet, require_permissions
from msgflux.runtime.workspace import workspace_path
from msgflux.runtime.workspace_local import LocalWorkspace


class _LocalProjectFilesystem(LocalWorkspace):
    """Project-rooted file adapter with explicit local capability defaults."""

    def __init__(self, root, *, read_only):
        super().__init__(f"local-{uuid4().hex}", root)
        self.read_only = read_only

    def _authorize(self, operation, path):
        from msgflux.runtime.context import get_execution_scope  # noqa: PLC0415

        canonical = workspace_path(path)
        scope = get_execution_scope()
        if scope.environment is None or scope.environment.filesystem is not self:
            raise PermissionError("Filesystem is not bound to the live environment")
        scope.environment.require_active()
        if scope.abort_signal is not None:
            scope.abort_signal.raise_if_aborted()
        if self.read_only and operation in {"write", "delete", "mkdir"}:
            raise PermissionError("Workspace is read-only")
        capability = f"filesystem.{operation}"
        if (scope.permissions or PermissionSet()).allows((capability,)):
            require_permissions((capability,))
            return canonical
        # Advanced scopes may restrict the factory's broad defaults to exact files.
        return super()._authorize(operation, canonical)


class AgentWorkspace:
    """Tools declare this dependency; operations use the live scope's authority.

    Construct with an existing environment, or use ``AgentWorkspace.local`` for a
    project with host execution. Local commands are not an OS sandbox. Relative
    file and command paths use the same virtual cwd. No context manager or
    persistent connection is required for the local factory.
    """

    def __init__(self, environment: ExecutionEnvironment, *, cwd: str = "/"):
        if not isinstance(environment, ExecutionEnvironment):
            raise TypeError("AgentWorkspace requires an ExecutionEnvironment")
        self._environment = environment
        self._cwd = workspace_path(cwd)
        self._default_permissions = None

    @classmethod
    def local(cls, root, *, read_only: bool = False) -> AgentWorkspace:
        if type(read_only) is not bool:
            raise TypeError("read_only must be a boolean")
        filesystem = _LocalProjectFilesystem(
            os.path.realpath(os.fspath(root)), read_only=read_only
        )
        environment = ExecutionEnvironment(
            filesystem,
            process_executor=None if read_only else LocalProcessExecutor(filesystem),
            requirements=SandboxRequirements(),
            write_guarantee="cooperative_compare",
        )
        workspace = cls(environment)
        grants = {"filesystem.read", "filesystem.list"}
        if not read_only:
            grants.update(
                {
                    "filesystem.write",
                    "filesystem.delete",
                    "filesystem.mkdir",
                    "process.execute",
                }
            )
        workspace._default_permissions = PermissionSet(grants)
        return workspace

    @classmethod
    def from_environment(
        cls, environment: ExecutionEnvironment, *, cwd: str = "/"
    ) -> AgentWorkspace:
        return cls(environment, cwd=cwd)

    @property
    def cwd(self):
        return self._cwd

    @property
    def read_only(self):
        return bool(getattr(self._environment.filesystem, "read_only", False))

    @property
    def can_execute(self):
        return self._environment.process_executor is not None

    def with_cwd(self, cwd: str) -> AgentWorkspace:
        view = AgentWorkspace(self._environment, cwd=self.resolve(cwd))
        view._default_permissions = self._default_permissions
        return view

    def resolve(self, path: str) -> str:
        if not isinstance(path, str) or not path:
            raise ValueError("Workspace path must be non-empty text")
        return workspace_path(
            path if path.startswith("/") else f"{self.cwd.rstrip('/')}/{path}"
        )

    @contextmanager
    def _bound(self):
        from msgflux.runtime.context import (  # noqa: PLC0415
            _CURRENT_SCOPE,
            ExecutionScope,
            execution_context,
            get_execution_scope,
        )

        if _CURRENT_SCOPE.get() is None:
            with execution_context(scope=ExecutionScope(workspace=self)):
                yield
        else:
            if get_execution_scope().environment is not self._environment:
                raise PermissionError("Workspace is not bound to the live environment")
            self._environment.require_active()
            yield

    def _read(self, method, path, **kwargs):
        with self._bound():
            return getattr(self._environment.filesystem, method)(
                self.resolve(path), **kwargs
            )

    def read_prefix(self, path, *, max_bytes):
        return self._read("read_prefix", path, max_bytes=max_bytes)

    def read_bytes(self, path, *, max_bytes=1_000_000):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        data = self.read_prefix(path, max_bytes=max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("File exceeds max_bytes")
        return data

    def read_text(self, path, *, max_bytes=1_000_000, encoding="utf-8"):
        return self.read_bytes(path, max_bytes=max_bytes).decode(encoding)

    def read_lines(self, path, *, offset=1, limit=2000, max_bytes=1_000_000):
        return self._read(
            "read_lines", path, offset=offset, limit=limit, max_bytes=max_bytes
        )

    def listdir(self, path="."):
        return self._read("listdir", path)

    def scandir(self, path=".", *, max_entries=10_000):
        return self._read("scandir", path, max_entries=max_entries)

    def _editor(self, *, require_approval=True):
        return self._environment.workspace_editor(require_approval=require_approval)

    def _prepare(self, operation, path, *args):
        with self._bound():
            return getattr(self._editor(), operation)(self.resolve(path), *args)

    def prepare_write(self, path, content):
        return self._prepare("prepare_write", path, content)

    def prepare_create(self, path, content):
        return self._prepare("prepare_create", path, content)

    def prepare_edit(self, path, old, new):
        return self._prepare("prepare_edit", path, old, new)

    def prepare_transform(self, path, transform):
        return self._prepare("prepare_transform", path, transform)

    def prepare_delete(self, path):
        return self._prepare("prepare_delete", path)

    def prepare_delete_target(self, path):
        return self._prepare("prepare_delete_target", path)

    def _apply_prepared(self, change):
        """Builtin guard has already consumed approval for this exact proposal."""
        with self._bound():
            self._editor(require_approval=False).apply(change)

    def _mutate(self, operation, path, *args):
        from msgflux.runtime.approvals import agent as approvals  # noqa: PLC0415

        with self._bound():
            editor = self._editor(require_approval=approvals.approval_batch_active())
            change = getattr(editor, operation)(self.resolve(path), *args)
            editor.apply(change)

    def write_text(self, path, content):
        self._mutate("prepare_write", path, content)

    def edit_text(self, path, old, new):
        self._mutate("prepare_edit", path, old, new)

    def delete(self, path):
        self._mutate("prepare_delete_target", path)

    def mkdir(self, path):
        from msgflux.runtime.approvals import agent as approvals  # noqa: PLC0415

        if approvals.approval_batch_active():
            raise PermissionError(
                "Directory creation requires a reviewed workspace tool"
            )
        self._read("mkdir", path)

    async def arun(
        self, command, *, timeout=30, max_output_bytes=1_000_000, on_output=None
    ):
        if isinstance(command, ProcessRequest):
            request = command
        else:
            argv = (
                ("bash", "--noprofile", "--norc", "-c", command)
                if isinstance(command, str)
                else command
            )
            if not isinstance(argv, (tuple, list)):
                raise TypeError(
                    "command must be a string, argv sequence or ProcessRequest"
                )
            request = ProcessRequest(
                tuple(argv),
                cwd=self.cwd,
                timeout_seconds=timeout,
                max_output_bytes=max_output_bytes,
            )
        with self._bound():
            return await self._environment.arun(request, on_output=on_output)

    def run(self, command, **kwargs):
        from msgflux._private.executor import Executor  # noqa: PLC0415

        return Executor.get_instance().submit(self.arun, command, **kwargs).result()

    async def aread_prefix(self, path, **kwargs):
        return await asyncio.to_thread(self.read_prefix, path, **kwargs)

    async def aread_bytes(self, path, **kwargs):
        return await asyncio.to_thread(self.read_bytes, path, **kwargs)

    async def aread_text(self, path, **kwargs):
        return await asyncio.to_thread(self.read_text, path, **kwargs)

    async def aread_lines(self, path, **kwargs):
        return await asyncio.to_thread(self.read_lines, path, **kwargs)

    async def alistdir(self, path="."):
        return await asyncio.to_thread(self.listdir, path)

    async def ascandir(self, path=".", **kwargs):
        return await asyncio.to_thread(self.scandir, path, **kwargs)

    async def awrite_text(self, path, content):
        await asyncio.to_thread(self.write_text, path, content)

    async def aedit_text(self, path, old, new):
        await asyncio.to_thread(self.edit_text, path, old, new)

    async def adelete(self, path):
        await asyncio.to_thread(self.delete, path)

    async def amkdir(self, path):
        await asyncio.to_thread(self.mkdir, path)


def resolve_workspace(workspace=None):
    """Select the workspace bound to the active execution, or an explicit instance."""
    from msgflux.runtime.context import get_execution_scope  # noqa: PLC0415

    scope = get_execution_scope()
    workspace = scope.workspace if workspace is None else workspace
    if not isinstance(workspace, AgentWorkspace):
        raise PermissionError("Workspace tools require a live workspace")
    if (
        scope.environment is not None
        and scope.environment is not workspace._environment
    ):
        raise PermissionError("Workspace is not bound to the live environment")
    return workspace
