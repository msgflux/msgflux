"""Explicit tool dependency for a single filesystem and execution environment."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import contextmanager
from uuid import uuid4

from msgflux.runtime.environment import (
    ExecutionEnvironment,
    ProcessRequest,
    ProcessResult,
)
from msgflux.runtime.isolation import SandboxRequirements
from msgflux.runtime.local_executor import LocalProcessExecutor
from msgflux.runtime.permissions import PermissionSet
from msgflux.runtime.workspace import workspace_path
from msgflux.runtime.workspace_changes import PreparedFileChange
from msgflux.runtime.workspace_contracts import WorkspaceEntry, WorkspaceIdentity
from msgflux.runtime.workspace_local import LocalWorkspace


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
        defaults = PermissionSet(grants)
        filesystem = LocalWorkspace(
            f"local-{uuid4().hex}",
            os.path.realpath(os.fspath(root)),
            read_only=read_only,
            capabilities=defaults,
        )
        environment = ExecutionEnvironment(
            filesystem,
            process_executor=None if read_only else LocalProcessExecutor(filesystem),
            requirements=SandboxRequirements(),
            write_guarantee="cooperative_compare",
        )
        workspace = cls(environment)
        workspace._default_permissions = defaults
        return workspace

    @classmethod
    def from_environment(
        cls, environment: ExecutionEnvironment, *, cwd: str = "/"
    ) -> AgentWorkspace:
        return cls(environment, cwd=cwd)

    @property
    def cwd(self) -> str:
        return self._cwd

    @property
    def read_only(self) -> bool:
        return bool(getattr(self._environment.filesystem, "read_only", False))

    @property
    def supports_execution(self) -> bool:
        return self._environment.process_executor is not None

    @property
    def default_permissions(self) -> PermissionSet | None:
        return self._default_permissions

    @property
    def identity(self) -> WorkspaceIdentity:
        return self._environment.filesystem.identity

    @property
    def workspace_id(self) -> str:
        return self._environment.filesystem.workspace_id

    @property
    def requirements(self) -> SandboxRequirements:
        return self._environment.requirements

    def shares_environment(self, other: object) -> bool:
        return (
            isinstance(other, AgentWorkspace)
            and self._environment is other._environment
        )

    def require_active(self) -> None:
        self._environment.require_active()

    @property
    def editor(self) -> AgentWorkspaceEditor:
        return AgentWorkspaceEditor(self)

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
        )

        if _CURRENT_SCOPE.get() is None:
            with execution_context(scope=ExecutionScope(workspace=self)):
                yield
        else:
            require_workspace_authority(self)
            yield

    def _read(self, method, path, **kwargs):
        with self._bound():
            return getattr(self._environment.filesystem, method)(
                self.resolve(path), **kwargs
            )

    def read_prefix(self, path: str, *, max_bytes: int) -> bytes:
        return self._read("read_prefix", path, max_bytes=max_bytes)

    def read_bytes(self, path: str, *, max_bytes: int = 1_000_000) -> bytes:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        data = self.read_prefix(path, max_bytes=max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("File exceeds max_bytes")
        return data

    def read_text(
        self, path: str, *, max_bytes: int = 1_000_000, encoding: str = "utf-8"
    ) -> str:
        return self.read_bytes(path, max_bytes=max_bytes).decode(encoding)

    def read_lines(
        self,
        path: str,
        *,
        offset: int = 1,
        limit: int = 2000,
        max_bytes: int = 1_000_000,
    ) -> bytes:
        return self._read(
            "read_lines", path, offset=offset, limit=limit, max_bytes=max_bytes
        )

    def listdir(self, path: str = ".") -> tuple[str, ...]:
        return self._read("listdir", path)

    def scandir(
        self, path: str = ".", *, max_entries: int = 10_000
    ) -> tuple[WorkspaceEntry, ...]:
        return self._read("scandir", path, max_entries=max_entries)

    def write_text(self, path: str, content: str) -> None:
        editor = self.editor
        editor.apply(editor.prepare_write(path, content))

    def edit_text(self, path: str, old: str, new: str) -> None:
        editor = self.editor
        editor.apply(editor.prepare_edit(path, old, new))

    def delete(self, path: str) -> None:
        editor = self.editor
        editor.apply(editor.prepare_delete_target(path))

    def mkdir(self, path: str) -> None:
        from msgflux.runtime.approvals import agent as approvals  # noqa: PLC0415

        if approvals.approval_batch_active():
            raise PermissionError(
                "Directory creation requires a reviewed workspace tool"
            )
        self._read("mkdir", path)

    async def arun(
        self,
        command: str | tuple[str, ...] | list[str] | ProcessRequest,
        *,
        timeout: float = 30,
        max_output_bytes: int = 1_000_000,
        on_output=None,
    ) -> ProcessResult:
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

    def run(self, command, **kwargs) -> ProcessResult:
        from msgflux._private.executor import Executor  # noqa: PLC0415

        return Executor.get_instance().submit(self.arun, command, **kwargs).result()

    async def aread_prefix(self, path: str, **kwargs) -> bytes:
        return await asyncio.to_thread(self.read_prefix, path, **kwargs)

    async def aread_bytes(self, path: str, **kwargs) -> bytes:
        return await asyncio.to_thread(self.read_bytes, path, **kwargs)

    async def aread_text(self, path: str, **kwargs) -> str:
        return await asyncio.to_thread(self.read_text, path, **kwargs)

    async def aread_lines(self, path: str, **kwargs) -> bytes:
        return await asyncio.to_thread(self.read_lines, path, **kwargs)

    async def alistdir(self, path: str = ".") -> tuple[str, ...]:
        return await asyncio.to_thread(self.listdir, path)

    async def ascandir(self, path: str = ".", **kwargs) -> tuple[WorkspaceEntry, ...]:
        return await asyncio.to_thread(self.scandir, path, **kwargs)

    async def awrite_text(self, path: str, content: str) -> None:
        await asyncio.to_thread(self.write_text, path, content)

    async def aedit_text(self, path: str, old: str, new: str) -> None:
        await asyncio.to_thread(self.edit_text, path, old, new)

    async def adelete(self, path: str) -> None:
        await asyncio.to_thread(self.delete, path)

    async def amkdir(self, path: str) -> None:
        await asyncio.to_thread(self.mkdir, path)


class AgentWorkspaceEditor:
    """Cwd-scoped public editor; prepared proposals carry no authority."""

    def __init__(self, workspace: AgentWorkspace):
        self._workspace = workspace

    def _editor(self, *, require_approval=True):
        return self._workspace._environment.workspace_editor(
            require_approval=require_approval
        )

    def prepare_write(self, path: str, content: str) -> PreparedFileChange:
        with self._workspace._bound():
            return self._editor().prepare_write(self._workspace.resolve(path), content)

    def prepare_create(self, path: str, content: str) -> PreparedFileChange:
        with self._workspace._bound():
            return self._editor().prepare_create(self._workspace.resolve(path), content)

    def prepare_edit(self, path: str, old: str, new: str) -> PreparedFileChange:
        with self._workspace._bound():
            return self._editor().prepare_edit(self._workspace.resolve(path), old, new)

    def prepare_transform(
        self, path: str, transform: Callable[[str], str]
    ) -> PreparedFileChange:
        with self._workspace._bound():
            return self._editor().prepare_transform(
                self._workspace.resolve(path), transform
            )

    def prepare_delete(self, path: str) -> PreparedFileChange:
        with self._workspace._bound():
            return self._editor().prepare_delete(self._workspace.resolve(path))

    def prepare_delete_target(self, path: str) -> PreparedFileChange:
        with self._workspace._bound():
            return self._editor().prepare_delete_target(self._workspace.resolve(path))

    def apply(
        self, change: PreparedFileChange, *, approval=None, approval_store=None
    ) -> None:
        from msgflux.runtime.approvals import agent as approvals  # noqa: PLC0415

        # Outside a protected tool batch this is an explicit host operation.
        with self._workspace._bound():
            if approvals.approval_batch_active():
                if approval is not None or approval_store is not None:
                    raise PermissionError(
                        "A reviewed batch proposal is already being applied"
                    )
                if not approvals.workspace_change_approved(change):
                    raise PermissionError(
                        "Only the consumed reviewed workspace proposal may be applied"
                    )
            self._editor(require_approval=False).apply(
                change, approval=approval, approval_store=approval_store
            )


def require_workspace_authority(workspace=None, *, filesystem=None, environment=None):
    """Validate live workspace, backing resource, binding, and cancellation."""
    from msgflux.runtime.context import get_execution_scope  # noqa: PLC0415

    scope = get_execution_scope()
    live = scope.workspace
    if live is None or not isinstance(live, AgentWorkspace):
        raise PermissionError("A live workspace is required")
    selected = live if workspace is None else workspace
    if not isinstance(selected, AgentWorkspace) or not selected.shares_environment(
        live
    ):
        raise PermissionError("Workspace is not bound to the live workspace")
    driver = selected._environment
    if environment is not None and environment is not driver:
        raise PermissionError("Environment is not bound to the live workspace")
    if filesystem is not None and filesystem is not driver.filesystem:
        raise PermissionError("Filesystem is not bound to the live workspace")
    driver.require_active()
    if scope.abort_signal is not None:
        scope.abort_signal.raise_if_aborted()
    return scope


def resolve_workspace(workspace=None):
    """Select the workspace bound to the active execution, or an explicit instance."""
    from msgflux.runtime.context import get_execution_scope  # noqa: PLC0415

    scope = get_execution_scope()
    workspace = scope.workspace if workspace is None else workspace
    if not isinstance(workspace, AgentWorkspace):
        raise PermissionError("Workspace tools require a live workspace")
    require_workspace_authority(workspace)
    return workspace
