"""Explicit tool dependency for a single filesystem and execution environment."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import contextmanager
from uuid import uuid4

from msgflux.runtime.abort import AbortSignal
from msgflux.runtime.isolation import SandboxRequirements
from msgflux.runtime.permissions import (
    PermissionSet,
    ResourcePermission,
    intersect_permissions,
    require_permissions,
)
from msgflux.runtime.workspace.changes import PreparedFileChange
from msgflux.runtime.workspace.command_inspection import CommandInspection
from msgflux.runtime.workspace.contracts import WorkspaceEntry, WorkspaceIdentity
from msgflux.runtime.workspace.environment import (
    ExecutionEnvironment,
    ProcessRequest,
    ProcessResult,
)
from msgflux.runtime.workspace.filesystem import workspace_path
from msgflux.runtime.workspace.local import LocalWorkspace
from msgflux.runtime.workspace.local_executor import LocalProcessExecutor
from msgflux.runtime.workspace.receipts import CommandReceipt


class AgentWorkspace:
    """Tools declare this dependency; operations use the live scope's authority.

    Construct with an existing environment, or use ``AgentWorkspace.local`` for a
    project with host execution. Local commands are not an OS sandbox. Relative
    file and command paths use the same virtual cwd. No context manager or
    persistent connection is required for the local factory.
    """

    def __init__(
        self,
        environment: ExecutionEnvironment,
        *,
        cwd: str = "/",
        permissions: PermissionSet | None = None,
    ):
        if not isinstance(environment, ExecutionEnvironment):
            raise TypeError("AgentWorkspace requires an ExecutionEnvironment")
        self._environment = environment
        # Only AgentWorkspace.open sets this. Environments passed by callers
        # remain borrowed, even when they happen to carry a live binding.
        self._owned_binding = None
        self._cwd = workspace_path(cwd)
        if permissions is not None and not isinstance(permissions, PermissionSet):
            raise TypeError("permissions must be a PermissionSet or None")
        self._permissions = permissions or PermissionSet()

    @classmethod
    def local(
        cls,
        root,
        *,
        read_only: bool = False,
        permissions: PermissionSet | None = None,
    ) -> AgentWorkspace:
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
        if permissions is not None and not isinstance(permissions, PermissionSet):
            raise TypeError("permissions must be a PermissionSet or None")
        ceiling = defaults if permissions is None else permissions
        filesystem = LocalWorkspace(
            f"local-{uuid4().hex}",
            os.path.realpath(os.fspath(root)),
            read_only=read_only,
        )
        environment = ExecutionEnvironment(
            filesystem,
            process_executor=None if read_only else LocalProcessExecutor(filesystem),
            requirements=SandboxRequirements(),
            write_guarantee="cooperative_compare",
        )
        return cls(environment, permissions=ceiling)

    @classmethod
    async def open(
        cls,
        backend,
        workspace_id: str,
        *,
        permissions: PermissionSet | None = None,
        cwd: str = "/",
        requirements: SandboxRequirements | None = None,
        write_guarantee="atomic_compare",
        max_edit_bytes: int = 1_000_000,
        abort_signal: AbortSignal | None = None,
    ) -> AgentWorkspace:
        """Open a backend resource and own its connection until ``aclose``.

        The workspace keeps only the binding returned by this one open. Cwd
        views share its environment, while ``from_environment`` stays borrowed.
        """
        from msgflux.runtime.workspace.backend import WorkspaceBackend  # noqa: PLC0415

        if not isinstance(backend, WorkspaceBackend):
            raise TypeError("backend must be a WorkspaceBackend")
        if not isinstance(workspace_id, str):
            raise TypeError("workspace_id must be a string")
        if permissions is not None and not isinstance(permissions, PermissionSet):
            raise TypeError("permissions must be a PermissionSet or None")

        binding = await backend.open(workspace_id, abort_signal=abort_signal)
        return await cls._from_binding(
            binding,
            cwd=cwd,
            permissions=permissions,
            requirements=requirements,
            write_guarantee=write_guarantee,
            max_edit_bytes=max_edit_bytes,
            expected_workspace_id=workspace_id,
            validate_cwd=False,
        )

    @classmethod
    async def reconnect(
        cls,
        backend,
        workspace_id: str,
        identity: WorkspaceIdentity,
        *,
        permissions: PermissionSet | None = None,
        cwd: str = "/",
        requirements: SandboxRequirements | None = None,
        write_guarantee="atomic_compare",
        max_edit_bytes: int = 1_000_000,
        abort_signal: AbortSignal | None = None,
    ) -> AgentWorkspace:
        """Reconnect to an existing resource with freshly supplied authority."""
        from msgflux.runtime.workspace.backend import WorkspaceBackend  # noqa: PLC0415

        if not isinstance(backend, WorkspaceBackend):
            raise TypeError("backend must be a WorkspaceBackend")
        if not isinstance(workspace_id, str):
            raise TypeError("workspace_id must be a string")
        if not isinstance(identity, WorkspaceIdentity):
            raise TypeError("identity must be a WorkspaceIdentity")
        if permissions is not None and not isinstance(permissions, PermissionSet):
            raise TypeError("permissions must be a PermissionSet or None")
        binding = await backend.reconnect(
            workspace_id, identity, abort_signal=abort_signal
        )
        return await cls._from_binding(
            binding,
            cwd=cwd,
            permissions=permissions,
            requirements=requirements,
            write_guarantee=write_guarantee,
            max_edit_bytes=max_edit_bytes,
            expected_workspace_id=workspace_id,
            expected_identity=identity,
            validate_cwd=True,
        )

    @classmethod
    async def _from_binding(
        cls,
        binding,
        *,
        cwd,
        permissions,
        requirements,
        write_guarantee,
        max_edit_bytes,
        expected_workspace_id=None,
        expected_identity=None,
        validate_cwd=False,
    ):
        from msgflux.runtime.workspace.backend import WorkspaceBinding  # noqa: PLC0415

        if not isinstance(binding, WorkspaceBinding):
            raise TypeError("backend must return a WorkspaceBinding")
        try:
            if expected_workspace_id is not None and (
                binding.filesystem.workspace_id != expected_workspace_id
            ):
                raise ValueError("Backend returned a different workspace resource")
            if expected_identity is not None and binding.identity != expected_identity:
                raise ValueError("Backend returned a different workspace identity")
            environment = ExecutionEnvironment.from_binding(
                binding,
                requirements=requirements,
                write_guarantee=write_guarantee,
                max_edit_bytes=max_edit_bytes,
            )
            workspace = cls(environment, cwd=cwd, permissions=permissions)
            if validate_cwd:
                cls._validate_cwd(environment.filesystem, workspace.cwd)
        except BaseException as failure:
            # Once open() has returned, a constructor error must not leak the
            # binding. Shield release so a concurrent cancellation cannot
            # interrupt cleanup; keep the original failure as the exception.
            cleanup = asyncio.create_task(binding.aclose())
            while not cleanup.done():
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            try:
                cleanup.result()
            except BaseException as cleanup_error:
                failure.add_note(f"Workspace binding cleanup failed: {cleanup_error}")
            raise
        workspace._owned_binding = binding
        return workspace

    @staticmethod
    def _validate_cwd(filesystem, cwd: str) -> None:
        """Check cwd through trusted backend hooks without granting permissions."""
        if cwd != "/" and filesystem._trusted_is_directory(cwd) is not True:
            raise NotADirectoryError(cwd)

    @classmethod
    def from_environment(
        cls,
        environment: ExecutionEnvironment,
        *,
        cwd: str = "/",
        permissions: PermissionSet | None = None,
    ) -> AgentWorkspace:
        return cls(environment, cwd=cwd, permissions=permissions)

    async def aclose(self) -> None:
        """Close the binding opened by this workspace, if it owns one."""
        if self._owned_binding is not None:
            await self._owned_binding.aclose()

    def permission(self, path: str, action: str) -> ResourcePermission:
        """Describe an exact cwd-relative resource grant; does not authorize it."""
        return self._environment.filesystem.permission(self.resolve(path), action)

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
    def permissions(self) -> PermissionSet:
        return self._permissions

    def effective_permissions(self, requested: PermissionSet | None) -> PermissionSet:
        """Apply this workspace's immutable ceiling to live scope permissions."""
        if requested is None:
            requested = self._permissions
        if not isinstance(requested, PermissionSet):
            raise TypeError("permissions must be a PermissionSet or None")
        grants = {
            grant
            for grant in requested.grants
            if not grant.startswith(("filesystem.", "process."))
        }
        grants.update(requested.grants & self._permissions.grants)
        resources = {
            resource
            for resource in requested.resources
            if not resource.resource.startswith("workspace:")
        }
        resources.update(
            intersect_permissions(
                self._permissions,
                requested,
                workspace_id=self.workspace_id,
            ).resources
        )
        if "process.workspace" in grants:
            # Executors receive concrete resource grants; this explicit capability
            # authorizes the whole mount of this workspace, never another root.
            resources.add(
                self._environment.filesystem.permission("/", "process.workspace")
            )
        return PermissionSet(frozenset(grants), frozenset(resources))

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
        view = AgentWorkspace(
            self._environment, cwd=self.resolve(cwd), permissions=self._permissions
        )
        # Views borrow the owner's connection. Closing any binding still gates
        # every view through the shared environment.
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
            live_scope = require_workspace_authority(self)
            with execution_context(
                scope=ExecutionScope(workspace=self, permissions=live_scope.permissions)
            ):
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

    async def ainspect_command(self, receipt: CommandReceipt) -> CommandInspection:
        """Inspect a host-loaded command receipt against this live workspace."""
        self.require_active()
        executor = self._environment.process_executor
        inspect_command = getattr(executor, "inspect_command", None)
        if not callable(inspect_command):
            raise PermissionError("Workspace has no command inspection executor")
        with self._bound():
            require_permissions(("process.execute",))
            if getattr(executor, "requires_workspace_process_grant", False):
                require_permissions(
                    (),
                    (
                        self._environment.filesystem.permission(
                            "/", "process.workspace"
                        ),
                    ),
                )
            return await inspect_command(receipt)

    def inspect_command(self, receipt: CommandReceipt) -> CommandInspection:
        """Synchronously inspect a command receipt from a host control path."""
        from msgflux._private.executor import Executor  # noqa: PLC0415

        return Executor.get_instance().submit(self.ainspect_command, receipt).result()

    async def aterminate_command(self, receipt: CommandReceipt) -> CommandInspection:
        """Explicitly signal a still-owned command after identity validation."""
        self.require_active()
        executor = self._environment.process_executor
        terminate_command = getattr(executor, "terminate_command", None)
        if not callable(terminate_command):
            raise PermissionError("Workspace has no command termination executor")
        with self._bound():
            require_permissions(("process.execute",))
            if getattr(executor, "requires_workspace_process_grant", False):
                require_permissions(
                    (),
                    (
                        self._environment.filesystem.permission(
                            "/", "process.workspace"
                        ),
                    ),
                )
            return await terminate_command(receipt)

    def terminate_command(self, receipt: CommandReceipt) -> CommandInspection:
        """Synchronously request termination from a host control path."""
        from msgflux._private.executor import Executor  # noqa: PLC0415

        return Executor.get_instance().submit(self.aterminate_command, receipt).result()

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
