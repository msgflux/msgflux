"""Host-owned backend factories and live, non-serializable resource bindings."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Literal

from msgflux.runtime.abort import AbortSignal
from msgflux.runtime.environment import ProcessExecutor
from msgflux.runtime.workspace import InMemoryWorkspace, WorkspaceFilesystem
from msgflux.runtime.workspace_contracts import WorkspaceIdentity


class WorkspaceBackend(ABC):
    """Reusable host service; never owns execution-local grants or principals.

    Adapters must clean up partially opened connections on failure/cancellation.
    Reconnection verifies an existing resource, never silently creates a new one.
    """

    @abstractmethod
    async def open(
        self, workspace_id: str, *, abort_signal: AbortSignal | None = None
    ) -> WorkspaceBinding:
        """Open a resource and return a new independently releasable binding."""

    async def reconnect(
        self,
        workspace_id: str,
        identity: WorkspaceIdentity,
        *,
        abort_signal: AbortSignal | None = None,
    ) -> WorkspaceBinding:
        raise NotImplementedError("Backend does not support reconnection")

    @abstractmethod
    async def _release(self, binding: WorkspaceBinding) -> None:
        """Release only this connection; never implicitly destroy its resource."""


class WorkspaceBinding:
    """One live connection to a backend resource, with no persisted authority.

    The host must drain/cancel active operations before closing. Closing gates
    new mediated operations but is not a remote process-termination primitive.
    Instances are intended for one async event loop, not concurrent loop sharing.
    """

    def __init__(
        self,
        backend: WorkspaceBackend,
        filesystem: WorkspaceFilesystem,
        process_executor: ProcessExecutor | None = None,
        *,
        ownership: Literal["owned", "borrowed"] = "borrowed",
    ):
        if not isinstance(backend, WorkspaceBackend):
            raise TypeError("backend must be a WorkspaceBackend")
        if not isinstance(filesystem, WorkspaceFilesystem):
            raise TypeError("filesystem must be a WorkspaceFilesystem")
        if process_executor is not None:
            if not isinstance(process_executor, ProcessExecutor):
                raise TypeError("process_executor must be a ProcessExecutor")
            if process_executor.supports_workspace(filesystem) is not True:
                raise PermissionError("Process executor cannot use this workspace")
        if ownership not in ("owned", "borrowed"):
            raise ValueError("ownership must be owned or borrowed")
        self._backend = backend
        self._filesystem = filesystem
        self._process_executor = process_executor
        self._identity = filesystem.identity
        self._ownership = ownership
        self._state = "open"
        self._close_lock = asyncio.Lock()
        filesystem._require_binding()

    @property
    def backend(self) -> WorkspaceBackend:
        return self._backend

    @property
    def filesystem(self) -> WorkspaceFilesystem:
        return self._filesystem

    @property
    def process_executor(self) -> ProcessExecutor | None:
        return self._process_executor

    @property
    def identity(self) -> WorkspaceIdentity:
        return self._identity

    @property
    def ownership(self) -> str:
        return self._ownership

    @property
    def state(self) -> str:
        return self._state

    def require_active(self) -> None:
        if self._state != "open":
            raise PermissionError("Workspace binding is not open")
        if self.filesystem.identity != self.identity:
            raise PermissionError("Workspace binding resource identity changed")

    async def aclose(self) -> None:
        """Detach once; a failed/uncertain release is never retried implicitly."""
        async with self._close_lock:
            if self._state == "closed":
                return
            if self._state == "release_failed":
                raise RuntimeError(
                    "Workspace release failed; host reconciliation required"
                )
            self._state = "closing"
            try:
                await self.backend._release(self)
            except BaseException:
                self._state = "release_failed"
                raise
            self._state = "closed"

    async def __aenter__(self):
        self.require_active()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.aclose()


class InMemoryWorkspaceBackend(WorkspaceBackend):
    """Reference backend retaining resources for its own process-local lifetime.

    Each open creates independent files; reconnect opens another binding to the
    exact retained resource. Release does not delete files. Discarding the backend
    and all its bindings releases storage; this is not cross-process durability.
    """

    def __init__(self, files: Mapping[str, bytes] | None = None):
        self._initial_files = dict(files or {})
        self._resources: dict[WorkspaceIdentity, InMemoryWorkspace] = {}

    async def open(
        self, workspace_id: str, *, abort_signal: AbortSignal | None = None
    ) -> WorkspaceBinding:
        if abort_signal is not None:
            abort_signal.raise_if_aborted()
        filesystem = InMemoryWorkspace(workspace_id, self._initial_files)
        binding = WorkspaceBinding(self, filesystem, ownership="owned")
        self._resources[filesystem.identity] = filesystem
        return binding

    async def reconnect(
        self,
        workspace_id: str,
        identity: WorkspaceIdentity,
        *,
        abort_signal: AbortSignal | None = None,
    ) -> WorkspaceBinding:
        if abort_signal is not None:
            abort_signal.raise_if_aborted()
        if not isinstance(identity, WorkspaceIdentity):
            raise TypeError("identity must be a WorkspaceIdentity")
        filesystem = self._resources.get(identity)
        if filesystem is None or filesystem.workspace_id != workspace_id:
            raise FileNotFoundError(
                "Workspace resource is unavailable or identity changed"
            )
        return WorkspaceBinding(self, filesystem)

    async def _release(self, binding: WorkspaceBinding) -> None:
        if binding.backend is not self:
            raise ValueError("Binding belongs to another backend")
