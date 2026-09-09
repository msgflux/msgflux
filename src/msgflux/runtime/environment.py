"""Live workspace and process-execution dependencies supplied by the host."""

from __future__ import annotations

import asyncio
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from msgflux.runtime.abort import AbortSignal, await_with_abort
from msgflux.runtime.isolation import SandboxCapabilities, SandboxRequirements
from msgflux.runtime.permissions import PermissionSet, require_permissions
from msgflux.runtime.workspace import WorkspaceFilesystem, workspace_path


@dataclass(frozen=True)
class ProcessRequest:
    """Explicit argv, virtual cwd and limits; no inherited host environment."""

    argv: tuple[str, ...]
    cwd: str = "/"
    timeout_seconds: float = 30
    max_output_bytes: int = 1_000_000

    def __post_init__(self):
        if isinstance(self.argv, (str, bytes)):
            raise TypeError("argv must be a collection of strings, not a shell command")
        argv = tuple(self.argv)
        if (
            not argv
            or not argv[0]
            or any(not isinstance(arg, str) or "\0" in arg for arg in argv)
        ):
            raise ValueError("argv must contain a program and valid string arguments")
        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "cwd", workspace_path(self.cwd))
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be positive and finite")
        if type(self.max_output_bytes) is not int or self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be a positive integer")


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes = b""
    stderr: bytes = b""

    def __post_init__(self):
        if type(self.returncode) is not int:
            raise TypeError("returncode must be an integer")
        if not isinstance(self.stdout, bytes) or not isinstance(self.stderr, bytes):
            raise TypeError("Process output must be bytes")


class ProcessExecutor(ABC):
    """Trusted adapter: enforce grants, limits and cleanup before returning.

    Implementations must reject unsupported policies and terminate/reap children
    on cancellation. A method returning True is not proof of OS isolation.
    No concrete process executor is provided by this module.
    """

    @property
    @abstractmethod
    def capabilities(self) -> SandboxCapabilities:
        raise NotImplementedError

    @abstractmethod
    def supports_workspace(self, filesystem: WorkspaceFilesystem) -> bool:
        raise NotImplementedError

    @abstractmethod
    async def execute(
        self,
        request: ProcessRequest,
        *,
        filesystem: WorkspaceFilesystem,
        permissions: PermissionSet,
        requirements: SandboxRequirements,
        abort_signal: AbortSignal | None,
    ) -> ProcessResult:
        """Use this workspace, never silently substitute the host filesystem."""
        raise NotImplementedError


@dataclass(frozen=True)
class ExecutionEnvironment:
    filesystem: WorkspaceFilesystem
    process_executor: ProcessExecutor | None = None
    requirements: SandboxRequirements = field(
        default_factory=lambda: SandboxRequirements(
            {"filesystem", "network", "process", "resource_limits"}
        )
    )

    def __post_init__(self):
        if not isinstance(self.filesystem, WorkspaceFilesystem):
            raise TypeError("filesystem must be a WorkspaceFilesystem")
        if self.process_executor is not None and not isinstance(
            self.process_executor, ProcessExecutor
        ):
            raise TypeError("process_executor must be a ProcessExecutor or None")
        if not isinstance(self.requirements, SandboxRequirements):
            raise TypeError("requirements must be SandboxRequirements")

    async def arun(self, request: ProcessRequest) -> ProcessResult:
        from msgflux.runtime.context import get_execution_scope  # noqa: PLC0415

        if not isinstance(request, ProcessRequest):
            raise TypeError("Expected ProcessRequest")
        scope = get_execution_scope()
        if scope.environment is not self:
            raise PermissionError("Environment is not bound to the current execution")
        require_permissions(("process.execute",))
        executor = self.process_executor
        if executor is None:
            raise PermissionError("No isolated process executor configured")
        executor.capabilities.require(self.requirements)
        if executor.supports_workspace(self.filesystem) is not True:
            raise PermissionError("Process executor cannot use this workspace")
        if scope.abort_signal is not None:
            scope.abort_signal.raise_if_aborted()
        result = await asyncio.wait_for(
            await_with_abort(
                executor.execute(
                    request,
                    filesystem=self.filesystem,
                    permissions=scope.permissions,
                    requirements=self.requirements,
                    abort_signal=scope.abort_signal,
                ),
                scope.abort_signal,
            ),
            timeout=request.timeout_seconds,
        )
        if not isinstance(result, ProcessResult):
            raise TypeError("Process executor must return ProcessResult")
        if len(result.stdout) + len(result.stderr) > request.max_output_bytes:
            raise RuntimeError("Process executor violated its output limit")
        return result
