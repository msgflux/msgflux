"""Live workspace and process-execution dependencies supplied by the host."""

from __future__ import annotations

import asyncio
import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from msgflux.runtime.abort import AbortSignal, await_with_abort
from msgflux.runtime.agent_run import get_agent_run
from msgflux.runtime.context import get_execution_context
from msgflux.runtime.isolation import SandboxCapabilities, SandboxRequirements
from msgflux.runtime.permissions import PermissionSet, require_permissions
from msgflux.runtime.workspace.contracts import WriteGuarantee
from msgflux.runtime.workspace.filesystem import WorkspaceFilesystem, workspace_path
from msgflux.runtime.workspace.receipts import (
    CommandExecution,
    bind_command_execution,
    decode_command_receipt,
    get_command_receipt_persist,
    new_command_receipt,
    resolved_command_execution_ids,
    task_command_receipts,
)
from msgflux.runtime.workspace.references import encode_workspace_reference

ProcessOutputCallback = Callable[[Literal["stdout", "stderr"], bytes], Awaitable[None]]
MAX_PROCESS_OUTPUT_CHUNK = 65_536

if TYPE_CHECKING:
    from msgflux.runtime.workspace.backend import WorkspaceBinding
    from msgflux.runtime.workspace.changes import WorkspaceEditor


@dataclass(frozen=True)
class ProcessRequest:
    """Explicit argv, virtual cwd and limits; executor owns environment policy."""

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

    async def execute(
        self,
        request: ProcessRequest,
        *,
        filesystem: WorkspaceFilesystem,
        permissions: PermissionSet,
        requirements: SandboxRequirements,
        abort_signal: AbortSignal | None,
    ) -> ProcessResult:
        """Collect a streaming execution into a bounded ``ProcessResult``."""
        stdout = bytearray()
        stderr = bytearray()

        async def collect(channel, data):
            if channel not in ("stdout", "stderr"):
                raise ValueError("Output channel must be stdout or stderr")
            if not isinstance(data, bytes):
                raise TypeError("Output chunks must be bytes")
            if len(data) > MAX_PROCESS_OUTPUT_CHUNK:
                raise ValueError("Output chunks may not exceed 65536 bytes")
            if len(stdout) + len(stderr) + len(data) > request.max_output_bytes:
                raise RuntimeError("Process executor exceeded its output limit")
            target = stdout if channel == "stdout" else stderr
            target.extend(data)

        result = await self.execute_stream(
            request,
            filesystem=filesystem,
            permissions=permissions,
            requirements=requirements,
            abort_signal=abort_signal,
            on_output=collect,
        )
        if not isinstance(result, ProcessResult):
            raise TypeError("Process executor must return ProcessResult")
        if result.stdout or result.stderr:
            raise RuntimeError(
                "Streaming process executor returned duplicate buffered output"
            )
        return ProcessResult(result.returncode, bytes(stdout), bytes(stderr))

    @abstractmethod
    async def execute_stream(
        self,
        request: ProcessRequest,
        *,
        filesystem: WorkspaceFilesystem,
        permissions: PermissionSet,
        requirements: SandboxRequirements,
        abort_signal: AbortSignal | None,
        on_output: ProcessOutputCallback,
    ) -> ProcessResult:
        """Use this workspace and deliver bounded output chunks incrementally."""
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
    binding: WorkspaceBinding | None = field(default=None, repr=False, compare=False)
    write_guarantee: WriteGuarantee = field(default="atomic_compare", kw_only=True)
    max_edit_bytes: int = field(default=1_000_000, kw_only=True)

    def __post_init__(self):  # noqa: C901
        if type(self.max_edit_bytes) is not int or self.max_edit_bytes <= 0:
            raise ValueError("max_edit_bytes must be a positive integer")
        if not isinstance(self.filesystem, WorkspaceFilesystem):
            raise TypeError("filesystem must be a WorkspaceFilesystem")
        if self.process_executor is not None and not isinstance(
            self.process_executor, ProcessExecutor
        ):
            raise TypeError("process_executor must be a ProcessExecutor or None")
        if not isinstance(self.requirements, SandboxRequirements):
            raise TypeError("requirements must be SandboxRequirements")
        if self.write_guarantee not in ("atomic_compare", "cooperative_compare"):
            raise ValueError("Unknown workspace write guarantee")
        if self.filesystem.requires_binding and self.binding is None:
            raise ValueError("Managed filesystem requires a workspace binding")
        if self.binding is not None:
            from msgflux.runtime.workspace.backend import (  # noqa: PLC0415
                WorkspaceBinding,
            )

            if not isinstance(self.binding, WorkspaceBinding):
                raise TypeError("binding must be a WorkspaceBinding")
            if (
                self.filesystem is not self.binding.filesystem
                or self.process_executor is not self.binding.process_executor
            ):
                raise ValueError("Environment services must belong to its binding")
            self.require_active()
            if self.process_executor is not None:
                self.process_executor.capabilities.require(self.requirements)

    @classmethod
    def from_binding(
        cls,
        binding: WorkspaceBinding,
        *,
        requirements: SandboxRequirements | None = None,
        write_guarantee: WriteGuarantee = "atomic_compare",
        max_edit_bytes: int = 1_000_000,
    ) -> ExecutionEnvironment:
        from msgflux.runtime.workspace.backend import WorkspaceBinding  # noqa: PLC0415

        if not isinstance(binding, WorkspaceBinding):
            raise TypeError("binding must be a WorkspaceBinding")
        kwargs = {} if requirements is None else {"requirements": requirements}
        return cls(
            filesystem=binding.filesystem,
            process_executor=binding.process_executor,
            binding=binding,
            write_guarantee=write_guarantee,
            max_edit_bytes=max_edit_bytes,
            **kwargs,
        )

    def workspace_editor(self, *, require_approval: bool = True) -> WorkspaceEditor:
        """Build an editor with host policy; never infer policy from the backend.

        Capabilities are checked here, not when constructing the environment,
        so a strictly configured environment can still read a cooperative backend.
        """
        from msgflux.runtime.workspace.api import require_workspace_authority  # noqa: PLC0415, I001
        from msgflux.runtime.workspace.changes import WorkspaceEditor  # noqa: PLC0415

        require_workspace_authority(environment=self)
        self.filesystem.require_write_guarantee(self.write_guarantee)
        return WorkspaceEditor(
            self.filesystem,
            require_approval=require_approval,
            write_guarantee=self.write_guarantee,
            max_edit_bytes=self.max_edit_bytes,
        )

    def require_active(self) -> None:
        if self.filesystem.requires_binding and self.binding is None:
            raise PermissionError("Managed filesystem requires a workspace binding")
        if self.binding is not None:
            self.binding.require_active()

    async def arun(  # noqa: C901
        self,
        request: ProcessRequest,
        *,
        on_output: ProcessOutputCallback | None = None,
    ) -> ProcessResult:
        from msgflux.runtime.workspace.api import require_workspace_authority  # noqa: PLC0415, I001

        if not isinstance(request, ProcessRequest):
            raise TypeError("Expected ProcessRequest")
        scope = require_workspace_authority(environment=self)
        require_permissions(("process.execute",))
        executor = self.process_executor
        if executor is None:
            raise PermissionError("No process executor configured")
        executor.capabilities.require(self.requirements)
        if executor.supports_workspace(self.filesystem) is not True:
            raise PermissionError("Process executor cannot use this workspace")
        if scope.abort_signal is not None:
            scope.abort_signal.raise_if_aborted()
        if on_output is not None and not callable(on_output):
            raise TypeError("on_output must be callable")
        execution_context = get_execution_context()
        receipt_execution = await _begin_command_receipt(execution_context, self, scope)
        streamed_bytes = 0
        output_lock = asyncio.Lock()

        async def deliver(channel, data):
            nonlocal streamed_bytes
            async with output_lock:
                if channel not in ("stdout", "stderr"):
                    raise ValueError("Output channel must be stdout or stderr")
                if not isinstance(data, bytes):
                    raise TypeError("Output chunks must be bytes")
                if len(data) > MAX_PROCESS_OUTPUT_CHUNK:
                    raise ValueError("Output chunks may not exceed 65536 bytes")
                streamed_bytes += len(data)
                if streamed_bytes > request.max_output_bytes:
                    raise RuntimeError("Process executor exceeded its output limit")
                result = on_output
                if result is not None:
                    value = result(channel, data)
                    if not inspect.isawaitable(value):
                        raise TypeError("on_output must return an awaitable")
                    await value

        bind = (
            bind_command_execution(receipt_execution)
            if receipt_execution is not None
            else nullcontext()
        )
        try:
            with bind:
                operation = (
                    executor.execute_stream(
                        request,
                        filesystem=self.filesystem,
                        permissions=scope.permissions,
                        requirements=self.requirements,
                        abort_signal=scope.abort_signal,
                        on_output=deliver,
                    )
                    if on_output is not None
                    else executor.execute(
                        request,
                        filesystem=self.filesystem,
                        permissions=scope.permissions,
                        requirements=self.requirements,
                        abort_signal=scope.abort_signal,
                    )
                )
                result = await asyncio.wait_for(
                    await_with_abort(operation, scope.abort_signal),
                    timeout=request.timeout_seconds,
                )
        except BaseException:
            if receipt_execution is not None and receipt_execution.receipt.state in {
                "intent",
                "launched",
            }:
                await receipt_execution.update("unknown")
            raise
        if not isinstance(result, ProcessResult):
            raise TypeError("Process executor must return ProcessResult")
        if on_output is not None and (result.stdout or result.stderr):
            raise RuntimeError(
                "Streaming process executor returned duplicate buffered output"
            )
        if len(result.stdout) + len(result.stderr) > request.max_output_bytes:
            raise RuntimeError("Process executor violated its output limit")
        if on_output is not None and streamed_bytes > request.max_output_bytes:
            raise RuntimeError("Process executor violated its output limit")
        if receipt_execution is not None and receipt_execution.receipt.state in {
            "intent",
            "launched",
        }:
            await receipt_execution.update(
                "completed",
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result


async def _begin_command_receipt(execution, environment, scope):
    """Persist a command intent before handing control to a process adapter."""
    recorder = execution.get("task_activity_recorder")
    task_handle = execution.get("task_handle")
    task_id = getattr(task_handle, "task_id", None)
    agent_persist = get_command_receipt_persist()
    if recorder is None and agent_persist is None:
        return None
    _reject_unknown_prior_command(execution, scope, recorder=recorder)
    identity = environment.filesystem.identity
    receipt = new_command_receipt(
        workspace_reference=encode_workspace_reference(scope.workspace),
        backend=identity.backend,
        owner_id=getattr(task_handle, "_owner_id", None) or scope.principal,
        run_id=scope.run_id,
        task_id=task_id,
        tool_call_id=execution.get("tool_call_id"),
        message_offset=execution.get("tool_call_message_offset"),
    )

    def persist(current):
        if recorder is not None:
            activity = recorder.add(
                kind="command_receipt",
                summary=f"Workspace command {current.state}",
                metadata={"receipt": current.to_dict()},
            )
            if activity is None:
                raise RuntimeError("Could not persist workspace command receipt")
            return
        agent_persist(current)

    command = CommandExecution(receipt, persist)
    await command.update("intent")
    return command


def _reject_unknown_prior_command(execution, scope, *, recorder) -> None:
    """Prevent a live Agent from starting another command after an unknown result."""
    if recorder is not None:
        receipts = task_command_receipts(recorder._store, recorder.task_id)
    else:
        run = get_agent_run()
        raw = run.get_extension("command_receipts", []) if run is not None else []
        receipts = tuple(decode_command_receipt(item) for item in raw)
    if not receipts:
        return
    state = None
    store = execution.get("checkpoint_store")
    if store is not None and all(
        isinstance(value, str) and value
        for value in (scope.namespace, scope.thread_id, scope.run_id)
    ):
        state = store.load_state(scope.namespace, scope.thread_id, scope.run_id)
    extensions = (
        state.get("runtime", {}).get("extensions", {})
        if isinstance(state, Mapping)
        else {}
    )
    resolved = resolved_command_execution_ids(extensions, receipts=receipts)
    if any(
        receipt.state == "unknown" and receipt.execution_id not in resolved
        for receipt in receipts
    ):
        from msgflux.exceptions import TaskPauseRequestedError  # noqa: PLC0415

        raise TaskPauseRequestedError(
            message=(
                "A previous workspace command outcome is unknown; host "
                "reconciliation is required before another command can start."
            )
        )
