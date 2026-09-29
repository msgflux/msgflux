"""Direct host process execution for explicitly unsandboxed environments."""

from __future__ import annotations

import asyncio
import os

from msgflux.exceptions import AbortRequestedError
from msgflux.runtime.abort import AbortSignal
from msgflux.runtime.environment import ProcessExecutor, ProcessRequest, ProcessResult
from msgflux.runtime.isolation import SandboxCapabilities, SandboxRequirements
from msgflux.runtime.permissions import PermissionSet
from msgflux.runtime.process_capture import drain_subprocess
from msgflux.runtime.workspace_local import LocalWorkspace


class LocalProcessExecutor(ProcessExecutor):
    """Execute argv on the host, mapping virtual cwd to a local workspace.

    This executor inherits the host environment and provides no OS isolation.
    ``process.execute`` authorizes commands with normal host access, including
    paths outside the workspace. Output, deadlines and cancellation are bounded
    by the existing subprocess capture helper.
    """

    capabilities = SandboxCapabilities()

    def __init__(self, filesystem: LocalWorkspace):
        if not isinstance(filesystem, LocalWorkspace):
            raise TypeError("LocalProcessExecutor requires LocalWorkspace")
        self.filesystem = filesystem

    def supports_workspace(self, filesystem) -> bool:
        return filesystem is self.filesystem

    async def execute_stream(
        self,
        request: ProcessRequest,
        *,
        filesystem,
        permissions: PermissionSet,
        requirements: SandboxRequirements,
        abort_signal: AbortSignal | None,
        on_output,
    ) -> ProcessResult:
        if not callable(on_output):
            raise TypeError("on_output must be callable")
        if not self.supports_workspace(filesystem):
            raise PermissionError("Process executor cannot use this workspace")
        self.capabilities.require(requirements)
        if not permissions.allows(("process.execute",)):
            raise PermissionError("Missing process.execute permission")
        if abort_signal is not None:
            abort_signal.raise_if_aborted()
        # Verify that the host-owned root still identifies the selected resource.
        os.close(self.filesystem._open_root())
        cwd = os.path.join(filesystem.host_root, request.cwd.lstrip("/"))
        launch = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *request.argv,
                cwd=cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        )
        try:
            process = await asyncio.shield(launch)
        except BaseException as error:
            # Cancellation may win while the OS is still creating the child.
            cleanup = asyncio.create_task(_cleanup_launch(launch, request))
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
                error.add_note(f"Process launch cleanup failed: {cleanup_error}")
            raise
        code = await drain_subprocess(
            process,
            on_output,
            max_output_bytes=request.max_output_bytes,
            timeout_seconds=request.timeout_seconds,
            abort_signal=abort_signal,
            owns_process_group=True,
        )
        return ProcessResult(code)


async def _cleanup_launch(launch, request):
    try:
        process = await launch
    except Exception:
        return  # Launch failed before producing a process.
    signal = AbortSignal()
    signal.abort("Process launch cancelled")

    async def discard(_channel, _data):
        pass

    try:
        await drain_subprocess(
            process,
            discard,
            max_output_bytes=request.max_output_bytes,
            timeout_seconds=request.timeout_seconds,
            abort_signal=signal,
            owns_process_group=True,
        )
    except AbortRequestedError:
        pass


__all__ = ["LocalProcessExecutor"]
