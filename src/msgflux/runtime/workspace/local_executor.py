"""Direct host process execution for explicitly unsandboxed environments."""

from __future__ import annotations

import asyncio
import os
import signal

from msgflux.exceptions import AbortRequestedError
from msgflux.runtime.abort import AbortSignal
from msgflux.runtime.isolation import SandboxCapabilities, SandboxRequirements
from msgflux.runtime.permissions import PermissionSet
from msgflux.runtime.workspace.command_inspection import (
    command_inspection,
    local_identity_matches,
    local_process_identity,
    receipt_identity_error,
    receipt_mapping,
    receipt_resource,
)
from msgflux.runtime.workspace.environment import (
    ProcessExecutor,
    ProcessRequest,
    ProcessResult,
)
from msgflux.runtime.workspace.local import LocalWorkspace
from msgflux.runtime.workspace.process_capture import (
    drain_subprocess,
    process_group_present,
)
from msgflux.runtime.workspace.receipts import (
    MAX_RECEIPT_OUTPUT_BYTES,
    get_command_execution,
)
from msgflux.utils.errors import exception_message


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

    async def execute_stream(  # noqa: C901
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
        execution = get_command_execution()
        resource = local_process_identity(process.pid) or {"pid": process.pid}
        stdout = bytearray()
        stderr = bytearray()

        async def capture(channel, data):
            if execution is not None:
                target = stdout if channel == "stdout" else stderr
                remaining = MAX_RECEIPT_OUTPUT_BYTES - len(target)
                if remaining > 0:
                    target.extend(data[:remaining])
            await on_output(channel, data)

        draining = False
        drain_completed = False
        try:
            if execution is not None:
                await execution.update("launched", resource=resource)
            draining = True
            code = await drain_subprocess(
                process,
                capture,
                max_output_bytes=request.max_output_bytes,
                timeout_seconds=request.timeout_seconds,
                abort_signal=abort_signal,
                owns_process_group=True,
            )
            drain_completed = True
            if execution is not None:
                await execution.update(
                    "completed",
                    resource=resource,
                    returncode=code,
                    stdout=bytes(stdout),
                    stderr=bytes(stderr),
                )
            return ProcessResult(code)
        except BaseException as error:
            if not draining and process.returncode is None:
                signal = AbortSignal()
                signal.abort("Command receipt could not be persisted")

                async def discard(_channel, _data):
                    return None

                try:
                    await drain_subprocess(
                        process,
                        discard,
                        max_output_bytes=request.max_output_bytes,
                        timeout_seconds=max(1, request.timeout_seconds),
                        abort_signal=signal,
                        owns_process_group=True,
                    )
                except BaseException as cleanup_error:
                    error.add_note(f"Local process cleanup failed: {cleanup_error}")
            if execution is not None:
                # A local deadline, abort or output bound is settled by killing
                # this executor's own process group. The receipt records
                # ``terminated`` only when the drain failed from that cleanup,
                # the leader was reaped with a known status, both pipes reached
                # EOF and no member of the owned group survives. A custom
                # executor, a failed completion persistence, an unconfirmed
                # descendant or a crash stays ``unknown`` and keeps blocking.
                settled = None
                if not drain_completed:
                    try:
                        settled = _confirmed_local_termination(
                            process,
                            launched=execution.receipt.state == "launched",
                        )
                    except BaseException as settlement_error:
                        settled = None
                        error.add_note(
                            "Local termination could not be confirmed: "
                            f"{exception_message(settlement_error)}"
                        )
                try:
                    await execution.update(
                        "terminated" if settled is not None else "unknown",
                        resource=resource,
                        returncode=settled,
                        stdout=bytes(stdout),
                        stderr=bytes(stderr),
                    )
                except BaseException as receipt_error:
                    error.add_note(f"Command receipt update failed: {receipt_error}")
            raise

    async def inspect_command(self, receipt):
        """Inspect a saved local command without claiming its absent exit status."""
        error = receipt_identity_error(
            receipt, self.filesystem, backend=self.filesystem.identity.backend
        )
        if error is not None:
            return command_inspection(receipt, "blocked", "mismatch", error)
        try:
            os.close(self.filesystem._open_root())
        except OSError as error:
            return command_inspection(receipt, "blocked", "unavailable", str(error))
        record = receipt_mapping(receipt)
        if (
            record["state"] in {"completed", "terminated"}
            and record.get("returncode") is not None
        ):
            return command_inspection(
                receipt,
                "completed",
                "unchecked",
                "Recorded local termination"
                if record["state"] == "terminated"
                else None,
                stdout=record.get("stdout"),
                stderr=record.get("stderr"),
                returncode=record["returncode"],
            )
        resource = receipt_resource(receipt)
        if resource is None:
            return command_inspection(
                receipt,
                "unknown",
                "unchecked",
                "command has no persisted process resource identity",
            )
        matches, status, reason = local_identity_matches(resource)
        if matches is True:
            return command_inspection(receipt, "running", "present", reason)
        if status == "missing":
            return command_inspection(receipt, "unknown", "missing", reason)
        if status == "mismatch":
            return command_inspection(receipt, "blocked", "mismatch", reason)
        return command_inspection(receipt, "unknown", "unavailable", reason)

    async def terminate_command(self, receipt):  # noqa: C901
        """Signal only a process whose boot and start identity still match."""
        error = receipt_identity_error(
            receipt, self.filesystem, backend=self.filesystem.identity.backend
        )
        if error is not None:
            return command_inspection(receipt, "blocked", "mismatch", error)
        try:
            os.close(self.filesystem._open_root())
        except OSError as error:
            return command_inspection(receipt, "blocked", "unavailable", str(error))
        resource = receipt_resource(receipt)
        if resource is None:
            return command_inspection(
                receipt,
                "blocked",
                "unchecked",
                "command has no persisted process resource identity",
            )
        matches, status, reason = local_identity_matches(resource)
        if matches is not True:
            classification = "blocked" if status == "mismatch" else "unknown"
            return command_inspection(receipt, classification, status, reason)

        pid = resource["pid"]
        pidfd_open = getattr(os, "pidfd_open", None)
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        if pidfd_open is None or pidfd_send_signal is None:
            return command_inspection(
                receipt,
                "blocked",
                "unavailable",
                "safe local termination requires pidfd support",
            )
        try:
            pidfd = pidfd_open(pid)
        except ProcessLookupError:
            return command_inspection(
                receipt, "unknown", "missing", "recorded process exited before signal"
            )
        try:
            matches, status, reason = local_identity_matches(resource)
            if matches is not True:
                return command_inspection(
                    receipt,
                    "blocked" if status == "mismatch" else "unknown",
                    status,
                    reason,
                )
            pidfd_send_signal(pidfd, signal.SIGTERM)
            status, reason = await _wait_for_local_exit(resource, timeout=1)
            if status == "missing":
                return command_inspection(
                    receipt,
                    "unknown",
                    "missing",
                    "process stopped; its exit status is unavailable",
                )
            if status != "present":
                return command_inspection(receipt, "unknown", status, reason)
            pidfd_send_signal(pidfd, signal.SIGKILL)
            status, reason = await _wait_for_local_exit(resource, timeout=1)
            if status == "missing":
                return command_inspection(
                    receipt,
                    "unknown",
                    "missing",
                    "process stopped; its exit status is unavailable",
                )
            if status != "present":
                return command_inspection(receipt, "unknown", status, reason)
            return command_inspection(
                receipt,
                "unknown",
                "present",
                "termination was signaled but the process remains present",
            )
        finally:
            os.close(pidfd)


def _confirmed_local_termination(process, *, launched: bool) -> int | None:
    """Return the reaped leader status only for a fully settled local group.

    ``terminated`` needs four pieces of evidence: this executor launched the
    process, its leader was reaped with a known status, both captured pipes
    reached EOF, and no member of the owned process group survives. A missing
    piece keeps the receipt ``unknown`` instead of certifying the cleanup.
    """
    if not launched:
        return None
    returncode = process.returncode
    if type(returncode) is not int:
        return None
    for stream in (process.stdout, process.stderr):
        if stream is None or not stream.at_eof():
            return None
    if process_group_present(process.pid):
        return None
    return returncode


async def _wait_for_local_exit(resource, *, timeout):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        _matches, status, reason = local_identity_matches(resource)
        if status != "present":
            return status, reason
        await asyncio.sleep(0.05)
    return "present", "recorded local process remains present"


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
