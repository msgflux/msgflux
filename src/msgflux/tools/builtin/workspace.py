"""Workspace tools using only host-bound runtime dependencies."""

import asyncio
from copy import deepcopy
from typing import Optional, Union

from msgflux.data.types import Image
from msgflux.runtime.environment import ExecutionEnvironment, ProcessRequest
from msgflux.runtime.workspace import WorkspaceFilesystem, workspace_path
from msgflux.tools.config import tool_config
from msgflux.tools.handles import ToolLibraryHandle
from msgflux.tools.shell import ShellCommandResult, ShellResult
from msgflux.tools.types import Hidden
from msgflux.tools.workspace_changes import WorkspaceChangeTool
from msgflux.utils.inspect import get_mime_type


def _tool_path(path: str, cwd: str) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    return workspace_path(path if path.startswith("/") else f"{cwd.rstrip('/')}/{path}")


@tool_config(runtime_inputs=["filesystem", "handle"], retry=False)
class ReadFileTool:
    """Read text lines or publish an image from the authorized workspace.

    Args:
        path: File path, absolute or relative to the configured workspace cwd.
        offset: First text line to read, starting at 1 (default: 1).
        limit: Maximum text lines to read (default and ceiling: 2000).
    """

    name = "read"
    display_name = "Read"
    annotations = {
        "path": str,
        "offset": Optional[int],
        "limit": Optional[int],
        "return": str,
    }

    _VISION_GUIDANCE = (
        "Images read by this tool are attached in a subsequent user-role message "
        "linked to the tool call. Inspect that attachment; the tool result only "
        "confirms publication."
    )

    def __init__(self, *, supports_vision: bool = False, cwd: str = "/"):
        if not isinstance(supports_vision, bool):
            raise TypeError("supports_vision must be a boolean")
        self.supports_vision = supports_vision
        self.cwd = workspace_path(cwd)
        self.tool_config = deepcopy(self.tool_config)
        guidance = self.tool_config.get("usage_guidance")
        self.tool_config["usage_guidance"] = (
            "\n\n".join(part for part in (guidance, self._VISION_GUIDANCE) if part)
            if supports_vision
            else guidance
        )

    def __call__(
        self,
        path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        *,
        filesystem: Hidden[WorkspaceFilesystem],
        handle: Hidden[ToolLibraryHandle] = None,
    ) -> str:
        path = _tool_path(path, self.cwd)
        first, count, is_image = self._read_options(path, offset, limit)
        data = (
            filesystem.read_bytes(path)
            if is_image
            else filesystem.read_lines(path, offset=first, limit=count)
        )
        return self._result(path, data, handle)

    async def acall(
        self,
        path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        *,
        filesystem: Hidden[WorkspaceFilesystem],
        handle: Hidden[ToolLibraryHandle] = None,
    ) -> str:
        path = _tool_path(path, self.cwd)
        first, count, is_image = self._read_options(path, offset, limit)
        data = (
            await filesystem.aread_bytes(path)
            if is_image
            else await filesystem.aread_lines(path, offset=first, limit=count)
        )
        return self._result(path, data, handle)

    def _read_options(self, path, offset, limit):
        if any(
            value is not None and (type(value) is not int or value <= 0)
            for value in (offset, limit)
        ):
            raise ValueError("offset and limit must be positive integers")
        is_image = get_mime_type(path).startswith("image/")
        if is_image and (offset is not None or limit is not None):
            raise ValueError("offset and limit are only supported for text files")
        return offset or 1, min(limit or 2000, 2000), is_image

    def _result(self, path: str, data: bytes, handle: ToolLibraryHandle | None) -> str:
        if len(data) > 1_000_000:
            raise ValueError("File exceeds read_file's 1,000,000-byte limit")
        mime_type = get_mime_type(path)
        if mime_type.startswith("image/"):
            if not self.supports_vision:
                raise ValueError("Image reading is disabled for this tool")
            if mime_type not in {"image/png", "image/jpeg", "image/gif", "image/webp"}:
                raise ValueError("Unsupported image format; use PNG, JPEG, GIF or WebP")
            if handle is None:
                raise RuntimeError("Image reading requires an agent inbox")
            published = handle.get_notification().message(
                [Image(data)()], description="Image read by this tool call."
            )
            if published is None:
                raise RuntimeError("Image reading requires an agent inbox")
            return "Image attached as a subsequent user-role message."
        return data.decode("utf-8")


@tool_config(
    tool_kind="shell",
    runtime_inputs=["environment"],
    required_permissions=["process.execute"],
    retry=False,
)
class BashTool:
    """Run Bash in the configured isolated executor, never on the host directly.

    Use an absolute virtual workspace cwd. Execution is limited to 30 seconds
    and 1,000,000 combined stdout/stderr bytes. Requires a configured executor
    with Bash and authorization for every resource accessed by the command.

    Args:
        command: Bash command or ordered batch of independent commands.
        timeout_ms: Per-command execution deadline in milliseconds, at most 30000.
    """

    name = "bash"
    display_name = "Bash"
    annotations = {
        "command": Union[str, list[str]],
        "timeout_ms": Optional[int],
        "return": ShellResult,
    }

    def __init__(self, *, cwd: str = "/"):
        self.cwd = workspace_path(cwd)
        self.tool_config = deepcopy(self.tool_config)

    def __call__(
        self,
        command: Union[str, list[str]],
        timeout_ms: Optional[int] = None,
        *,
        environment: Hidden[ExecutionEnvironment],
    ) -> ShellResult:
        from msgflux.nn.functional import wait_for  # noqa: PLC0415

        return wait_for(
            self.acall,
            command=command,
            timeout_ms=timeout_ms,
            environment=environment,
        )

    async def acall(
        self,
        command: Union[str, list[str]],
        timeout_ms: Optional[int] = None,
        *,
        environment: Hidden[ExecutionEnvironment],
    ) -> ShellResult:
        commands = [command] if isinstance(command, str) else command
        if (
            not isinstance(commands, list)
            or not commands
            or any(
                not isinstance(item, str) or not item.strip() or "\0" in item
                for item in commands
            )
        ):
            raise ValueError("command must be a non-empty string or list of commands")
        if timeout_ms is not None and (type(timeout_ms) is not int or timeout_ms <= 0):
            raise ValueError("timeout_ms must be a positive integer")
        timeout = min(timeout_ms or 30_000, 30_000) / 1000
        remaining = 1_000_000
        outputs = []
        # Validate every request before the first possible external effect.
        requests = [
            ProcessRequest(
                ("bash", "--noprofile", "--norc", "-c", item),
                cwd=self.cwd,
                timeout_seconds=timeout,
                max_output_bytes=remaining,
            )
            for item in commands
        ]
        for prepared_request in requests:
            if remaining <= 0:
                outputs.append(
                    ShellCommandResult(
                        status="not_executed",
                        stderr="Batch output limit exhausted; command not executed.",
                    )
                )
                continue
            request = ProcessRequest(
                prepared_request.argv,
                cwd=prepared_request.cwd,
                timeout_seconds=timeout,
                max_output_bytes=remaining,
            )
            try:
                result = await environment.arun(request)
            except asyncio.TimeoutError:
                outputs.append(
                    ShellCommandResult(status="timed_out", stderr="Command timed out.")
                )
                continue
            remaining -= len(result.stdout) + len(result.stderr)
            outputs.append(
                ShellCommandResult(
                    status="exited",
                    returncode=result.returncode,
                    stdout=result.stdout.decode("utf-8", errors="replace"),
                    stderr=result.stderr.decode("utf-8", errors="replace"),
                )
            )
        return ShellResult(results=tuple(outputs))


@tool_config(runtime_inputs=["filesystem"], retry=False)
class WriteTool(WorkspaceChangeTool):
    """Create or overwrite a UTF-8 file in the authorized workspace.

    Args:
        path: File path, absolute or relative to the configured workspace cwd.
        content: Complete new text to write.
    """

    name = "write"
    display_name = "Write"
    annotations = {"path": str, "content": str, "return": dict[str, str]}

    def prepare_workspace_change(self, arguments, filesystem):
        return self._editor(filesystem).prepare_write(
            _tool_path(arguments["path"], self.cwd), arguments["content"]
        )

    def __call__(
        self, path: str, content: str, *, filesystem: Hidden[WorkspaceFilesystem]
    ) -> dict[str, str]:
        return self._apply({"path": path, "content": content}, filesystem)

    async def acall(
        self, path: str, content: str, *, filesystem: Hidden[WorkspaceFilesystem]
    ) -> dict[str, str]:
        return await asyncio.to_thread(self, path, content, filesystem=filesystem)


@tool_config(runtime_inputs=["filesystem"], retry=False)
class EditTool(WorkspaceChangeTool):
    """Replace one exact, unambiguous text occurrence in a UTF-8 file.

    Args:
        path: File path, absolute or relative to the configured workspace cwd.
        old: Non-empty text that must occur exactly once in the file.
        new: Replacement text; an empty string removes the matched text.
    """

    name = "edit"
    display_name = "Edit"
    annotations = {"path": str, "old": str, "new": str, "return": dict[str, str]}

    def prepare_workspace_change(self, arguments, filesystem):
        return self._editor(filesystem).prepare_edit(
            _tool_path(arguments["path"], self.cwd), arguments["old"], arguments["new"]
        )

    def __call__(
        self, path: str, old: str, new: str, *, filesystem: Hidden[WorkspaceFilesystem]
    ) -> dict[str, str]:
        return self._apply({"path": path, "old": old, "new": new}, filesystem)

    async def acall(
        self, path: str, old: str, new: str, *, filesystem: Hidden[WorkspaceFilesystem]
    ) -> dict[str, str]:
        return await asyncio.to_thread(self, path, old, new, filesystem=filesystem)


__all__ = ["ReadFileTool", "BashTool", "WriteTool", "EditTool"]
