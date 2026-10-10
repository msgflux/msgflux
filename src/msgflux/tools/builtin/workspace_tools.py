"""Workspace tools using only host-bound runtime dependencies."""

import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

from msgflux.data.types import Image
from msgflux.runtime.workspace.api import AgentWorkspace, resolve_workspace
from msgflux.runtime.workspace.environment import ProcessRequest
from msgflux.runtime.workspace.filesystem import (
    _read_byte_limit_error,
    _ReadByteLimitError,
)
from msgflux.tools.config import tool_config
from msgflux.tools.handles import ToolLibraryHandle
from msgflux.tools.shell import ShellCommandResult, ShellResult
from msgflux.tools.specs import ContextBinding
from msgflux.tools.types import Hidden
from msgflux.tools.workspace_changes import WorkspaceChangeTool
from msgflux.utils.inspect import get_mime_type


def _tool_path(path: str, workspace: AgentWorkspace) -> str:
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    return workspace.resolve(path)


def _read_offloaded_tool_result(
    path: str,
    *,
    workspace: AgentWorkspace,
    offset: int,
    limit: int,
    max_bytes: int = 1_000_000,
) -> bytes | None:
    """Read a result only when path names content in the active thread store."""
    # This private import avoids making the workspace runtime depend on Agent.
    from msgflux.nn.modules.agent.resources import (  # noqa: PLC0415
        _get_tool_result_store,
    )
    from msgflux.runtime.permissions import require_permissions  # noqa: PLC0415
    from msgflux.runtime.tool_results import (  # noqa: PLC0415
        LocalToolResultStore,
        ToolResultRef,
    )
    from msgflux.runtime.workspace.api import (  # noqa: PLC0415
        require_workspace_authority,
    )

    store = _get_tool_result_store(create=False)
    if store is None:
        return None
    if not isinstance(store, LocalToolResultStore):
        raise TypeError("Managed tool result store must be local")
    try:
        relative = Path(path).relative_to(store.root)
    except ValueError:
        return None
    if len(relative.parts) != 2 or relative.parts[1] != "content":
        return None
    result_id = relative.parts[0]

    # The artifact is outside the project filesystem. Preserve the current
    # workspace binding and live read capability, but do not grant filesystem
    # access to the host path or route any other operation through this store.
    with workspace._bound():
        require_workspace_authority(workspace)
        require_permissions(("filesystem.read",))
        reference = store.get(result_id)
        if not isinstance(reference, ToolResultRef):
            raise TypeError("Tool result store returned an invalid reference")
        store.verify(reference)
        return store._read_lines(
            reference,
            offset=offset,
            limit=limit,
            max_bytes=max_bytes,
        )


@tool_config(runtime_inputs=["workspace", "handle"], retry=False)
class ReadFileTool:
    """Read text lines or publish an image from the authorized workspace.

    Args:
        path: File path, absolute or relative to the configured workspace cwd.
        offset: First text line to read, starting at 1 (default: 1).
        limit: Maximum text lines to read (default and ceiling: 2000).
    """

    name = "read"
    display_name = "Read"
    default_usage_guidance = (
        "Read only the lines needed for the task. Use offset and limit to "
        "paginate text. If a page exceeds the configured byte limit, request "
        "fewer lines by reducing limit. If a single line exceeds the byte "
        "limit, read cannot retrieve it; use another available tool, such as "
        "Bash, to extract a smaller portion. Rejected reads return no file content."
    )
    annotations = {
        "path": str,
        "offset": Optional[int],
        "limit": Optional[int],
        "return": str,
    }

    _VISION_GUIDANCE = (
        "Images read by this tool are attached in a subsequent user-role message "
        "linked to the tool call. Inspect that attachment; the tool result only "
        "confirms publication. For images, set offset and limit to null."
    )

    def __init__(
        self,
        *,
        supports_vision: bool = False,
        max_image_bytes: int = 1_000_000,
        max_text_bytes: int = 32 * 1024,
    ):
        if not isinstance(supports_vision, bool):
            raise TypeError("supports_vision must be a boolean")
        self.supports_vision = supports_vision
        if type(max_image_bytes) is not int or max_image_bytes <= 0:
            raise ValueError("max_image_bytes must be a positive integer")
        self.max_image_bytes = max_image_bytes
        if type(max_text_bytes) is not int or max_text_bytes <= 0:
            raise ValueError("max_text_bytes must be a positive integer")
        self.max_text_bytes = max_text_bytes
        self.tool_config = deepcopy(self.tool_config)
        configured_guidance = self.tool_config.get("usage_guidance")
        if (
            supports_vision
            and isinstance(configured_guidance, str)
            and configured_guidance
        ):
            self.tool_config["usage_guidance"] = (
                configured_guidance + "\n\n" + self._VISION_GUIDANCE
            )
        self.default_usage_guidance = "\n\n".join(
            part
            for part in (
                type(self).default_usage_guidance,
                self._VISION_GUIDANCE if supports_vision else None,
            )
            if part
        )

    def __call__(
        self,
        path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        *,
        workspace: Hidden[AgentWorkspace] = None,
        handle: Hidden[ToolLibraryHandle] = None,
    ) -> str:
        first, count, is_image = self._read_options(path, offset, limit)
        workspace = resolve_workspace(workspace)
        path = _tool_path(path, workspace)
        if is_image:
            data = workspace.read_prefix(path, max_bytes=self.max_image_bytes + 1)
        else:
            data = self._read_text(workspace, path, first, count)
        return self._result(path, data, handle)

    async def acall(
        self,
        path: str,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
        *,
        workspace: Hidden[AgentWorkspace] = None,
        handle: Hidden[ToolLibraryHandle] = None,
    ) -> str:
        first, count, is_image = self._read_options(path, offset, limit)
        workspace = resolve_workspace(workspace)
        path = _tool_path(path, workspace)
        if is_image:
            data = await workspace.aread_prefix(
                path, max_bytes=self.max_image_bytes + 1
            )
        else:
            data = await asyncio.to_thread(
                self._read_text, workspace, path, first, count
            )
        return self._result(path, data, handle)

    def _text_budget_error(self, error: _ReadByteLimitError):
        if error.single_line:
            return ValueError(
                f"A single line exceeds the byte limit ({self.max_text_bytes} bytes); "
                "read cannot retrieve it. Use another available tool, such as Bash. "
                "No content was returned."
            )
        return ValueError(
            f"Read exceeds the byte limit ({self.max_text_bytes} bytes). "
            "Use offset and a smaller limit to paginate. No content was returned."
        )

    def _read_text(self, workspace, path, first, count):
        try:
            data = _read_offloaded_tool_result(
                path,
                workspace=workspace,
                offset=first,
                limit=count,
                max_bytes=self.max_text_bytes,
            )
            if data is None:
                data = workspace.read_lines(
                    path, offset=first, limit=count, max_bytes=self.max_text_bytes
                )
            return data
        except _ReadByteLimitError as error:
            raise self._text_budget_error(error) from error

    def _read_options(self, path, offset, limit):
        if any(
            value is not None and (type(value) is not int or value <= 0)
            for value in (offset, limit)
        ):
            raise ValueError("offset and limit must be positive integers")
        is_image = get_mime_type(path).startswith("image/")
        if is_image and not self.supports_vision:
            raise ValueError("Image reading is disabled for this tool")
        if is_image and (offset is not None or limit is not None):
            raise ValueError("offset and limit are only supported for text files")
        return offset or 1, min(limit or 2000, 2000), is_image

    def _result(self, path: str, data: bytes, handle: ToolLibraryHandle | None) -> str:
        mime_type = get_mime_type(path)
        ceiling = (
            self.max_image_bytes
            if mime_type.startswith("image/")
            else self.max_text_bytes
        )
        if len(data) > ceiling:
            if not mime_type.startswith("image/"):
                raise self._text_budget_error(_read_byte_limit_error(data, ceiling))
            raise ValueError("File exceeds the configured read byte limit")
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
    runtime_inputs=[
        "workspace",
        ContextBinding(source="shell_capture", required=False),
    ],
    required_permissions=["process.execute"],
    retry=False,
)
class BashTool:
    """Run Bash using the host-configured process executor.

    Use the injected workspace working directory. Execution is limited to 30 seconds
    and, by default, 1,000,000 combined stdout/stderr bytes. The host may override
    the capture budget. Requires a configured executor
    with Bash and the required execution permission. Isolation and filesystem
    access depend on the configured executor and its declared requirements.

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

    def __init__(self):
        self.tool_config = deepcopy(self.tool_config)

    def __call__(
        self,
        command: Union[str, list[str]],
        timeout_ms: Optional[int] = None,
        *,
        workspace: Hidden[AgentWorkspace] = None,
        shell_capture: Hidden[object] = None,
    ) -> ShellResult:
        from msgflux.nn.functional import wait_for  # noqa: PLC0415

        return wait_for(
            self.acall,
            command=command,
            timeout_ms=timeout_ms,
            workspace=workspace,
            shell_capture=shell_capture,
        )

    async def acall(
        self,
        command: Union[str, list[str]],
        timeout_ms: Optional[int] = None,
        *,
        workspace: Hidden[AgentWorkspace] = None,
        shell_capture: Hidden[object] = None,
    ) -> ShellResult:
        workspace = resolve_workspace(workspace)
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
                cwd=workspace.resolve("."),
                timeout_seconds=timeout,
                max_output_bytes=remaining,
            )
            for item in commands
        ]
        if shell_capture is not None:
            return await shell_capture.run(workspace, requests)
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
                result = await workspace.arun(request)
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


@tool_config(runtime_inputs=["workspace"], retry=False)
class WriteTool(WorkspaceChangeTool):
    """Create or overwrite a UTF-8 file in the authorized workspace.

    Args:
        path: File path, absolute or relative to the configured workspace cwd.
        content: Complete new text to write.
    """

    name = "write"
    display_name = "Write"
    annotations = {"path": str, "content": str, "return": dict[str, str]}

    def prepare_workspace_change(self, arguments, workspace):
        workspace = resolve_workspace(workspace)
        return workspace.editor.prepare_write(
            _tool_path(arguments["path"], workspace), arguments["content"]
        )

    def __call__(
        self,
        path: str,
        content: str,
        *,
        workspace: Hidden[AgentWorkspace] = None,
    ) -> dict[str, str]:
        return self._apply(
            {"path": path, "content": content},
            resolve_workspace(workspace),
        )

    async def acall(
        self,
        path: str,
        content: str,
        *,
        workspace: Hidden[AgentWorkspace] = None,
    ) -> dict[str, str]:
        return await asyncio.to_thread(self, path, content, workspace=workspace)


@tool_config(runtime_inputs=["workspace"], retry=False)
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

    def prepare_workspace_change(self, arguments, workspace):
        workspace = resolve_workspace(workspace)
        return workspace.editor.prepare_edit(
            _tool_path(arguments["path"], workspace), arguments["old"], arguments["new"]
        )

    def __call__(
        self,
        path: str,
        old: str,
        new: str,
        *,
        workspace: Hidden[AgentWorkspace] = None,
    ) -> dict[str, str]:
        return self._apply(
            {"path": path, "old": old, "new": new},
            resolve_workspace(workspace),
        )

    async def acall(
        self,
        path: str,
        old: str,
        new: str,
        *,
        workspace: Hidden[AgentWorkspace] = None,
    ) -> dict[str, str]:
        return await asyncio.to_thread(self, path, old, new, workspace=workspace)


@tool_config(runtime_inputs=["workspace"], retry=False)
class DeleteTool(WorkspaceChangeTool):
    """Delete one UTF-8 file or empty directory using the workspace review policy.

    Does not delete non-empty directories, binary files or the workspace root.
    The removed text or empty-directory description is in the approval preview.

    Args:
        path: File path, absolute or relative to the configured workspace cwd.
    """

    name = "delete"
    display_name = "Delete"
    annotations = {"path": str, "return": dict[str, str]}

    def prepare_workspace_change(self, arguments, workspace):
        workspace = resolve_workspace(workspace)
        return workspace.editor.prepare_delete_target(
            _tool_path(arguments["path"], workspace)
        )

    def __call__(
        self,
        path: str,
        *,
        workspace: Hidden[AgentWorkspace] = None,
    ) -> dict[str, str]:
        return self._apply(
            {"path": path},
            resolve_workspace(workspace),
        )

    async def acall(
        self,
        path: str,
        *,
        workspace: Hidden[AgentWorkspace] = None,
    ) -> dict[str, str]:
        return await asyncio.to_thread(self, path, workspace=workspace)


__all__ = ["ReadFileTool", "BashTool", "WriteTool", "EditTool", "DeleteTool"]
