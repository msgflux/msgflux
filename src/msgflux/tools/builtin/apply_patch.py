"""Single-file V4A patch tool, independent of provider wire protocols."""

import asyncio
from typing import Literal, Optional

from msgflux.runtime.workspace import WorkspaceFilesystem
from msgflux.tools.builtin.workspace import _tool_path
from msgflux.tools.config import tool_config
from msgflux.tools.patch import apply_diff
from msgflux.tools.types import Hidden
from msgflux.tools.workspace_changes import WorkspaceChangeTool


@tool_config(tool_kind="apply_patch", runtime_inputs=["filesystem"], retry=False)
class ApplyPatchTool(WorkspaceChangeTool):
    """Create, update or delete one UTF-8 file using a V4A diff.

    Args:
        operation: Create a new file, update an existing file, or delete it.
        path: File path, absolute or relative to the configured workspace cwd.
        diff: V4A body: plus-prefixed lines for create; @@ context hunks for update.
            Omit or use null for delete. Do not include multi-file envelopes.
    """

    name = "apply_patch"
    display_name = "ApplyPatch"
    annotations = {
        "operation": Literal["create", "update", "delete"],
        "path": str,
        "diff": Optional[str],
        "return": dict[str, str],
    }

    def prepare_workspace_change(self, arguments, filesystem):
        operation, diff = arguments["operation"], arguments.get("diff")
        path = _tool_path(arguments["path"], self.cwd)
        editor = self._editor(filesystem)
        if operation == "delete":
            if diff is not None:
                raise ValueError("Delete operations must not contain a diff")
            return editor.prepare_delete(path)
        if not isinstance(diff, str):
            raise ValueError("Create and update require a text diff")
        if operation == "create":
            return editor.prepare_create(path, apply_diff("", diff, mode="create"))
        if operation == "update":
            return editor.prepare_transform(path, lambda text: apply_diff(text, diff))
        raise ValueError("Unknown patch operation")

    def __call__(
        self,
        operation: Literal["create", "update", "delete"],
        path: str,
        diff: Optional[str] = None,
        *,
        filesystem: Hidden[WorkspaceFilesystem],
    ) -> dict[str, str]:
        return self._apply(
            {"operation": operation, "path": path, "diff": diff}, filesystem
        )

    async def acall(
        self,
        operation: Literal["create", "update", "delete"],
        path: str,
        diff: Optional[str] = None,
        *,
        filesystem: Hidden[WorkspaceFilesystem],
    ) -> dict[str, str]:
        return await asyncio.to_thread(
            self, operation, path, diff, filesystem=filesystem
        )
