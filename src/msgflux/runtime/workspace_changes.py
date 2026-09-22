"""Provider-neutral text changes, review previews and authorized application."""

import asyncio
import hashlib
from collections.abc import Callable
from difflib import unified_diff
from typing import Literal

import msgspec

from msgflux.runtime.approvals.base import ApprovalStore
from msgflux.runtime.approvals.records import ApprovalBinding, ApprovalRecord
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.workspace import (
    WorkspaceConflictError,
    WorkspaceFilesystem,
    workspace_path,
)
from msgflux.runtime.workspace_contracts import WorkspaceIdentity, WriteGuarantee


class PreparedFileChange(
    msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True
):
    """Detached review, not authority. File None means absent; dirs have no text."""

    workspace_id: str
    path: str
    before: str | None
    after: str | None
    schema_version: int = 1
    workspace_identity: WorkspaceIdentity | None = None
    write_guarantee: WriteGuarantee = "atomic_compare"
    target_kind: Literal["file", "empty_directory"] = "file"
    directory_token: str | None = None

    def __post_init__(self):
        if self.workspace_identity is not None and not isinstance(
            self.workspace_identity, WorkspaceIdentity
        ):
            raise TypeError("Expected WorkspaceIdentity")
        if self.write_guarantee not in ("atomic_compare", "cooperative_compare"):
            raise ValueError("Unknown workspace write guarantee")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Unsupported prepared change version")
        if not isinstance(self.workspace_id, str) or not self.workspace_id:
            raise ValueError("Prepared changes require a workspace identity")
        if workspace_path(self.path) != self.path:
            raise ValueError("Prepared change paths must be canonical")
        if any(
            value is not None and not isinstance(value, str)
            for value in (self.before, self.after)
        ):
            raise TypeError("Prepared file contents must be text or None")
        self._validate_target()
        for value in (self.before, self.after):
            if value is not None:
                value.encode("utf-8")

    def _validate_target(self):
        if self.target_kind not in {"file", "empty_directory"}:
            raise ValueError("Unknown workspace change target kind")
        if self.target_kind == "empty_directory":
            if (
                self.path == "/"
                or self.before is not None
                or self.after is not None
                or not isinstance(self.directory_token, str)
                or not self.directory_token
            ):
                raise ValueError("Invalid empty-directory deletion proposal")
        elif self.directory_token is not None:
            raise ValueError("File proposals cannot contain a directory token")
        elif self.before == self.after:
            raise ValueError("Prepared change must modify the file")

    @property
    def operation(self) -> str:
        if self.target_kind == "empty_directory":
            return "delete"
        if self.before is None:
            return "create"
        return "delete" if self.after is None else "update"

    @property
    def digest(self) -> str:
        return hashlib.sha256(msgspec.json.encode(self)).hexdigest()

    @property
    def diff(self) -> str:
        """Unified review diff with explicit missing-newline markers."""
        if self.target_kind == "empty_directory":
            return f"Delete empty directory: {self.path}/\n"
        source = "/dev/null" if self.before is None else f"a{self.path}"
        target = "/dev/null" if self.after is None else f"b{self.path}"
        lines = unified_diff(
            (self.before or "").splitlines(keepends=True),
            (self.after or "").splitlines(keepends=True),
            fromfile=source,
            tofile=target,
        )
        return (
            "".join(
                line
                if line.endswith("\n")
                else line + "\n\\ No newline at end of file\n"
                for line in lines
            )
            or f"--- {source}\n+++ {target}\n"
        )


class WorkspaceEditor:
    """Shared host backend for write, exact edit and future patch frontends.

    Approval and atomic comparison are required by default. A trusted host may
    disable approval or explicitly select cooperative comparison; live workspace
    permissions, resource identity and the selected guarantee remain enforced.
    """

    def __init__(
        self,
        filesystem: WorkspaceFilesystem,
        *,
        require_approval: bool = True,
        write_guarantee: WriteGuarantee = "atomic_compare",
    ):
        if not isinstance(filesystem, WorkspaceFilesystem):
            raise TypeError("WorkspaceEditor requires a WorkspaceFilesystem")
        if type(require_approval) is not bool:
            raise TypeError("require_approval must be a boolean")
        self.filesystem = filesystem
        self.require_approval = require_approval
        if write_guarantee not in ("atomic_compare", "cooperative_compare"):
            raise ValueError("Unknown workspace write guarantee")
        self.write_guarantee = write_guarantee

    def _read(self, path):
        try:
            return self.filesystem.read_text(path)
        except FileNotFoundError:
            return None

    def _prepare(self, path, before, after):
        change = PreparedFileChange(
            workspace_id=self.filesystem.workspace_id,
            workspace_identity=self.filesystem.identity,
            write_guarantee=self.write_guarantee,
            path=path,
            before=before,
            after=after,
        )
        self._authorize(change)
        return change

    def prepare_write(self, path: str, content: str) -> PreparedFileChange:
        """Create or overwrite text; preserve the exact old bytes for review."""
        if not isinstance(content, str):
            raise TypeError("content must be text")
        path = workspace_path(path)
        return self._prepare(path, self._read(path), content)

    def prepare_edit(self, path: str, old: str, new: str) -> PreparedFileChange:
        """Replace one unambiguous exact match, with no whitespace guessing."""
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("old must be non-empty text and new must be text")
        path = workspace_path(path)
        before = self._read(path)
        if before is None:
            raise FileNotFoundError(path)
        start = before.find(old)
        if start < 0 or before.find(old, start + 1) >= 0:
            raise ValueError("old must match exactly once")
        return self._prepare(
            path, before, before[:start] + new + before[start + len(old) :]
        )

    def prepare_create(self, path: str, content: str) -> PreparedFileChange:
        """Prepare a creation that cannot overwrite an existing file."""
        path = workspace_path(path)
        if self._read(path) is not None:
            raise FileExistsError(path)
        return self._prepare(path, None, content)

    def prepare_transform(
        self, path: str, transform: Callable[[str], str]
    ) -> PreparedFileChange:
        """Apply a host-owned pure text transform to an existing file snapshot."""
        path = workspace_path(path)
        before = self._read(path)
        if before is None:
            raise FileNotFoundError(path)
        after = transform(before)
        if not isinstance(after, str):
            raise TypeError("Text transforms must return text")
        return self._prepare(path, before, after)

    def prepare_delete(self, path: str) -> PreparedFileChange:
        """Prepare deletion including the complete removed text for review."""
        path = workspace_path(path)
        before = self._read(path)
        if before is None:
            raise FileNotFoundError(path)
        return self._prepare(path, before, None)

    def prepare_delete_target(self, path: str) -> PreparedFileChange:
        """Prepare deletion of one UTF-8 file or one empty directory."""
        path = workspace_path(path)
        token = self.filesystem.deletion_directory_token(path)
        if token is None:
            return self.prepare_delete(path)
        change = PreparedFileChange(
            workspace_id=self.filesystem.workspace_id,
            workspace_identity=self.filesystem.identity,
            write_guarantee=self.write_guarantee,
            path=path,
            before=None,
            after=None,
            target_kind="empty_directory",
            directory_token=token,
        )
        self._authorize(change)
        return change

    def _authorize(self, change):
        if not isinstance(change, PreparedFileChange):
            raise TypeError("Expected a PreparedFileChange")
        if change.workspace_id != self.filesystem.workspace_id:
            raise PermissionError("Prepared change belongs to another workspace")
        if change.workspace_identity != self.filesystem.identity:
            raise PermissionError(
                "Prepared change resource changed; prepare a new review"
            )
        if change.write_guarantee != self.write_guarantee:
            raise PermissionError("Prepared change write guarantee changed")
        self.filesystem._authorize(
            "list" if change.target_kind == "empty_directory" else "read", change.path
        )
        self.filesystem._authorize(
            "delete" if change.after is None else "write", change.path
        )
        self.filesystem.require_write_guarantee(self.write_guarantee)

    def approval_binding(
        self,
        change: PreparedFileChange,
        *,
        tool_call_id: str,
        tool_name: str,
        tool_revision: str,
        policy_version: str,
    ) -> ApprovalBinding:
        """Bind the review to exact contents, path and live execution identity."""
        self._authorize(change)
        scope = get_execution_scope()
        return ApprovalBinding.from_call(
            namespace=scope.namespace,
            thread_id=scope.thread_id,
            run_id=scope.run_id,
            principal=scope.principal,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_revision=tool_revision,
            policy_version=policy_version,
            arguments={"prepared_change": change.digest},
            resources={
                "workspace_id": change.workspace_id,
                "workspace_identity": msgspec.to_builtins(self.filesystem.identity),
                "write_guarantee": self.write_guarantee,
                "path": change.path,
                "operation": change.operation,
                "isolation": sorted(scope.environment.requirements.mechanisms),
            },
        )

    def apply(
        self,
        change: PreparedFileChange,
        *,
        approval: ApprovalRecord | None = None,
        approval_store: ApprovalStore | None = None,
    ) -> None:
        """Apply once; never retry an uncertain approval consumption automatically."""
        self._authorize(change)
        current = (
            self.filesystem.deletion_directory_token(change.path)
            if change.target_kind == "empty_directory"
            else self._read(change.path)
        )
        expected = (
            change.directory_token
            if change.target_kind == "empty_directory"
            else change.before
        )
        if current != expected:
            raise WorkspaceConflictError("Workspace target changed since preparation")
        if approval is not None:
            if not isinstance(approval, ApprovalRecord) or not isinstance(
                approval_store, ApprovalStore
            ):
                raise TypeError("Approved application requires a record and its store")
            previous = approval.binding
            binding = self.approval_binding(
                change,
                tool_call_id=previous.tool_call_id,
                tool_name=previous.tool_name,
                tool_revision=previous.tool_revision,
                policy_version=previous.policy_version,
            )
            approval_store.consume(approval.request_id, binding=binding)
        elif self.require_approval or approval_store is not None:
            raise PermissionError("A reviewed approval is required for this change")
        if change.target_kind == "empty_directory":
            self.filesystem.checked_rmdir(
                change.path,
                expected=change.directory_token,
                guarantee=self.write_guarantee,
            )
            return
        self.filesystem.checked_replace(
            change.path,
            expected=None if change.before is None else change.before.encode("utf-8"),
            replacement=None if change.after is None else change.after.encode("utf-8"),
            guarantee=self.write_guarantee,
        )

    async def aprepare_write(self, path: str, content: str) -> PreparedFileChange:
        return await asyncio.to_thread(self.prepare_write, path, content)

    async def aprepare_edit(self, path: str, old: str, new: str) -> PreparedFileChange:
        return await asyncio.to_thread(self.prepare_edit, path, old, new)

    async def aprepare_delete(self, path: str) -> PreparedFileChange:
        return await asyncio.to_thread(self.prepare_delete, path)

    async def aprepare_delete_target(self, path: str) -> PreparedFileChange:
        return await asyncio.to_thread(self.prepare_delete_target, path)

    async def aprepare_create(self, path: str, content: str) -> PreparedFileChange:
        return await asyncio.to_thread(self.prepare_create, path, content)

    async def aprepare_transform(
        self, path: str, transform: Callable[[str], str]
    ) -> PreparedFileChange:
        return await asyncio.to_thread(self.prepare_transform, path, transform)

    async def aapply(self, change: PreparedFileChange, **kwargs) -> None:
        await asyncio.to_thread(self.apply, change, **kwargs)
