"""Provider-neutral text changes, review previews and authorized application."""

import asyncio
import errno
import hashlib
from collections.abc import Callable
from difflib import unified_diff
from typing import Literal

import msgspec

from msgflux.runtime.approvals.base import ApprovalStore
from msgflux.runtime.approvals.records import ApprovalBinding, ApprovalRecord
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.workspace.contracts import WorkspaceIdentity, WriteGuarantee
from msgflux.runtime.workspace.filesystem import (
    WorkspaceConflictError,
    WorkspaceFilesystem,
    workspace_path,
)


class TextEdit(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """One ordered exact replacement for an EditTool batch."""

    old: str
    new: str
    replace_all: bool = False

    def __post_init__(self):
        if not isinstance(self.old, str) or not self.old:
            raise ValueError("old must be non-empty text")
        if not isinstance(self.new, str):
            raise TypeError("new must be text")
        if type(self.replace_all) is not bool:
            raise TypeError("replace_all must be a boolean")


def _utf8_size(text: str) -> int:
    """Return UTF-8 byte length without materializing one large byte string."""
    if text.isascii():
        return len(text)
    return sum(
        len(text[offset : offset + 65536].encode("utf-8"))
        for offset in range(0, len(text), 65536)
    )


def _replace_all_text(text: str, old: str, new: str) -> str:
    return text.replace(old, new)


def _replace_once_text(text: str, start: int, old: str, new: str) -> str:
    return text[:start] + new + text[start + len(old) :]


def _replace_exact_bounded(
    text: str,
    text_bytes: int,
    old: str,
    new: str,
    *,
    replace_all: bool,
    max_bytes: int,
) -> tuple[str, int]:
    """Apply one exact edit only after proving its UTF-8 result fits."""
    if type(replace_all) is not bool:
        raise TypeError("replace_all must be a boolean")
    if not isinstance(old, str) or not old or not isinstance(new, str):
        raise ValueError("old must be non-empty text and new must be text")
    old_bytes = _utf8_size(old)
    new_bytes = _utf8_size(new)
    if old_bytes > max_bytes or new_bytes > max_bytes:
        raise ValueError("File exceeds max_edit_bytes")
    if replace_all:
        count = text.count(old)
        if count == 0:
            raise ValueError("old must match at least once")
        start = None
    else:
        start = text.find(old)
        if start < 0 or text.find(old, start + 1) >= 0:
            raise ValueError("old must match exactly once")
        count = 1
    result_bytes = text_bytes - count * old_bytes + count * new_bytes
    if result_bytes > max_bytes:
        raise ValueError("File exceeds max_edit_bytes")
    if replace_all:
        return _replace_all_text(text, old, new), result_bytes
    return _replace_once_text(text, start, old, new), result_bytes


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
        max_edit_bytes: int = 1_000_000,
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
        if type(max_edit_bytes) is not int or max_edit_bytes <= 0:
            raise ValueError("max_edit_bytes must be a positive integer")
        self.max_edit_bytes = max_edit_bytes

    def _check_text(self, text):
        if text is None:
            return
        if not isinstance(text, str):
            raise TypeError("File contents must be text")
        if len(text) > self.max_edit_bytes:
            raise ValueError("File exceeds max_edit_bytes")
        if _utf8_size(text) > self.max_edit_bytes:
            raise ValueError("File exceeds max_edit_bytes")

    def _read(self, path):
        try:
            data = self.filesystem.read_prefix(path, max_bytes=self.max_edit_bytes + 1)
            if len(data) > self.max_edit_bytes:
                raise ValueError("File exceeds max_edit_bytes")
            return data.decode("utf-8")
        except FileNotFoundError:
            return None

    def _prepare(self, path, before, after):
        self._check_text(before)
        self._check_text(after)
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
        self._check_text(content)
        path = workspace_path(path)
        return self._prepare(path, self._read(path), content)

    def prepare_edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> PreparedFileChange:
        """Replace exact matches, with no whitespace guessing.

        The default requires one unambiguous match. ``replace_all`` replaces all
        non-overlapping occurrences and requires at least one match.
        """
        if type(replace_all) is not bool:
            raise TypeError("replace_all must be a boolean")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ValueError("old must be non-empty text and new must be text")
        return self.prepare_edits(
            path, (TextEdit(old=old, new=new, replace_all=replace_all),)
        )

    def prepare_edits(self, path: str, edits) -> PreparedFileChange:
        """Prepare ordered exact edits from one snapshot with per-step bounds."""
        if edits is None:
            raise ValueError("edits must be a non-empty sequence")
        edits = tuple(edits)
        if not edits:
            raise ValueError("edits must be a non-empty sequence")
        if any(not isinstance(edit, TextEdit) for edit in edits):
            raise TypeError("edits must contain TextEdit values")
        for edit in edits:
            self._check_text(edit.old)
            self._check_text(edit.new)
        path = workspace_path(path)
        before = self._read(path)
        if before is None:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", path)
        after = before
        after_bytes = self._checked_bytes(before)
        for edit in edits:
            after, after_bytes = _replace_exact_bounded(
                after,
                after_bytes,
                edit.old,
                edit.new,
                replace_all=edit.replace_all,
                max_bytes=self.max_edit_bytes,
            )
        return self._prepare(path, before, after)

    def _checked_bytes(self, text):
        if not isinstance(text, str):
            raise TypeError("File contents must be text")
        size = _utf8_size(text)
        if size > self.max_edit_bytes:
            raise ValueError("File exceeds max_edit_bytes")
        return size

    def prepare_create(self, path: str, content: str) -> PreparedFileChange:
        """Prepare a creation that cannot overwrite an existing file."""
        path = workspace_path(path)
        if self._read(path) is not None:
            raise FileExistsError(f"File already exists: {path}")
        return self._prepare(path, None, content)

    def prepare_transform(
        self, path: str, transform: Callable[[str], str]
    ) -> PreparedFileChange:
        """Apply a host-owned pure text transform to an existing file snapshot."""
        path = workspace_path(path)
        before = self._read(path)
        if before is None:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", path)
        after = transform(before)
        if not isinstance(after, str):
            raise TypeError("Text transforms must return text")
        return self._prepare(path, before, after)

    def prepare_delete(self, path: str) -> PreparedFileChange:
        """Prepare deletion including the complete removed text for review."""
        path = workspace_path(path)
        before = self._read(path)
        if before is None:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory", path)
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
        self._check_text(change.before)
        self._check_text(change.after)
        if change.workspace_id != self.filesystem.workspace_id:
            raise PermissionError("Prepared change belongs to another workspace")
        if change.workspace_identity != self.filesystem.identity:
            raise PermissionError(
                "Prepared change resource changed; prepare a new review"
            )
        if change.write_guarantee != self.write_guarantee:
            raise PermissionError("Prepared change write guarantee changed")
        from msgflux.runtime.workspace.api import require_workspace_authority  # noqa: PLC0415, I001

        require_workspace_authority(filesystem=self.filesystem)
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
                "isolation": sorted(scope.workspace.requirements.mechanisms),
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

    async def aprepare_edit(
        self,
        path: str,
        old: str,
        new: str,
        *,
        replace_all: bool = False,
    ) -> PreparedFileChange:
        return await asyncio.to_thread(
            self.prepare_edit, path, old, new, replace_all=replace_all
        )

    async def aprepare_edits(self, path: str, edits) -> PreparedFileChange:
        return await asyncio.to_thread(self.prepare_edits, path, edits)

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
