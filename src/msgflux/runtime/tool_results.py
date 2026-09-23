"""Immutable, location-independent tool results with incremental local storage."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator
from uuid import uuid4

import msgspec

from msgflux._private.tool_result_reference import (
    ToolResultRef,
)
from msgflux._private.tool_result_reference import (
    validate_result_id as _validate_id,
)
from msgflux.runtime.workspace_local import _check_posix, _root_directory


def _positive(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def get_tool_result_reference(result: object) -> ToolResultRef | None:
    """Extract a typed reference from an offload envelope or Shell result.

    Also accepts JSON-decoded Shell results and native history items. Never
    parses text, accesses storage or grants permission to read a result.
    Malformed structured references raise rather than silently disappear.
    """
    if isinstance(result, Mapping):
        if result.get("type") == "tool_result_reference":
            reference = result.get("reference")
            if reference is None:
                raise ValueError("Offloaded tool result is missing its reference")
        elif "output_reference" in result:
            reference = result.get("output_reference")
        elif (
            result.get("type") in ("shell_call_output", "function_call_output")
            or result.get("role") == "tool"
        ):
            metadata = result.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise ValueError("Tool result metadata must be a mapping")
            reference = metadata.get("tool_result_reference")
        else:
            return None
    elif isinstance(result, msgspec.Struct):
        reference = getattr(result, "output_reference", None)
    else:
        return None
    if reference is None or isinstance(reference, ToolResultRef):
        return reference
    return msgspec.convert(reference, type=ToolResultRef, strict=True)


class ToolResultIntegrityError(ValueError):
    """Stored content or metadata does not match the durable reference."""


class ToolResultTooLargeError(ValueError):
    """A write exceeded its configured per-result storage budget."""


class ToolResultQuotaError(ToolResultTooLargeError):
    """The aggregate content/metadata budget would be exceeded."""


class ToolResultUsage(msgspec.Struct, frozen=True):
    size_bytes: int
    results: int
    pending: int


class ToolResultStore(ABC):
    """Host-owned result storage; references are identities, not access grants."""

    @abstractmethod
    def put(
        self, chunks: Iterable[bytes], *, media_type: str = "application/octet-stream"
    ) -> ToolResultRef:
        """Publish only after the complete chunk stream has been stored."""

    @abstractmethod
    def get(self, result_id: str) -> ToolResultRef:
        """Load the descriptor of a published result, or raise if missing."""

    @abstractmethod
    def iter_bytes(
        self,
        reference: ToolResultRef,
        *,
        offset: int = 0,
        limit: int | None = None,
        chunk_size: int = 65536,
    ) -> Iterator[bytes]:
        """Read a byte range incrementally; consumers must close abandoned iterators."""

    def read(
        self, reference: ToolResultRef, *, offset: int = 0, limit: int = 65536
    ) -> bytes:
        """Materialize at most ``limit`` bytes; use iter_bytes for full transmission."""
        _positive(limit, "limit")
        return b"".join(self.iter_bytes(reference, offset=offset, limit=limit))

    def verify(self, reference: ToolResultRef) -> None:
        """Check all bytes incrementally, without retaining the complete output."""
        digest = hashlib.sha256()
        size = 0
        for chunk in self.iter_bytes(reference):
            digest.update(chunk)
            size += len(chunk)
        if size != reference.size_bytes or digest.hexdigest() != reference.sha256:
            raise ToolResultIntegrityError(
                "Tool result content does not match reference"
            )


class LocalToolResultStore(ToolResultStore):
    """POSIX store in a trusted host directory, not a model filesystem sandbox.

    Results are immutable through this API. Interrupted writes may leave private
    staging directories; no automatic garbage collection deletes historical data.
    Quotas count file bytes, not filesystem block/inode overhead. Writers using
    this API coordinate through an advisory directory lock on local POSIX disks.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        max_result_bytes: int = 64 * 1024 * 1024,
        max_store_bytes: int = 1024 * 1024 * 1024,
    ) -> None:
        _check_posix()
        _positive(max_result_bytes, "max_result_bytes")
        _positive(max_store_bytes, "max_store_bytes")
        self.root = Path(root).expanduser().absolute()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.max_result_bytes = max_result_bytes
        self.max_store_bytes = max_store_bytes
        self._root_parts = self.root.parts[1:]
        with _root_directory(self._root_parts) as fd:
            st = os.fstat(fd)
            self._identity = (st.st_dev, st.st_ino)

    @contextmanager
    def _root(self):
        with _root_directory(self._root_parts) as fd:
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) != self._identity:
                raise PermissionError("Tool result store root was replaced")
            yield fd

    @staticmethod
    @contextmanager
    def _file(parent: int, name: str):
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise PermissionError(
                    "Tool result entries must be regular files without hard links"
                )
            with os.fdopen(fd, "rb", closefd=False) as stream:
                yield stream
        finally:
            os.close(fd)

    @staticmethod
    @contextmanager
    def _result(root: int, result_id: str):
        _validate_id(result_id)
        fd = os.open(
            result_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root
        )
        try:
            yield fd
        finally:
            os.close(fd)

    @classmethod
    def _metadata(cls, directory: int, result_id: str) -> ToolResultRef:
        with cls._file(directory, "metadata.json") as stream:
            data = stream.read(4097)
        if len(data) > 4096:
            raise ToolResultIntegrityError("Tool result metadata is too large")
        try:
            ref = msgspec.json.decode(data, type=ToolResultRef)
        except (msgspec.DecodeError, ValueError) as exc:
            raise ToolResultIntegrityError("Invalid tool result metadata") from exc
        if ref.result_id != result_id:
            raise ToolResultIntegrityError("Tool result metadata ID mismatch")
        return ref

    def get(self, result_id: str) -> ToolResultRef:
        _validate_id(result_id)
        with self._root() as root, self._result(root, result_id) as directory:
            ref = self._metadata(directory, result_id)
            with self._file(directory, "content") as stream:
                if os.fstat(stream.fileno()).st_size != ref.size_bytes:
                    raise ToolResultIntegrityError("Tool result size mismatch")
            return ref

    def put(
        self, chunks: Iterable[bytes], *, media_type: str = "application/octet-stream"
    ) -> ToolResultRef:
        result_id = f"res_{uuid4().hex}"
        # Validate metadata before starting or consuming a caller's generator.
        ToolResultRef(result_id, 0, hashlib.sha256().hexdigest(), media_type)
        staging = f".pending-{uuid4().hex}"
        with self._locked_root() as root:
            available = self.max_store_bytes - self._usage(root).size_bytes
            bounded = self._quota_chunks(chunks, result_id, media_type, available)
            os.mkdir(staging, mode=0o700, dir_fd=root)
            directory = os.open(
                staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root
            )
            published = False
            try:
                ref = self._write(directory, bounded, result_id, media_type)
                os.fsync(directory)
                # A published result is a nonempty directory: rename cannot
                # overwrite it, even if an ID collision is forced.
                os.rename(staging, result_id, src_dir_fd=root, dst_dir_fd=root)
                published = True
                os.fsync(root)
                return ref
            except BaseException as error:
                if not published:
                    self._discard(root, staging, directory, error)
                raise
            finally:
                os.close(directory)

    @contextmanager
    def _locked_root(self):
        import fcntl  # noqa: PLC0415

        with self._root() as root:
            fcntl.flock(root, fcntl.LOCK_EX)
            try:
                yield root
            finally:
                fcntl.flock(root, fcntl.LOCK_UN)

    @staticmethod
    def _directories(root):
        # scandir(fd) duplicates the descriptor but can share its directory
        # offset. Use a fresh open file description for every inventory pass.
        scan_root = os.open(".", os.O_RDONLY | os.O_DIRECTORY, dir_fd=root)
        try:
            with os.scandir(scan_root) as entries:
                for count, entry in enumerate(entries, 1):
                    if count > 100_000:
                        raise ToolResultQuotaError(
                            "Store inventory exceeds 100000 entries"
                        )
                    if not re.fullmatch(r"(?:res_|\.pending-)[0-9a-f]{32}", entry.name):
                        raise ToolResultIntegrityError("Unexpected result store entry")
                    if not entry.is_dir(follow_symlinks=False):
                        raise ToolResultIntegrityError(
                            "Store entry is not a safe directory"
                        )
                    yield entry.name
        finally:
            os.close(scan_root)

    def _usage(self, root):
        size = results = pending = 0
        for name in self._directories(root):
            directory = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root
            )
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if entry.name not in {"content", "metadata.json"}:
                            raise ToolResultIntegrityError("Unexpected result file")
                        with self._file(directory, entry.name) as stream:
                            size += os.fstat(stream.fileno()).st_size
            finally:
                os.close(directory)
            results += name.startswith("res_")
            pending += name.startswith(".pending-")
        return ToolResultUsage(size, results, pending)

    def usage(self) -> ToolResultUsage:
        """Count published and interrupted staging bytes under the writer lock."""
        with self._locked_root() as root:
            return self._usage(root)

    @staticmethod
    def _quota_chunks(chunks, result_id, media_type, available):
        size = 0

        def check():
            metadata = ToolResultRef(result_id, size, "0" * 64, media_type)
            if size + len(msgspec.json.encode(metadata)) > available:
                raise ToolResultQuotaError("Tool result store exceeds max_store_bytes")

        check()
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise TypeError("Tool result chunks must be bytes")
            size += len(chunk)
            check()
            yield chunk

    def collect_garbage(  # noqa: C901
        self,
        retained: Iterable[ToolResultRef],
        *,
        quiescent: bool = False,
        dry_run: bool = True,
    ) -> tuple[str, ...]:
        """Offline maintenance with the host's COMPLETE reference inventory.

        Stop all checkpoint writers and readers first, including other processes.
        Inventory must include every thread, fork, historical event and external
        consumer sharing this store. No liveness is inferred from file age.
        Default is preview only; ordinary writes never evict results.
        """
        if type(dry_run) is not bool or type(quiescent) is not bool:
            raise TypeError("Maintenance flags must be booleans")
        if not quiescent:
            raise ValueError("A quiescent complete reference inventory is required")
        refs = {}
        for ref in retained:
            if not isinstance(ref, ToolResultRef):
                raise TypeError("Expected typed retained references")
            if ref.result_id in refs and refs[ref.result_id] != ref:
                raise ToolResultIntegrityError("Conflicting retained references")
            refs[ref.result_id] = ref
        with self._locked_root() as root:
            self._usage(root)  # Validate all entries before any deletion.
            for ref in refs.values():
                if self.get(ref.result_id) != ref:
                    raise ToolResultIntegrityError("Retained reference mismatch")
                self.verify(ref)
            candidates = tuple(sorted(set(self._directories(root)) - refs.keys()))
            if not dry_run:
                for name in candidates:
                    directory = os.open(
                        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root
                    )
                    try:
                        for filename in ("content", "metadata.json"):
                            try:
                                os.unlink(filename, dir_fd=directory)
                            except FileNotFoundError:
                                pass
                        os.rmdir(name, dir_fd=root)
                    finally:
                        os.close(directory)
                os.fsync(root)
            return candidates

    @staticmethod
    def _discard(root, staging, directory, error: BaseException) -> None:
        # A rename may have completed before an interruption was delivered.
        # Never unlink via the open descriptor if staging no longer owns it.
        try:
            staged = os.stat(staging, dir_fd=root, follow_symlinks=False)
            opened = os.fstat(directory)
            if (staged.st_dev, staged.st_ino) != (opened.st_dev, opened.st_ino):
                error.add_note("Staging changed; cleanup skipped")
                return
        except FileNotFoundError:
            return
        except OSError as cleanup_error:
            error.add_note(f"Unable to inspect staging for cleanup: {cleanup_error}")
            return
        for name in ("content", "metadata.json"):
            try:
                os.unlink(name, dir_fd=directory)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                error.add_note(f"Unable to clean staging file: {cleanup_error}")
        try:
            os.rmdir(staging, dir_fd=root)
        except OSError as cleanup_error:
            error.add_note(f"Unable to remove staging directory: {cleanup_error}")

    def _write(self, directory, chunks, result_id, media_type) -> ToolResultRef:
        digest = hashlib.sha256()
        size = 0
        fd = os.open(
            "content", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory
        )
        with os.fdopen(fd, "wb") as stream:
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise TypeError("Tool result chunks must be bytes")
                if len(chunk) > self.max_result_bytes - size:
                    raise ToolResultTooLargeError(
                        "Tool result exceeds max_result_bytes"
                    )
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        ref = ToolResultRef(result_id, size, digest.hexdigest(), media_type)
        fd = os.open(
            "metadata.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=directory,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(msgspec.json.encode(ref))
            stream.flush()
            os.fsync(stream.fileno())
        return ref

    def iter_bytes(
        self,
        reference: ToolResultRef,
        *,
        offset: int = 0,
        limit: int | None = None,
        chunk_size: int = 65536,
    ) -> Iterator[bytes]:
        if not isinstance(reference, ToolResultRef):
            raise TypeError("reference must be ToolResultRef")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if limit is not None:
            _positive(limit, "limit")
        _positive(chunk_size, "chunk_size")
        with self._root() as root, self._result(root, reference.result_id) as directory:
            if self._metadata(directory, reference.result_id) != reference:
                raise ToolResultIntegrityError(
                    "Tool result reference does not match metadata"
                )
            with self._file(directory, "content") as stream:
                if os.fstat(stream.fileno()).st_size != reference.size_bytes:
                    raise ToolResultIntegrityError("Tool result size mismatch")
                stream.seek(offset)
                remaining = max(0, reference.size_bytes - offset)
                if limit is not None:
                    remaining = min(remaining, limit)
                while remaining:
                    chunk = stream.read(min(chunk_size, remaining))
                    if not chunk:
                        raise ToolResultIntegrityError(
                            "Tool result was truncated during read"
                        )
                    remaining -= len(chunk)
                    yield chunk
