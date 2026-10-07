"""Owner-only metadata files and POSIX process locks for local service discovery."""

from __future__ import annotations

import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

import msgspec

from msgflux.runtime.service.local.records import LocalServiceRecord

RECORD_NAME = "daemon.json"


def prepare_runtime_dir(path: Path) -> Path:
    """Create or secure the daemon directory and return its canonical path."""
    _require_posix()
    path = Path(path).expanduser()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Runtime directory must be a real directory, not a symlink")
    if metadata.st_uid != os.getuid():
        raise PermissionError("Runtime directory must be owned by the current user")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        path.chmod(0o700)
    return path.resolve(strict=True)


def read_record(runtime_dir: Path) -> LocalServiceRecord | None:
    path = Path(runtime_dir) / RECORD_NAME
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    _validate_private_file(path, metadata)
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as source:
            opened_metadata = os.fstat(source.fileno())
            _validate_private_file(path, opened_metadata)
            content = source.read()
        return msgspec.json.decode(content, type=LocalServiceRecord)
    except (msgspec.DecodeError, TypeError, ValueError) as exc:
        raise ValueError("Local service metadata is malformed") from exc


def write_record(runtime_dir: Path, record: LocalServiceRecord) -> None:
    directory = Path(runtime_dir)
    target = directory / RECORD_NAME
    try:
        _validate_private_file(target, target.lstat())
    except FileNotFoundError:
        pass
    payload = msgspec.json.encode(record)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".daemon-", dir=directory)
    temporary = Path(temporary_name)
    output = None
    try:
        output = os.fdopen(descriptor, "wb", closefd=True)
        with output:
            os.fchmod(output.fileno(), 0o600)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        _fsync_directory(directory)
    except BaseException:
        if output is None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def remove_record(runtime_dir: Path, instance_id: str) -> None:
    """Remove metadata only when it still belongs to the named daemon."""
    directory = Path(runtime_dir)
    record = read_record(directory)
    if record is None or record.instance_id != instance_id:
        return
    (directory / RECORD_NAME).unlink()
    _fsync_directory(directory)


@contextmanager
def process_lock(path: Path, *, blocking: bool = False) -> Iterator[BinaryIO | None]:
    """Lock a private file; yield None if a nonblocking lock is already held.

    The returned handle owns the flock until the context exits. Callers doing
    async coordination should poll this nonblocking operation between awaits.
    """
    _require_posix()
    import fcntl  # noqa: PLC0415

    lock_path = Path(path)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    handle = os.fdopen(descriptor, "r+b", closefd=True)
    try:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
            raise PermissionError("Lock file must be an owned regular file")
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            os.fchmod(handle.fileno(), 0o600)
        operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(handle.fileno(), operation)
        except BlockingIOError:
            yield None
            return
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def _validate_private_file(path: Path, metadata: os.stat_result) -> None:
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{path.name} must be a regular file, not a symlink")
    if metadata.st_uid != os.getuid():
        raise PermissionError(f"{path.name} must be owned by the current user")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PermissionError(f"{path.name} must have mode 0600")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_posix() -> None:
    if os.name != "posix":
        raise OSError("Local AgentService process management requires POSIX")
