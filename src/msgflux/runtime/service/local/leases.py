"""Process-lifetime owner evidence for local approval recovery."""

from __future__ import annotations

import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from msgflux.runtime.service.local.files import prepare_runtime_dir
from msgflux.runtime.service.records import ServiceRecoveryRequiredError


class LocalOwnerLease:
    """Hold an owner lock independently of the daemon's discovery directory.

    The journal records the exact file identity before this owner starts work.
    An old owner's unlocked, unchanged file proves only its service has stopped;
    callers still validate checkpoints and external work before recovery.
    """

    def __init__(self, service) -> None:
        if os.name != "posix":
            raise OSError("Local service owner leases require POSIX locks")
        path = service.store._path
        if path is None:
            raise ValueError("Local service owner leases require a persistent journal")
        self._store = service.store
        self._directory = prepare_runtime_dir(
            Path(path).parent / f".{Path(path).name}.owners"
        )
        self._descriptor: int | None = None
        self._path = self._lease_path(service._owner_id)
        try:
            descriptor = os.open(
                self._path,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError as error:
            raise ServiceRecoveryRequiredError(
                "The service owner already has a registered lease file"
            ) from error
        try:
            import fcntl  # noqa: PLC0415

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            metadata = os.fstat(descriptor)
            self._store.register_owner_lease(
                service._owner_id,
                path=str(self._path),
                device=metadata.st_dev,
                inode=metadata.st_ino,
            )
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor

    def _lease_path(self, owner_id: str) -> Path:
        if not isinstance(owner_id, str) or not re.fullmatch("[a-f0-9]{32}", owner_id):
            raise ServiceRecoveryRequiredError(
                "The previous service owner has no supported lease identity"
            )
        return self._directory / f"{owner_id}.lock"

    def close(self) -> None:
        """Release after the owning service has settled its workers and resources."""
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None

    @contextmanager
    def reclaim(self, owner_id: str) -> Iterator[None]:
        """Hold verified quiescence evidence until the owner transfer is committed."""
        if self._descriptor is None:
            raise ServiceRecoveryRequiredError(
                "The local service owner lease is closed"
            )
        evidence = self._store.owner_lease(owner_id)
        if evidence is None:
            raise ServiceRecoveryRequiredError(
                "The previous service owner has no registered lifetime lease"
            )
        path, device, inode = evidence
        expected = self._lease_path(owner_id)
        if Path(path) != expected:
            raise ServiceRecoveryRequiredError(
                "The previous service owner lease path does not match the journal"
            )
        descriptor = self._open_reclaim(expected, device, inode)
        try:
            yield
        finally:
            os.close(descriptor)

    def _open_reclaim(self, path: Path, device: int, inode: int) -> int:
        descriptor = None
        try:
            descriptor = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
            metadata = os.fstat(descriptor)
            self._validate_metadata(metadata, device, inode)

            import fcntl  # noqa: PLC0415

            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Replacement between open() and flock() must not manufacture proof.
            self._validate_metadata(path.lstat(), device, inode)
            return descriptor
        except OSError as error:
            if descriptor is not None:
                os.close(descriptor)
            raise ServiceRecoveryRequiredError(
                "Cannot establish previous service owner quiescence: "
                "its lifetime lease is unavailable or still held"
            ) from error
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            raise

    @staticmethod
    def _validate_metadata(metadata, device: int, inode: int) -> None:
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_dev != device
            or metadata.st_ino != inode
        ):
            raise ServiceRecoveryRequiredError(
                "The previous service owner lifetime lease has changed"
            )
