"""Shared formatting-preserving, atomic edits to service TOML configuration."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

import msgspec

from msgflux.runtime.service.local.files import _fsync_directory, process_lock
from msgflux.runtime.service.local.manifest import _ServiceConfig


def toml_parser():
    try:
        import tomlkit  # noqa: PLC0415
    except ImportError as error:
        raise ImportError(
            "Editing service configuration requires msgflux[service]"
        ) from error
    return tomlkit


def edit_service_config(path: Path, mutate, *, validate=None) -> tuple[Path, bool]:
    """Lock, edit and publish one document; validators may inspect a temporary file.

    ``mutate`` receives the parsed TOML document. ``validate`` optionally receives
    the candidate's path before publication, including for unchanged documents.
    Neither callback is allowed to publish the target itself.
    """
    target = Path(path).expanduser().resolve(strict=True)
    parser = toml_parser()
    with process_lock(target.with_name(f".{target.name}.lock"), blocking=True):
        original, metadata = _snapshot(target)
        try:
            document = parser.parse(original.decode("utf-8"))
        except (UnicodeDecodeError, parser.exceptions.ParseError) as error:
            raise ValueError(f"Invalid service config {target}: {error}") from error
        _validate_document(document, target)
        mutate(document)
        _validate_document(document, target)
        candidate = parser.dumps(document).encode("utf-8")
        changed = candidate != original
        _publish(target, original, metadata, candidate, validate, changed)
    return target, changed


def _validate_document(document, target):
    try:
        msgspec.convert(document.unwrap(), type=_ServiceConfig, strict=True)
    except msgspec.ValidationError as error:
        raise ValueError(f"Invalid service config {target}: {error}") from error


def _snapshot(path):
    with path.open("rb") as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Service config must be a regular file")
        content = stream.read()
    return content, metadata


def _require_unchanged(path, original, metadata):
    current, current_metadata = _snapshot(path)
    published = path.lstat()
    expected = metadata.st_dev, metadata.st_ino, stat.S_IMODE(metadata.st_mode)
    if (
        current != original
        or (
            current_metadata.st_dev,
            current_metadata.st_ino,
            stat.S_IMODE(current_metadata.st_mode),
        )
        != expected
        or (published.st_dev, published.st_ino, stat.S_IMODE(published.st_mode))
        != expected
    ):
        raise RuntimeError("Service config changed while editing; retry the command")


def _publish(path, original, metadata, candidate, validate, changed):
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".toml", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), stat.S_IMODE(metadata.st_mode))
            output.write(candidate)
            output.flush()
            os.fsync(output.fileno())
        if validate is not None:
            validate(temporary)
        _require_unchanged(path, original, metadata)
        if changed:
            os.replace(temporary, path)
            _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
