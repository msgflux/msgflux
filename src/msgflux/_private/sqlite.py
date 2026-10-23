"""Private file creation for SQLite adapters using ordinary database paths."""

import os
import sqlite3
from pathlib import Path


def prepare_sqlite_file(path: str | os.PathLike[str]) -> None:
    """Create a missing database privately, preserving existing file permissions."""
    filename = os.fspath(path)
    if filename in ("", ":memory:"):
        return
    # SQLite accepts a trailing separator on a database filename. Strip it
    # without resolving symlinks or parent components before precreation.
    filename = os.fspath(Path(filename))
    try:
        try:
            descriptor = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        try:
            # Restore owner access if umask removed it, so SQLite can reopen
            # the file we just created. Existing files never reach this step.
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
        finally:
            os.close(descriptor)
    except OSError as error:
        # Keep the adapters' SQLite error contract and retain the OS cause.
        failure = sqlite3.OperationalError("unable to open database file")
        failure.sqlite_errorcode = sqlite3.SQLITE_CANTOPEN
        failure.sqlite_errorname = "SQLITE_CANTOPEN"
        raise failure from error
