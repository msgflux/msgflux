"""Owned SQLite resources bound to one Agent thread."""

from __future__ import annotations

import errno
import json
import os
import re
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Iterator
from uuid import uuid4

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore
from msgflux.runtime.approvals import SQLiteApprovalStore
from msgflux.runtime.service.store import SQLiteServiceStore
from msgflux.runtime.tool_results import LocalToolResultStore, ToolOutputOffloadConfig
from msgflux.tasks import SQLiteTaskStore

_THREAD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")

_BACKEND_LAYOUT_FORMAT = "msgflux-agent-backend"
_BACKEND_LAYOUT_VERSION = 1
_BACKEND_LAYOUT_KIND = "shared-sqlite"
_BACKEND_LAYOUT_NAME = "layout.json"
_BACKEND_LOCK_NAME = "backend-layout.lock"
_LEGACY_THREAD_DATABASES = (
    "checkpoints.sqlite3",
    "tasks.sqlite3",
    "inbox.sqlite3",
    "approvals.sqlite3",
    "checkpoint.sqlite",
)


class AgentResources:
    """Open an owned set of persistent stores for each thread on demand.

    The root object has no filesystem side effects. Call :meth:`bind` from the
    host's existing per-thread Agent factory to open separate SQLite adapters
    under ``agent_dir/threads/<thread_id>/``. The returned bundle owns those
    adapters; model, workspace and Agent objects remain outside its lifecycle.
    """

    def __init__(self, agent_dir: str | os.PathLike[str]) -> None:
        self.agent_dir = Path(agent_dir).expanduser().absolute()
        if self.agent_dir == Path(self.agent_dir.anchor):
            raise ValueError("agent_dir must not be a filesystem root")

    def bind(
        self,
        thread_id: str,
        *,
        namespace: str,
        verbose: bool = False,
    ) -> BoundAgentResources:
        """Open the persistent stores for one stable thread identity."""
        if not isinstance(thread_id, str) or not _THREAD_ID.fullmatch(thread_id):
            raise ValueError("thread_id must be a safe single directory component")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if type(verbose) is not bool:
            raise TypeError("verbose must be a boolean")

        with _agent_layout_lock(self.agent_dir):
            _reject_shared_backend_layout(self.agent_dir)
            threads_dir = self.agent_dir / "threads"
            _secure_directory(threads_dir)
            thread_dir = threads_dir / thread_id
            _secure_directory(thread_dir)

            opened: list[Any] = []
            try:
                checkpoint_path = thread_dir / "checkpoints.sqlite3"
                _prepare_database_file(checkpoint_path)
                checkpoints = SQLiteCheckpointStore(path=str(checkpoint_path))
                opened.append(checkpoints)

                task_path = thread_dir / "tasks.sqlite3"
                _prepare_database_file(task_path)
                tasks = SQLiteTaskStore(path=str(task_path))
                opened.append(tasks)

                inbox_path = thread_dir / "inbox.sqlite3"
                _prepare_database_file(inbox_path)
                inbox_store = SQLiteAgentInboxStore(path=str(inbox_path))
                opened.append(inbox_store)
                inbox = AgentInbox(
                    store=inbox_store,
                    namespace=namespace,
                    thread_id=thread_id,
                    owner=namespace,
                    verbose=verbose,
                )

                approval_path = thread_dir / "approvals.sqlite3"
                _prepare_database_file(approval_path)
                approvals = SQLiteApprovalStore(path=str(approval_path))
                opened.append(approvals)
                return BoundAgentResources(
                    thread_id=thread_id,
                    namespace=namespace,
                    thread_dir=thread_dir,
                    checkpoint_store=checkpoints,
                    task_store=tasks,
                    inbox_store=inbox_store,
                    agent_inbox=inbox,
                    approval_store=approvals,
                )
            except BaseException as error:
                for resource in reversed(opened):
                    try:
                        resource.close()
                    except BaseException as cleanup_error:
                        error.add_note(
                            f"Resource cleanup also failed: {cleanup_error!r}"
                        )
                raise

    def has_thread(self, thread_id: str) -> bool:
        """Report an existing legacy per-thread checkpoint without binding."""
        if not isinstance(thread_id, str) or not _THREAD_ID.fullmatch(thread_id):
            return False
        backend_dir, _marker = _backend_layout_paths(self.agent_dir)
        try:
            backend_metadata = backend_dir.lstat()
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(backend_metadata.st_mode) or not stat.S_ISDIR(
                backend_metadata.st_mode
            ):
                raise ValueError("Agent backend layout path must be a real directory")
            raise ValueError(
                "agent_dir already uses the shared SQLite Agent backend layout; "
                "construct SQLiteAgentBackend instead of AgentResources"
            )
        path = self.agent_dir / "threads" / thread_id / "checkpoints.sqlite3"
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Agent resource database must be a regular file")
        return True

    def service_store(self) -> SQLiteServiceStore:
        """Open the host-owned service journal with private file permissions."""
        runtime_dir = self.agent_dir / "runtime"
        _secure_directory(self.agent_dir)
        _secure_directory(runtime_dir)
        path = runtime_dir / "service.sqlite3"
        _prepare_database_file(path)
        try:
            return SQLiteServiceStore(path=path)
        except BaseException:
            # Keep any created file for safe diagnosis/retry, but ensure it
            # remains private even when initialization itself fails.
            _secure_database_file(path)
            raise


class BoundAgentResources:
    """Per-thread store handles owned by the host's Agent factory."""

    def __init__(
        self,
        *,
        thread_id: str,
        namespace: str,
        thread_dir: Path,
        checkpoint_store: SQLiteCheckpointStore,
        task_store: SQLiteTaskStore,
        inbox_store: SQLiteAgentInboxStore,
        agent_inbox: AgentInbox,
        approval_store: SQLiteApprovalStore,
        backend: object | None = None,
        _close_callback=None,
    ) -> None:
        self.thread_id = thread_id
        self.namespace = namespace
        self.thread_dir = thread_dir
        self.checkpoint_store = checkpoint_store
        self.task_store = task_store
        self.inbox_store = inbox_store
        self.agent_inbox = agent_inbox
        self.approval_store = approval_store
        self.backend = backend
        self._close_callback = _close_callback
        self._closed = False
        self._closing = False
        self._closed_resources: set[int] = set()
        self.offload_config = None
        self._tool_results = None
        self._tool_results_lock = RLock()
        self._close_lock = RLock()

    def tool_result_store(self, *, create=True, config=None):
        """Resolve this thread's artifact store without creating it on observation."""
        with self._tool_results_lock:
            if self._closed or self._closing:
                raise RuntimeError("Agent resources are closed")
            root = self.thread_dir / "tool-results"
            if self._tool_results is None:
                if not create and not root.exists():
                    return None
                _secure_directory(self.thread_dir.parent)
                _secure_directory(self.thread_dir)
                limits = config or self.offload_config or ToolOutputOffloadConfig()
                self._tool_results = LocalToolResultStore(
                    root,
                    max_result_bytes=limits.max_result_bytes,
                    max_store_bytes=limits.max_store_bytes,
                )
            elif config is not None and (
                self._tool_results.max_result_bytes != config.max_result_bytes
                or self._tool_results.max_store_bytes != config.max_store_bytes
            ):
                raise ValueError(
                    "Nested Agent cannot replace tool output storage limits"
                )
            return self._tool_results

    def close(self) -> None:
        """Close owned SQLite connections once; retry only handles that failed."""
        with self._close_lock, self._tool_results_lock:
            if self._closed:
                return
            self._closing = True
            if self._close_callback is not None:
                try:
                    self._close_callback()
                except Exception as error:
                    raise ExceptionGroup(
                        "Failed to release Agent backend binding", [error]
                    ) from error
                self._closed = True
                return
            errors = []
            for resource in (
                self.approval_store,
                self.inbox_store,
                self.task_store,
                self.checkpoint_store,
            ):
                identity = id(resource)
                if identity in self._closed_resources:
                    continue
                try:
                    resource.close()
                except Exception as error:
                    errors.append(error)
                else:
                    self._closed_resources.add(identity)
            if errors:
                raise ExceptionGroup("Failed to close Agent resource stores", errors)
            self._closed = True


def _secure_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Agent resource path must be a real directory")
    if metadata.st_uid != os.getuid():
        raise PermissionError("Agent resource directories must be user-owned")
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        path.chmod(0o700)


def _secure_database_file(path: Path) -> None:
    _prepare_database_file(path)


def _validate_database_path(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Agent resource database must be a regular file")
    if metadata.st_uid != os.getuid():
        raise PermissionError("Agent resource databases must be user-owned")


def _prepare_database_file(path: Path) -> None:
    """Create/open a database path without following links, at mode 0600."""
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(
                "Agent resource database must be a regular file"
            ) from error
        raise
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Agent resource database must be a regular file")
        if metadata.st_uid != os.getuid():
            raise PermissionError("Agent resource databases must be user-owned")
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _backend_layout_paths(agent_dir: Path) -> tuple[Path, Path]:
    runtime_dir = agent_dir / "runtime"
    backend_dir = runtime_dir / "backend"
    return backend_dir, backend_dir / _BACKEND_LAYOUT_NAME


@contextmanager
def _agent_layout_lock(agent_dir: Path, *, create: bool = True) -> Iterator[None]:
    """Serialize legacy and shared layout initialization for one agent root."""
    runtime_dir = agent_dir / "runtime"
    if create:
        _secure_directory(agent_dir)
        _secure_directory(runtime_dir)
    # Imported lazily so normal in-process Agent construction does not load the
    # local service discovery modules until a managed root is bound.
    from msgflux.runtime.service.local.files import (  # noqa: PLC0415
        process_lock,
    )

    with ExitStack() as stack:
        try:
            handle = stack.enter_context(
                process_lock(
                    runtime_dir / _BACKEND_LOCK_NAME, blocking=True, create=create
                )
            )
        except FileNotFoundError:
            if create:
                raise
            # A never-bound root has no lock. Observation must not initialize it.
        else:
            if handle is None:
                raise RuntimeError(
                    "Could not acquire the Agent layout initialization lock"
                )
        yield


def _reject_shared_backend_layout(agent_dir: Path) -> None:
    """Refuse the legacy per-thread layout when a shared backend root exists."""
    backend_dir, _marker = _backend_layout_paths(agent_dir)
    try:
        metadata = backend_dir.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Agent backend layout path must be a real directory")
    raise ValueError(
        "agent_dir already uses the shared SQLite Agent backend layout; "
        "construct SQLiteAgentBackend instead of AgentResources"
    )


def _shared_backend_layout_exists(agent_dir: Path) -> bool:
    """Validate an existing shared layout marker without creating any path."""
    backend_dir, marker_path = _backend_layout_paths(agent_dir)
    try:
        metadata = backend_dir.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Agent backend layout path must be a real directory")
    if metadata.st_uid != os.getuid():
        raise PermissionError("Agent backend directories must be user-owned")
    _read_backend_layout_marker(marker_path)
    return True


def _prepare_shared_backend_layout(agent_dir: Path) -> None:
    """Create or validate the shared layout while the caller holds the lock."""
    _secure_directory(agent_dir)
    backend_dir, marker_path = _backend_layout_paths(agent_dir)
    try:
        metadata = backend_dir.lstat()
    except FileNotFoundError:
        metadata = None
    if metadata is not None:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Agent backend layout path must be a real directory")
        _secure_directory(backend_dir)
        _read_backend_layout_marker(marker_path)
        return

    if _legacy_thread_databases_present(agent_dir):
        raise ValueError(
            "agent_dir contains legacy per-thread Agent stores; the shared "
            "SQLite Agent backend does not migrate or adopt them"
        )
    _secure_directory(backend_dir)
    try:
        _write_backend_layout_marker(backend_dir, marker_path)
    except BaseException:
        # Only remove the directory when this call created an otherwise empty
        # root. A marker or database left by a partial initialization remains
        # for explicit diagnosis instead of being silently adopted.
        try:
            backend_dir.rmdir()
        except OSError:
            pass
        raise


def _read_backend_layout_marker(marker_path: Path) -> None:  # noqa: C901
    try:
        metadata = marker_path.lstat()
    except FileNotFoundError as error:
        raise ValueError(
            "Agent backend directory is missing its versioned layout marker"
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Agent backend layout marker must be a regular file")
    if metadata.st_uid != os.getuid():
        raise PermissionError("Agent backend layout marker must be user-owned")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(marker_path, flags)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(
                "Agent backend layout marker must be a regular file"
            ) from error
        raise
    try:
        opened_metadata = os.fstat(descriptor)
        if stat.S_ISLNK(opened_metadata.st_mode) or not stat.S_ISREG(
            opened_metadata.st_mode
        ):
            raise ValueError("Agent backend layout marker must be a regular file")
        if opened_metadata.st_uid != os.getuid():
            raise PermissionError("Agent backend layout marker must be user-owned")
        if stat.S_IMODE(opened_metadata.st_mode) != 0o600:
            os.fchmod(descriptor, 0o600)
        payload = b""
        while True:
            chunk = os.read(descriptor, 4096)
            if not chunk:
                break
            payload += chunk
    finally:
        os.close(descriptor)
    try:
        marker = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Agent backend layout marker is malformed") from error
    if (
        not isinstance(marker, dict)
        or set(marker) != {"format", "layout", "version"}
        or marker.get("format") != _BACKEND_LAYOUT_FORMAT
        or marker.get("layout") != _BACKEND_LAYOUT_KIND
        or type(marker.get("version")) is not int
        or marker.get("version") != _BACKEND_LAYOUT_VERSION
    ):
        raise ValueError("Agent backend layout marker is malformed or unsupported")


def _write_backend_layout_marker(backend_dir: Path, marker_path: Path) -> None:
    payload = json.dumps(
        {
            "format": _BACKEND_LAYOUT_FORMAT,
            "layout": _BACKEND_LAYOUT_KIND,
            "version": _BACKEND_LAYOUT_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    temporary = backend_dir / f".layout-{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            descriptor = -1
            output.write(payload)
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, marker_path)
        _fsync_directory(backend_dir)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _legacy_thread_databases_present(agent_dir: Path) -> bool:
    threads_dir = agent_dir / "threads"
    try:
        metadata = threads_dir.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("Agent thread root must be a real directory")
    found = False
    with os.scandir(threads_dir) as entries:
        for entry in entries:
            if entry.is_symlink():
                raise ValueError("Agent thread paths must not be symlinks")
            if not entry.is_dir(follow_symlinks=False):
                continue
            for name in _LEGACY_THREAD_DATABASES:
                candidate = Path(entry.path) / name
                try:
                    candidate_metadata = candidate.lstat()
                except FileNotFoundError:
                    continue
                if stat.S_ISLNK(candidate_metadata.st_mode) or not stat.S_ISREG(
                    candidate_metadata.st_mode
                ):
                    raise ValueError(
                        "Legacy Agent thread databases must be regular files"
                    )
                found = True
    return found


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
