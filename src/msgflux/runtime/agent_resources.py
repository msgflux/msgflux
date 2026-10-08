"""Owned SQLite resources bound to one Agent thread."""

from __future__ import annotations

import errno
import os
import re
import stat
from pathlib import Path
from threading import RLock
from typing import Any

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore
from msgflux.runtime.approvals import SQLiteApprovalStore
from msgflux.runtime.service.store import SQLiteServiceStore
from msgflux.runtime.tool_results import LocalToolResultStore, ToolOutputOffloadConfig
from msgflux.tasks import SQLiteTaskStore

_THREAD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


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

        _secure_directory(self.agent_dir)
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
                    error.add_note(f"Resource cleanup also failed: {cleanup_error!r}")
            raise

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
    ) -> None:
        self.thread_id = thread_id
        self.namespace = namespace
        self.thread_dir = thread_dir
        self.checkpoint_store = checkpoint_store
        self.task_store = task_store
        self.inbox_store = inbox_store
        self.agent_inbox = agent_inbox
        self.approval_store = approval_store
        self._closed = False
        self.offload_config = None
        self._tool_results = None
        self._tool_results_lock = RLock()

    def tool_result_store(self, *, create=True, config=None):
        """Resolve this thread's artifact store without creating it on observation."""
        with self._tool_results_lock:
            if self._closed:
                raise RuntimeError("Agent resources are closed")
            root = self.thread_dir / "tool-results"
            if self._tool_results is None:
                if not create and not root.exists():
                    return None
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
        """Close owned SQLite connections once; never delete persisted state."""
        if self._closed:
            return
        self._closed = True
        errors = []
        for resource in (
            self.approval_store,
            self.inbox_store,
            self.task_store,
            self.checkpoint_store,
        ):
            try:
                resource.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise ExceptionGroup("Failed to close Agent resource stores", errors)


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
