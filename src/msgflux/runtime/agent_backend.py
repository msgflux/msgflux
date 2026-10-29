"""Host-owned shared SQLite persistence for many logical Agent bindings."""

from __future__ import annotations

import os
import stat
from threading import RLock
from typing import Any, Dict, List, Mapping

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.runtime.agent_inbox import AgentInbox, SQLiteAgentInboxStore
from msgflux.runtime.agent_resources import (
    _THREAD_ID,
    AgentResources,
    BoundAgentResources,
    _agent_layout_lock,
    _prepare_database_file,
    _prepare_shared_backend_layout,
    _secure_directory,
    _shared_backend_layout_exists,
)
from msgflux.runtime.approvals import SQLiteApprovalStore
from msgflux.tasks import SQLiteTaskStore
from msgflux.tasks.dataclasses import TaskActivity, TaskRecord
from msgflux.tasks.records import TaskSummary


class _ThreadScopedSQLiteTaskStore(SQLiteTaskStore):
    """A lightweight task view that shares one native SQLite connection.

    It is a concrete subclass rather than a dynamic delegate so runtime
    ``TaskStoreProtocol`` checks work on Python 3.12. The view never closes the
    shared pool and filters durable rows by the binding's thread before result
    or metadata payloads are materialized.
    """

    def __init__(self, store: SQLiteTaskStore, thread_id: str) -> None:
        # Deliberately bypass SQLiteTaskStore.__init__: this view owns no
        # connection and must reuse the backend pool's lock and adapter.
        self.path = store.path
        self._lock = store._lock
        self._conn = store._conn
        self._clock = store._clock
        self._thread_id = thread_id

    def _require_thread(self, thread_id: str) -> None:
        if thread_id != self._thread_id:
            raise ValueError(
                "Task store view cannot access another Agent thread: "
                "thread_id must match the bound thread"
            )

    def create(
        self,
        tool_name: str,
        *,
        task_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> TaskRecord:
        values = dict(metadata or {})
        selected = values.get("thread_id")
        if selected is None:
            values["thread_id"] = self._thread_id
        elif selected != self._thread_id:
            raise ValueError("Task metadata thread_id must match the bound thread")
        return super().create(tool_name, task_id=task_id, metadata=values)

    def get(self, task_id: str, *, include_result: bool = True) -> TaskRecord | None:
        return self._get_for_thread(
            self._thread_id, task_id, include_result=include_result
        )

    def list(
        self, *, status: str | None = None, include_result: bool = True
    ) -> List[TaskRecord]:
        return self._list_for_thread(
            self._thread_id, status=status, include_result=include_result
        )

    def list_summaries(self, *, thread_id: str, run_id: str) -> List[TaskSummary]:
        self._require_thread(thread_id)
        return super().list_summaries(thread_id=thread_id, run_id=run_id)

    def has_unfinished_for_thread(self, *, thread_id: str) -> bool:
        self._require_thread(thread_id)
        return super().has_unfinished_for_thread(thread_id=thread_id)

    def update_metadata(
        self, task_id: str, metadata: Mapping[str, Any]
    ) -> TaskRecord | None:
        values = dict(metadata)
        if "thread_id" in values:
            self._require_thread(values["thread_id"])
        return super().update_metadata(task_id, values)

    def list_activity(
        self, task_id: str, *, limit: int | None = None
    ) -> List[TaskActivity]:
        return self._task_call(task_id, super().list_activity, missing=[], limit=limit)

    def add_activity(
        self,
        task_id: str,
        *,
        kind: str,
        summary: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskActivity | None:
        return self._task_call(
            task_id,
            super().add_activity,
            kind=kind,
            summary=summary,
            metadata=metadata,
        )

    def get_last_activity(self, task_id: str) -> TaskActivity | None:
        return self._task_call(task_id, super().get_last_activity)

    def get_worker_lease(self, task_id):
        return self._task_call(task_id, super().get_worker_lease)

    def renew_worker(self, task_id, owner_id, *, lease_seconds):
        if lease_seconds <= 0:
            raise ValueError("`lease_seconds` must be positive")
        return self._task_call(
            task_id,
            super().renew_worker,
            owner_id,
            missing=False,
            lease_seconds=lease_seconds,
        )

    def release_worker(self, task_id, owner_id):
        return self._task_call(task_id, super().release_worker, owner_id, missing=False)

    def enqueue_message(self, task_id, message_id, message):
        return self._task_call(
            task_id, super().enqueue_message, message_id, message, missing=False
        )

    def pending_messages(self, task_id):
        return self._task_call(task_id, super().pending_messages, missing=[])

    def ack_messages(self, task_id, message_ids):
        return self._task_call(task_id, super().ack_messages, message_ids)

    def _task_call(self, task_id, method, *args, missing=None, **kwargs):
        # Check visibility and execute under the same native store lock. Lease
        # and message methods use direct SQL rather than calling self.get().
        with self._lock:
            if self.get(task_id, include_result=False) is None:
                return missing
            return method(task_id, *args, **kwargs)

    def close(self) -> None:
        """Borrowed views never close the backend's shared connection."""
        return None


class SQLiteAgentBackend(AgentResources):
    """Serve any number of durable Agent threads from four SQLite connections.

    Construction is lazy and creates no filesystem entries. The first
    :meth:`bind` validates or initializes the shared layout under the same
    process lock used by the legacy per-thread resource layout, then opens one
    checkpoint, task, inbox, and approval adapter for every logical binding.

    ``cache_bytes`` is a SQLite page-cache target split evenly across those four
    connections. It is configuration, not a hard bound on process RSS.
    """

    def __init__(
        self,
        agent_dir: str | os.PathLike[str],
        *,
        cache_bytes: int = 2 * 1024 * 1024,
    ) -> None:
        super().__init__(agent_dir)
        if type(cache_bytes) is not int or cache_bytes < 4 * 1024:
            raise ValueError("cache_bytes must be an integer of at least 4096 bytes")
        self.cache_bytes = cache_bytes
        self._page_cache_kib = cache_bytes // 4 // 1024
        self._lock = RLock()
        self._pool: (
            tuple[
                SQLiteCheckpointStore,
                SQLiteTaskStore,
                SQLiteAgentInboxStore,
                SQLiteApprovalStore,
            ]
            | None
        ) = None
        self._pending_cleanup: tuple[Any, ...] = ()
        self._closed_resources: set[int] = set()
        self._active_bindings = 0
        self._closed = False
        self._close_failed = False

    @property
    def active_bindings(self) -> int:
        """Number of live :class:`BoundAgentResources` leases."""
        with self._lock:
            return self._active_bindings

    @property
    def connection_count(self) -> int:
        """Open pooled adapters: zero before first bind and after close."""
        with self._lock:
            resources = self._pool or self._pending_cleanup
            return sum(
                id(resource) not in self._closed_resources for resource in resources
            )

    def bind(
        self,
        thread_id: str,
        *,
        namespace: str,
        verbose: bool = False,
    ) -> BoundAgentResources:
        """Borrow the shared stores for one logical Agent thread."""
        if not isinstance(thread_id, str) or not _THREAD_ID.fullmatch(thread_id):
            raise ValueError("thread_id must be a safe single directory component")
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        if type(verbose) is not bool:
            raise TypeError("verbose must be a boolean")

        with self._lock:
            self._reject_unavailable_bind_locked()
            with _agent_layout_lock(self.agent_dir):
                _prepare_shared_backend_layout(self.agent_dir)
                self._ensure_pool_locked()
                thread_dir = self.agent_dir / "threads" / thread_id
                _secure_directory(thread_dir.parent)
                _secure_directory(thread_dir)
            checkpoints, tasks, _inbox_store, approvals = self._pool
            task_view = _ThreadScopedSQLiteTaskStore(tasks, thread_id)
            inbox = AgentInbox(
                store=_inbox_store,
                namespace=namespace,
                thread_id=thread_id,
                owner=namespace,
                verbose=verbose,
            )
            binding = BoundAgentResources(
                thread_id=thread_id,
                namespace=namespace,
                thread_dir=thread_dir,
                checkpoint_store=checkpoints,
                task_store=task_view,
                inbox_store=_inbox_store,
                agent_inbox=inbox,
                approval_store=approvals,
                backend=self,
                _close_callback=self._release_binding,
            )
            self._active_bindings += 1
            return binding

    def has_thread(self, thread_id: str) -> bool:
        """Report an existing logical binding without creating or loading it.

        A binding may contain queued tasks, messages or approvals before its
        first checkpoint. Its private artifact directory records that identity.
        """
        if not isinstance(thread_id, str) or not _THREAD_ID.fullmatch(thread_id):
            return False
        with self._lock, _agent_layout_lock(self.agent_dir, create=False):
            if self._closed:
                return False
            if self._close_failed:
                raise RuntimeError("Agent backend close failed; retry close before use")
            if not _shared_backend_layout_exists(self.agent_dir):
                return False
            path = self.agent_dir / "threads" / thread_id
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                return False
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("Agent thread binding must be a real directory")
            if metadata.st_uid != os.getuid():
                raise PermissionError("Agent thread bindings must be user-owned")
            return True

    def close(self) -> None:
        """Close the pool after every binding lease has been released.

        A lease refusal does not poison the backend. A genuine adapter close
        failure quarantines new binds until a retry closes the remaining
        handles; handles that already closed are never closed twice.
        """
        with self._lock:
            if self._closed:
                return
            if self._active_bindings:
                raise RuntimeError(
                    "Cannot close SQLiteAgentBackend while "
                    f"{self._active_bindings} binding leases are active"
                )
            resources = self._pool or self._pending_cleanup
            if not resources:
                self._closed = True
                self._close_failed = False
                return
            errors = []
            for resource in reversed(resources):
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
                self._close_failed = True
                raise ExceptionGroup(
                    "Failed to close SQLite Agent backend stores", errors
                )
            self._pool = None
            self._pending_cleanup = ()
            self._closed = True
            self._close_failed = False

    async def aclose(self) -> None:
        """Close the host-owned pool from asynchronous shutdown code."""
        self.close()

    def _reject_unavailable_bind_locked(self) -> None:
        if self._closed:
            raise RuntimeError("SQLiteAgentBackend is closed")
        if self._close_failed:
            raise RuntimeError("Agent backend close failed; retry close before binding")

    def _ensure_pool_locked(self) -> None:
        if self._pool is not None:
            return
        backend_dir = self.agent_dir / "runtime" / "backend"
        opened = []
        try:
            checkpoint_path = backend_dir / "checkpoints.sqlite3"
            _prepare_database_file(checkpoint_path)
            checkpoints = SQLiteCheckpointStore(path=str(checkpoint_path))
            opened.append(checkpoints)

            task_path = backend_dir / "tasks.sqlite3"
            _prepare_database_file(task_path)
            tasks = SQLiteTaskStore(path=str(task_path))
            opened.append(tasks)
            # Logical bindings must not scan every other thread's task history.
            tasks._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_backend_thread_updated "
                "ON tasks(json_extract(metadata, '$.thread_id'), updated_at DESC)"
            )
            tasks._conn.commit()

            inbox_path = backend_dir / "inbox.sqlite3"
            _prepare_database_file(inbox_path)
            inbox = SQLiteAgentInboxStore(path=str(inbox_path))
            opened.append(inbox)

            approval_path = backend_dir / "approvals.sqlite3"
            _prepare_database_file(approval_path)
            approvals = SQLiteApprovalStore(path=str(approval_path))
            opened.append(approvals)

            for resource in opened:
                self._apply_cache_target(resource)
            self._pool = (checkpoints, tasks, inbox, approvals)
        except BaseException as error:
            cleanup_failed = False
            for resource in reversed(opened):
                try:
                    resource.close()
                except BaseException as cleanup_error:
                    cleanup_failed = True
                    error.add_note(
                        f"Backend pool cleanup also failed: {cleanup_error!r}"
                    )
                else:
                    self._closed_resources.add(id(resource))
            if cleanup_failed:
                # Retain failed handles so host shutdown can retry cleanup.
                self._pending_cleanup = tuple(opened)
                self._close_failed = True
            else:
                self._closed_resources.clear()
            raise

    def _apply_cache_target(self, resource: Any) -> None:
        # SQLite's negative cache_size is a per-connection KiB budget.
        resource._conn.execute(f"PRAGMA cache_size = -{self._page_cache_kib}")

    def _release_binding(self) -> None:
        with self._lock:
            if self._active_bindings <= 0:
                raise RuntimeError("Agent backend binding lease is already released")
            self._active_bindings -= 1
