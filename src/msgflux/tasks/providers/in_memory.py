from __future__ import annotations

from copy import deepcopy
from threading import RLock
from typing import Any, Dict, List, Mapping
from uuid import uuid4

from msgflux.exceptions import TaskIdCollisionError
from msgflux.tasks.dataclasses import TaskActivity, TaskRecord
from msgflux.tasks.registry import register_task_store
from msgflux.tasks.types import InMemoryTaskStoreType
from msgflux.utils.time import utc_now_isoformat


@register_task_store
class InMemoryTaskStore(InMemoryTaskStoreType):
    """Thread-safe in-memory store for background task state."""

    provider = "default"

    def __init__(self):
        self._lock = RLock()
        self._tasks: Dict[str, TaskRecord] = {}
        self._activities: Dict[str, List[TaskActivity]] = {}

    # --- Query Operations ---

    def create(
        self,
        tool_name: str,
        *,
        task_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> TaskRecord:
        now = utc_now_isoformat()
        task = TaskRecord(
            task_id=task_id or uuid4().hex[:8],
            tool_name=tool_name,
            status="queued",
            created_at=now,
            updated_at=now,
            metadata=deepcopy(metadata or {}),
        )
        with self._lock:
            if task.task_id in self._tasks:
                raise TaskIdCollisionError(task.task_id)
            self._tasks[task.task_id] = task
            self._activities[task.task_id] = []
            self._activities[task.task_id].append(
                TaskActivity(
                    task_id=task.task_id,
                    kind="status",
                    summary="Task queued.",
                    created_at=now,
                    metadata={"status": "queued", "tool": tool_name},
                )
            )
        return self.get(task.task_id)  # type: ignore[return-value]

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return deepcopy(task) if task is not None else None

    def list(self, *, status: str | None = None) -> List[TaskRecord]:
        with self._lock:
            tasks = list(self._tasks.values())
            if status is not None:
                tasks = [task for task in tasks if task.status == status]
            return deepcopy(tasks)

    def list_activity(
        self, task_id: str, *, limit: int | None = None
    ) -> List[TaskActivity]:
        with self._lock:
            activity = deepcopy(self._activities.get(task_id, []))
        if limit is not None:
            return activity[-limit:]
        return activity

    def get_last_activity(self, task_id: str) -> TaskActivity | None:
        with self._lock:
            activity = self._activities.get(task_id, [])
            if not activity:
                return None
            return deepcopy(activity[-1])

    def add_activity(
        self,
        task_id: str,
        *,
        kind: str,
        summary: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskActivity | None:
        with self._lock:
            if task_id not in self._tasks:
                return None
            activity = TaskActivity(
                task_id=task_id,
                kind=kind,
                summary=summary,
                created_at=utc_now_isoformat(),
                metadata=deepcopy(dict(metadata or {})),
            )
            self._activities.setdefault(task_id, []).append(activity)
            return deepcopy(activity)

    def update_metadata(
        self,
        task_id: str,
        metadata: Mapping[str, Any],
    ) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.metadata.update(deepcopy(dict(metadata)))
            task.updated_at = utc_now_isoformat()
            return deepcopy(task)

    # --- State Transitions ---

    def set_running(
        self,
        task_id: str,
        *,
        stage: str | None = None,
        message: str | None = None,
    ) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = "running"
            task.updated_at = utc_now_isoformat()
            if stage is not None:
                task.progress.stage = stage
            if message is not None:
                task.progress.message = message
            self._activities.setdefault(task_id, []).append(
                TaskActivity(
                    task_id=task_id,
                    kind="status",
                    summary="Task running.",
                    created_at=task.updated_at,
                    metadata={
                        "status": "running",
                        "stage": task.progress.stage,
                        "message": task.progress.message,
                    },
                )
            )
            return deepcopy(task)

    def update_progress(
        self,
        task_id: str,
        *,
        stage: str | None = None,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
        percent: float | None = None,
    ) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            if task.status == "queued":
                task.status = "running"
            task.updated_at = utc_now_isoformat()
            if stage is not None:
                task.progress.stage = stage
            if message is not None:
                task.progress.message = message
            if current is not None:
                task.progress.current = current
            if total is not None:
                task.progress.total = total
            if (
                percent is None
                and task.progress.current is not None
                and task.progress.total
            ):
                percent = (task.progress.current / task.progress.total) * 100
            if percent is not None:
                task.progress.percent = round(percent, 2)
            self._activities.setdefault(task_id, []).append(
                TaskActivity(
                    task_id=task_id,
                    kind="progress",
                    summary=task.progress.message
                    or task.progress.stage
                    or "Progress updated.",
                    created_at=task.updated_at,
                    metadata=task.progress.to_dict(),
                )
            )
            return deepcopy(task)

    def complete(self, task_id: str, result: Any) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            now = utc_now_isoformat()
            task.status = "completed"
            task.updated_at = now
            task.completed_at = now
            task.result = result
            task.error = None
            if task.progress.percent is None and task.progress.total:
                task.progress.percent = 100.0
            self._activities.setdefault(task_id, []).append(
                TaskActivity(
                    task_id=task_id,
                    kind="status",
                    summary="Task completed.",
                    created_at=now,
                    metadata={"status": "completed"},
                )
            )
            return deepcopy(task)

    def fail(self, task_id: str, error: Any) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            now = utc_now_isoformat()
            task.status = "failed"
            task.updated_at = now
            task.completed_at = now
            task.error = str(error)
            self._activities.setdefault(task_id, []).append(
                TaskActivity(
                    task_id=task_id,
                    kind="error",
                    summary="Task failed.",
                    created_at=now,
                    metadata={"status": "failed", "error": task.error},
                )
            )
            return deepcopy(task)

    def interrupt(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            now = utc_now_isoformat()
            task.status = "interrupted"
            task.updated_at = now
            task.completed_at = now
            task.metadata["interrupt_requested"] = False
            if reason:
                task.metadata["interrupt_reason"] = reason
            self._activities.setdefault(task_id, []).append(
                TaskActivity(
                    task_id=task_id,
                    kind="status",
                    summary="Task interrupted.",
                    created_at=now,
                    metadata={"status": "interrupted", "reason": reason},
                )
            )
            return deepcopy(task)

    def pause(self, task_id: str, *, reason: str | None = None) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            now = utc_now_isoformat()
            task.status = "paused"
            task.updated_at = now
            task.metadata["interrupt_requested"] = False
            if reason:
                task.metadata["pause_reason"] = reason
            self._activities.setdefault(task_id, []).append(
                TaskActivity(
                    task_id=task_id,
                    kind="status",
                    summary="Task paused.",
                    created_at=now,
                    metadata={"status": "paused", "reason": reason},
                )
            )
            return deepcopy(task)

    def request_interrupt(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.updated_at = utc_now_isoformat()
            task.metadata["interrupt_requested"] = True
            self._activities.setdefault(task_id, []).append(
                TaskActivity(
                    task_id=task_id,
                    kind="status",
                    summary="Interrupt requested.",
                    created_at=task.updated_at,
                    metadata={"status": task.status},
                )
            )
            return deepcopy(task)

    def clear_interrupt_request(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.metadata["interrupt_requested"] = False
            task.updated_at = utc_now_isoformat()
            return deepcopy(task)

    def requeue(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            now = utc_now_isoformat()
            task.status = "queued"
            task.updated_at = now
            task.completed_at = None
            task.result = None
            task.error = None
            task.metadata["interrupt_requested"] = False
            task.metadata.pop("interrupt_reason", None)
            task.metadata.pop("pause_reason", None)
            self._activities.setdefault(task_id, []).append(
                TaskActivity(
                    task_id=task_id,
                    kind="status",
                    summary="Task re-queued.",
                    created_at=now,
                    metadata={"status": "queued"},
                )
            )
            return deepcopy(task)
