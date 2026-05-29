from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Dict, List, Mapping
from uuid import uuid4

from msgflux.agent_inbox import AgentInbox, AgentNotification, ToolNotificationHandle
from msgflux.exceptions import TaskPauseRequestedError, TaskStopRequestedError

# --- Module Utilities ---


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Task Models ---


@dataclass
class TaskProgress:
    stage: str | None = None
    message: str | None = None
    current: int | None = None
    total: int | None = None
    percent: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskActivity:
    task_id: str
    kind: str
    summary: str
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRecord:
    task_id: str
    tool_name: str
    status: str
    created_at: str
    updated_at: str
    completed_at: str | None = None
    result: Any | None = None
    error: str | None = None
    progress: TaskProgress = field(default_factory=TaskProgress)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TaskStore:
    """Thread-safe in-memory store for background task state."""

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
        now = _utc_now()
        task = TaskRecord(
            task_id=task_id or uuid4().hex[:8],
            tool_name=tool_name,
            status="queued",
            created_at=now,
            updated_at=now,
            metadata=deepcopy(metadata or {}),
        )
        with self._lock:
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
                created_at=_utc_now(),
                metadata=deepcopy(dict(metadata or {})),
            )
            self._activities.setdefault(task_id, []).append(activity)
            return deepcopy(activity)

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
            task.updated_at = _utc_now()
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
            task.updated_at = _utc_now()
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
            now = _utc_now()
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
            now = _utc_now()
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

    def stop(self, task_id: str, *, reason: str | None = None) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            now = _utc_now()
            task.status = "stopped"
            task.updated_at = now
            task.completed_at = now
            task.metadata["stop_requested"] = False
            if reason:
                task.metadata["stop_reason"] = reason
            self._activities.setdefault(task_id, []).append(
                TaskActivity(
                    task_id=task_id,
                    kind="status",
                    summary="Task stopped.",
                    created_at=now,
                    metadata={"status": "stopped", "reason": reason},
                )
            )
            return deepcopy(task)

    def pause(self, task_id: str, *, reason: str | None = None) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            now = _utc_now()
            task.status = "paused"
            task.updated_at = now
            task.metadata["stop_requested"] = False
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

    def request_stop(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.updated_at = _utc_now()
            task.metadata["stop_requested"] = True
            self._activities.setdefault(task_id, []).append(
                TaskActivity(
                    task_id=task_id,
                    kind="status",
                    summary="Stop requested.",
                    created_at=task.updated_at,
                    metadata={"status": task.status},
                )
            )
            return deepcopy(task)

    def clear_stop_request(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.metadata["stop_requested"] = False
            task.updated_at = _utc_now()
            return deepcopy(task)

    def requeue(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            now = _utc_now()
            task.status = "queued"
            task.updated_at = now
            task.completed_at = None
            task.result = None
            task.error = None
            task.metadata["stop_requested"] = False
            task.metadata.pop("stop_reason", None)
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


class TaskActivityRecorder:
    """Small recorder for compact task activity entries."""

    def __init__(self, task_id: str, store: TaskStore):
        self.task_id = task_id
        self._store = store

    # --- Activity Publishing ---

    def add(
        self,
        *,
        kind: str,
        summary: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskActivity | None:
        return self._store.add_activity(
            self.task_id,
            kind=kind,
            summary=summary,
            metadata=metadata,
        )

    def tool_call(self, tool_name: str, parameters: Any) -> TaskActivity | None:
        summary = f"{tool_name}({self._truncate(str(parameters))})"
        return self.add(kind="tool_call", summary=summary, metadata={"tool": tool_name})

    @staticmethod
    def _truncate(value: str, *, limit: int = 140) -> str:
        text = " ".join(value.split())
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."


class TaskHandle:
    """Small mutable handle injected into background tools."""

    def __init__(
        self,
        task_id: str,
        store: TaskStore,
        *,
        tool_name: str | None = None,
        agent_inbox: AgentInbox | None = None,
    ):
        self.task_id = task_id
        self._store = store
        self._tool_name = tool_name
        self._agent_inbox = agent_inbox
        self._notification = ToolNotificationHandle(
            agent_inbox,
            ref=task_id,
            metadata={"tool": tool_name} if tool_name else None,
        )

    # --- Task State Updates ---

    def set_running(
        self,
        *,
        stage: str | None = None,
        message: str | None = None,
    ) -> TaskRecord | None:
        return self._store.set_running(
            task_id=self.task_id, stage=stage, message=message
        )

    def update_progress(
        self,
        *,
        stage: str | None = None,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
        percent: float | None = None,
    ) -> TaskRecord | None:
        return self._store.update_progress(
            task_id=self.task_id,
            stage=stage,
            message=message,
            current=current,
            total=total,
            percent=percent,
        )

    def complete(self, result: Any) -> TaskRecord | None:
        return self._store.complete(self.task_id, result)

    def fail(self, error: Any) -> TaskRecord | None:
        return self._store.fail(self.task_id, error)

    def stop(self, *, reason: str | None = None) -> TaskRecord | None:
        return self._store.stop(self.task_id, reason=reason)

    def pause(self, *, reason: str | None = None) -> TaskRecord | None:
        return self._store.pause(self.task_id, reason=reason)

    def is_stop_requested(self) -> bool:
        task = self._store.get(self.task_id)
        if task is None:
            return False
        return bool(task.metadata.get("stop_requested"))

    def raise_if_stopped(self) -> None:
        if self.is_stop_requested():
            raise TaskStopRequestedError(self.task_id)

    def raise_if_paused(self) -> None:
        task = self._store.get(self.task_id)
        if task is not None and task.status == "paused":
            raise TaskPauseRequestedError(self.task_id)

    # --- Agent Notifications ---

    def notify(
        self,
        *,
        status: str,
        hint: str | None = None,
        metadata: Dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        source: str = "task",
    ) -> AgentNotification | None:
        return self._notification.publish(
            status=status,
            hint=hint,
            metadata=metadata,
            dedupe_key=dedupe_key,
            source=source,
        )
