from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Mapping
from uuid import uuid4

from msgflux.exceptions import TaskIdCollisionError
from msgflux.tasks.dataclasses import TaskActivity, TaskProgress, TaskRecord
from msgflux.tasks.registry import register_task_store
from msgflux.tasks.types import SQLiteTaskStoreType
from msgflux.utils.time import utc_now_isoformat

_CREATE_TABLES = """\
CREATE TABLE IF NOT EXISTS tasks (
    task_id      TEXT PRIMARY KEY,
    tool_name    TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    completed_at TEXT,
    result       TEXT,
    error        TEXT,
    progress     TEXT NOT NULL,
    metadata     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_status
    ON tasks(status, updated_at DESC);

CREATE TABLE IF NOT EXISTS task_activity (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    summary    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata   TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(task_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_task_activity_task
    ON task_activity(task_id, id ASC);
"""


@register_task_store
class SQLiteTaskStore(SQLiteTaskStoreType):
    """SQLite-backed store for background task state."""

    provider = "default"

    def __init__(self, path: str = ".msgflux/tasks.sqlite3") -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_CREATE_TABLES)
        self._conn.commit()

    @staticmethod
    def _serialize(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _deserialize(text: str | None) -> Any:
        if text is None:
            return None
        return json.loads(text)

    def _row_to_task(self, row: sqlite3.Row | tuple[Any, ...]) -> TaskRecord:
        return TaskRecord(
            task_id=row[0],
            tool_name=row[1],
            status=row[2],
            created_at=row[3],
            updated_at=row[4],
            completed_at=row[5],
            result=self._deserialize(row[6]),
            error=row[7],
            progress=TaskProgress(**self._deserialize(row[8])),
            metadata=self._deserialize(row[9]),
        )

    def _save_task(self, task: TaskRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO tasks
                (task_id, tool_name, status, created_at, updated_at,
                 completed_at, result, error, progress, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                tool_name=excluded.tool_name,
                status=excluded.status,
                updated_at=excluded.updated_at,
                completed_at=excluded.completed_at,
                result=excluded.result,
                error=excluded.error,
                progress=excluded.progress,
                metadata=excluded.metadata
            """,
            (
                task.task_id,
                task.tool_name,
                task.status,
                task.created_at,
                task.updated_at,
                task.completed_at,
                self._serialize(task.result),
                task.error,
                self._serialize(task.progress.to_dict()),
                self._serialize(task.metadata),
            ),
        )

    def _insert_task(self, task: TaskRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO tasks
                (task_id, tool_name, status, created_at, updated_at,
                 completed_at, result, error, progress, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task.task_id,
                task.tool_name,
                task.status,
                task.created_at,
                task.updated_at,
                task.completed_at,
                self._serialize(task.result),
                task.error,
                self._serialize(task.progress.to_dict()),
                self._serialize(task.metadata),
            ),
        )

    def _append_activity(
        self,
        task_id: str,
        *,
        kind: str,
        summary: str,
        created_at: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskActivity:
        activity = TaskActivity(
            task_id=task_id,
            kind=kind,
            summary=summary,
            created_at=created_at,
            metadata=dict(metadata or {}),
        )
        self._conn.execute(
            """
            INSERT INTO task_activity (task_id, kind, summary, created_at, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                activity.task_id,
                activity.kind,
                activity.summary,
                activity.created_at,
                self._serialize(activity.metadata),
            ),
        )
        return activity

    def create(
        self,
        tool_name: str,
        *,
        task_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> TaskRecord:
        with self._lock:
            now = utc_now_isoformat()
            task = TaskRecord(
                task_id=task_id or uuid4().hex[:8],
                tool_name=tool_name,
                status="queued",
                created_at=now,
                updated_at=now,
                metadata=dict(metadata or {}),
            )
            try:
                self._insert_task(task)
            except sqlite3.IntegrityError as exc:
                self._conn.rollback()
                raise TaskIdCollisionError(task.task_id) from exc
            self._append_activity(
                task.task_id,
                kind="status",
                summary="Task queued.",
                created_at=now,
                metadata={"status": "queued", "tool": tool_name},
            )
            self._conn.commit()
            return self.get(task.task_id)  # type: ignore[return-value]

    def get(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT task_id, tool_name, status, created_at, updated_at,
                       completed_at, result, error, progress, metadata
                FROM tasks WHERE task_id=?
                """,
                (task_id,),
            ).fetchone()
            return self._row_to_task(row) if row is not None else None

    def list(self, *, status: str | None = None) -> List[TaskRecord]:
        with self._lock:
            query = (
                "SELECT task_id, tool_name, status, created_at, updated_at, "
                "completed_at, result, error, progress, metadata FROM tasks"
            )
            params: List[Any] = []
            if status is not None:
                query += " WHERE status=?"
                params.append(status)
            query += " ORDER BY updated_at DESC"
            rows = self._conn.execute(query, tuple(params)).fetchall()
            return [self._row_to_task(row) for row in rows]

    def list_activity(
        self, task_id: str, *, limit: int | None = None
    ) -> List[TaskActivity]:
        with self._lock:
            query = (
                "SELECT task_id, kind, summary, created_at, metadata "
                "FROM task_activity WHERE task_id=? ORDER BY id ASC"
            )
            params: List[Any] = [task_id]
            if limit is not None:
                query = (
                    "SELECT task_id, kind, summary, created_at, metadata "
                    "FROM task_activity WHERE task_id=? ORDER BY id DESC LIMIT ?"
                )
                params.append(limit)
            rows = self._conn.execute(query, tuple(params)).fetchall()
            activities = [
                TaskActivity(
                    task_id=row[0],
                    kind=row[1],
                    summary=row[2],
                    created_at=row[3],
                    metadata=self._deserialize(row[4]),
                )
                for row in rows
            ]
            if limit is not None:
                activities.reverse()
            return activities

    def get_last_activity(self, task_id: str) -> TaskActivity | None:
        items = self.list_activity(task_id, limit=1)
        return items[-1] if items else None

    def add_activity(
        self,
        task_id: str,
        *,
        kind: str,
        summary: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskActivity | None:
        with self._lock:
            if self.get(task_id) is None:
                return None
            activity = self._append_activity(
                task_id,
                kind=kind,
                summary=summary,
                created_at=utc_now_isoformat(),
                metadata=metadata,
            )
            self._conn.commit()
            return activity

    def update_metadata(
        self,
        task_id: str,
        metadata: Mapping[str, Any],
    ) -> TaskRecord | None:
        with self._lock:
            task = self.get(task_id)
            if task is None:
                return None
            task.metadata.update(dict(metadata))
            task.updated_at = utc_now_isoformat()
            self._save_task(task)
            self._conn.commit()
            return self.get(task.task_id)

    def _update_task(
        self,
        task: TaskRecord,
        *,
        activity: TaskActivity | None,
    ) -> TaskRecord:
        with self._lock:
            self._save_task(task)
            if activity is not None:
                self._append_activity(
                    activity.task_id,
                    kind=activity.kind,
                    summary=activity.summary,
                    created_at=activity.created_at,
                    metadata=activity.metadata,
                )
            self._conn.commit()
            return self.get(task.task_id)  # type: ignore[return-value]

    def set_running(
        self,
        task_id: str,
        *,
        stage: str | None = None,
        message: str | None = None,
    ) -> TaskRecord | None:
        task = self.get(task_id)
        if task is None:
            return None
        task.status = "running"
        task.updated_at = utc_now_isoformat()
        if stage is not None:
            task.progress.stage = stage
        if message is not None:
            task.progress.message = message
        return self._update_task(
            task,
            activity=TaskActivity(
                task_id=task_id,
                kind="status",
                summary="Task running.",
                created_at=task.updated_at,
                metadata={
                    "status": "running",
                    "stage": task.progress.stage,
                    "message": task.progress.message,
                },
            ),
        )

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
        task = self.get(task_id)
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
        return self._update_task(
            task,
            activity=TaskActivity(
                task_id=task_id,
                kind="progress",
                summary=task.progress.message
                or task.progress.stage
                or "Progress updated.",
                created_at=task.updated_at,
                metadata=task.progress.to_dict(),
            ),
        )

    def complete(self, task_id: str, result: Any) -> TaskRecord | None:
        task = self.get(task_id)
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
        return self._update_task(
            task,
            activity=TaskActivity(
                task_id=task_id,
                kind="status",
                summary="Task completed.",
                created_at=now,
                metadata={"status": "completed"},
            ),
        )

    def fail(self, task_id: str, error: Any) -> TaskRecord | None:
        task = self.get(task_id)
        if task is None:
            return None
        now = utc_now_isoformat()
        task.status = "failed"
        task.updated_at = now
        task.completed_at = now
        task.error = str(error)
        return self._update_task(
            task,
            activity=TaskActivity(
                task_id=task_id,
                kind="error",
                summary="Task failed.",
                created_at=now,
                metadata={"status": "failed", "error": task.error},
            ),
        )

    def interrupt(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TaskRecord | None:
        task = self.get(task_id)
        if task is None:
            return None
        now = utc_now_isoformat()
        task.status = "interrupted"
        task.updated_at = now
        task.completed_at = now
        task.metadata["interrupt_requested"] = False
        if reason:
            task.metadata["interrupt_reason"] = reason
        return self._update_task(
            task,
            activity=TaskActivity(
                task_id=task_id,
                kind="status",
                summary="Task interrupted.",
                created_at=now,
                metadata={"status": "interrupted", "reason": reason},
            ),
        )

    def pause(self, task_id: str, *, reason: str | None = None) -> TaskRecord | None:
        task = self.get(task_id)
        if task is None:
            return None
        now = utc_now_isoformat()
        task.status = "paused"
        task.updated_at = now
        task.metadata["interrupt_requested"] = False
        if reason:
            task.metadata["pause_reason"] = reason
        return self._update_task(
            task,
            activity=TaskActivity(
                task_id=task_id,
                kind="status",
                summary="Task paused.",
                created_at=now,
                metadata={"status": "paused", "reason": reason},
            ),
        )

    def request_interrupt(self, task_id: str) -> TaskRecord | None:
        task = self.get(task_id)
        if task is None:
            return None
        task.updated_at = utc_now_isoformat()
        task.metadata["interrupt_requested"] = True
        return self._update_task(
            task,
            activity=TaskActivity(
                task_id=task_id,
                kind="status",
                summary="Interrupt requested.",
                created_at=task.updated_at,
                metadata={"status": task.status},
            ),
        )

    def clear_interrupt_request(self, task_id: str) -> TaskRecord | None:
        task = self.get(task_id)
        if task is None:
            return None
        task.metadata["interrupt_requested"] = False
        task.updated_at = utc_now_isoformat()
        return self._update_task(task, activity=None)

    def requeue(self, task_id: str) -> TaskRecord | None:
        task = self.get(task_id)
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
        return self._update_task(
            task,
            activity=TaskActivity(
                task_id=task_id,
                kind="status",
                summary="Task re-queued.",
                created_at=now,
                metadata={"status": "queued"},
            ),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
