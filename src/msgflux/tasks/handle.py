from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from msgflux.exceptions import TaskInterruptRequestedError, TaskPauseRequestedError
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    AgentNotification,
    ToolNotificationHandle,
)
from msgflux.tasks.dataclasses import TaskRecord

if TYPE_CHECKING:
    from msgflux.tasks.store import TaskStore


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

    def interrupt(self, *, reason: str | None = None) -> TaskRecord | None:
        return self._store.interrupt(self.task_id, reason=reason)

    def pause(self, *, reason: str | None = None) -> TaskRecord | None:
        return self._store.pause(self.task_id, reason=reason)

    def is_interrupt_requested(self) -> bool:
        task = self._store.get(self.task_id)
        if task is None:
            return False
        return bool(task.metadata.get("interrupt_requested"))

    def raise_if_interrupted(self) -> None:
        if self.is_interrupt_requested():
            raise TaskInterruptRequestedError(self.task_id)

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
