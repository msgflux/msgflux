from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict
from uuid import uuid4

from msgflux.exceptions import (
    TaskInterruptRequestedError,
    TaskLeaseLostError,
    TaskPauseRequestedError,
)
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    AgentNotification,
    ToolNotificationHandle,
)
from msgflux.runtime.events import EventType, emit_event
from msgflux.tasks.dataclasses import TaskRecord

if TYPE_CHECKING:
    from msgflux.tasks.protocol import TaskStoreProtocol


class TaskHandle:
    """Small mutable handle injected into background tools."""

    def __init__(
        self,
        task_id: str,
        store: TaskStoreProtocol,
        *,
        tool_name: str | None = None,
        agent_inbox: AgentInbox | None = None,
    ):
        self.task_id = task_id
        self._store = store
        self._tool_name = tool_name
        self._agent_inbox = agent_inbox
        self._owner_id: str | None = None
        self._lease_lost = False
        self._notification = ToolNotificationHandle(
            agent_inbox,
            ref=task_id,
            metadata={"tool": tool_name} if tool_name else None,
        )

    # --- Task State Updates ---

    @property
    def has_worker_lease(self) -> bool:
        return self._owner_id is not None

    def start_worker(
        self, *, lease_seconds: float, recover_expired: bool = False
    ) -> TaskRecord:
        """Atomically claim queued work before calling a tool."""
        owner_id = uuid4().hex
        lease = self._store.claim_worker(
            self.task_id,
            owner_id,
            lease_seconds=lease_seconds,
            recover_expired=recover_expired,
        )
        if lease is None:
            raise TaskLeaseLostError(self.task_id)
        self._owner_id = owner_id
        record = self._store.get(self.task_id)
        if record is None:
            raise TaskLeaseLostError(self.task_id)
        self._emit_record(EventType.TASK_UPDATE, record)
        return record

    def renew_worker(self, *, lease_seconds: float) -> bool:
        if self._owner_id is None or self._lease_lost:
            return False
        renewed = self._store.renew_worker(
            self.task_id, self._owner_id, lease_seconds=lease_seconds
        )
        if not renewed:
            self._lease_lost = True
        return renewed

    def _owned_record(self, record: TaskRecord | None) -> TaskRecord | None:
        if self._owner_id is not None and record is None:
            raise TaskLeaseLostError(self.task_id)
        return record

    def pending_messages(self) -> list[tuple[str, str]]:
        """Return task-addressed messages awaiting durable inbox consumption."""
        return self._store.pending_messages(self.task_id)

    def ack_messages(self, message_ids: list[str]) -> None:
        self._store.ack_messages(self.task_id, message_ids)

    def forward_messages(self, inbox: AgentInbox) -> None:
        """Replay pending messages into this run; retries keep the same identity."""
        for message_id, message in self.pending_messages():
            inbox.publish(
                AgentNotification(
                    notification_id=message_id,
                    source="task_message",
                    ref=self.task_id,
                    status="message",
                    metadata={"direction": "root_to_task", "message": message},
                    dedupe_key=f"task_message:{message_id}",
                )
            )

    def _emit_record(self, event_type: str, record: TaskRecord | None) -> None:
        if record is None:
            return
        emit_event(
            event_type,
            {
                "task_id": record.task_id,
                "tool_name": record.tool_name,
                "status": record.status,
                "progress": record.progress.to_dict(),
            },
        )

    def set_running(
        self,
        *,
        stage: str | None = None,
        message: str | None = None,
    ) -> TaskRecord | None:
        record = self._store.set_running(
            task_id=self.task_id,
            stage=stage,
            message=message,
            owner_id=self._owner_id,
        )
        record = self._owned_record(record)
        self._emit_record(EventType.TASK_UPDATE, record)
        return record

    def update_progress(
        self,
        *,
        stage: str | None = None,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
        percent: float | None = None,
    ) -> TaskRecord | None:
        record = self._store.update_progress(
            task_id=self.task_id,
            stage=stage,
            message=message,
            current=current,
            total=total,
            percent=percent,
            owner_id=self._owner_id,
        )
        record = self._owned_record(record)
        self._emit_record(EventType.TASK_UPDATE, record)
        return record

    def complete(self, result: Any) -> TaskRecord | None:
        record = self._owned_record(
            self._store.complete(self.task_id, result, owner_id=self._owner_id)
        )
        self._emit_record(EventType.TASK_END, record)
        return record

    def fail(self, error: Any) -> TaskRecord | None:
        record = self._owned_record(
            self._store.fail(self.task_id, error, owner_id=self._owner_id)
        )
        self._emit_record(EventType.TASK_END, record)
        return record

    def interrupt(self, *, reason: str | None = None) -> TaskRecord | None:
        record = self._owned_record(
            self._store.interrupt(self.task_id, reason=reason, owner_id=self._owner_id)
        )
        self._emit_record(EventType.TASK_END, record)
        return record

    def pause(self, *, reason: str | None = None) -> TaskRecord | None:
        record = self._owned_record(
            self._store.pause(self.task_id, reason=reason, owner_id=self._owner_id)
        )
        self._emit_record(EventType.TASK_END, record)
        return record

    def is_interrupt_requested(self) -> bool:
        if self._lease_lost:
            raise TaskLeaseLostError(self.task_id)
        task = self._store.get(self.task_id)
        if task is None:
            return False
        return bool(task.metadata.get("interrupt_requested"))

    def raise_if_interrupted(self) -> None:
        if self.is_interrupt_requested():
            raise TaskInterruptRequestedError(self.task_id)

    def raise_if_paused(self) -> None:
        if self._lease_lost:
            raise TaskLeaseLostError(self.task_id)
        task = self._store.get(self.task_id)
        if task is not None and task.status == "paused":
            raise TaskPauseRequestedError(self.task_id)

    # --- Agent Notifications ---

    def notify(
        self,
        *,
        status: str,
        metadata: Dict[str, Any] | None = None,
        dedupe_key: str | None = None,
        source: str = "task",
    ) -> AgentNotification | None:
        return self._notification.publish(
            status=status,
            metadata=metadata,
            dedupe_key=dedupe_key,
            source=source,
        )
