"""Structural contract for background task stores.

Custom providers can implement this protocol without inheriting a concrete
store. The conformance tests exercise the semantics that signatures alone
cannot enforce, especially message replay and conditional resume.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Protocol, runtime_checkable

from msgflux.tasks.dataclasses import TaskActivity, TaskRecord
from msgflux.tasks.lease import TaskLease


@runtime_checkable
class TaskStoreProtocol(Protocol):
    """Operations required by background tools and durable agent tasks."""

    def create(
        self,
        tool_name: str,
        *,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord: ...

    def get(self, task_id: str) -> TaskRecord | None: ...

    def list(self, *, status: str | None = None) -> List[TaskRecord]: ...

    def list_activity(
        self, task_id: str, *, limit: int | None = None
    ) -> List[TaskActivity]: ...

    def get_last_activity(self, task_id: str) -> TaskActivity | None: ...

    def add_activity(
        self,
        task_id: str,
        *,
        kind: str,
        summary: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskActivity | None: ...

    def update_metadata(
        self, task_id: str, metadata: Mapping[str, Any]
    ) -> TaskRecord | None: ...

    def set_running(
        self,
        task_id: str,
        *,
        stage: str | None = None,
        message: str | None = None,
        owner_id: str | None = None,
    ) -> TaskRecord | None: ...

    def update_progress(
        self,
        task_id: str,
        *,
        stage: str | None = None,
        message: str | None = None,
        current: int | None = None,
        total: int | None = None,
        percent: float | None = None,
        owner_id: str | None = None,
    ) -> TaskRecord | None: ...

    def complete(
        self, task_id: str, result: Any, *, owner_id: str | None = None
    ) -> TaskRecord | None: ...

    def fail(
        self, task_id: str, error: Any, *, owner_id: str | None = None
    ) -> TaskRecord | None: ...

    def interrupt(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        owner_id: str | None = None,
    ) -> TaskRecord | None: ...

    def pause(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        owner_id: str | None = None,
    ) -> TaskRecord | None: ...

    def request_interrupt(self, task_id: str) -> TaskRecord | None: ...

    def clear_interrupt_request(self, task_id: str) -> TaskRecord | None: ...

    def requeue(
        self,
        task_id: str,
        *,
        expected_status: str | None = None,
        expected_generation: int | None = None,
        run_id: str | None = None,
    ) -> TaskRecord | None: ...

    def enqueue_message(self, task_id: str, message_id: str, message: str) -> bool: ...

    def pending_messages(self, task_id: str) -> List[tuple[str, str]]: ...

    def ack_messages(self, task_id: str, message_ids: List[str]) -> None: ...

    def claim_worker(
        self,
        task_id: str,
        owner_id: str,
        *,
        lease_seconds: float,
        recover_expired: bool = False,
    ) -> TaskLease | None: ...

    def get_worker_lease(self, task_id: str) -> TaskLease | None: ...

    def renew_worker(
        self, task_id: str, owner_id: str, *, lease_seconds: float
    ) -> bool: ...
