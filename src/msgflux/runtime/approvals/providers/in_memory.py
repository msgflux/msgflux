from __future__ import annotations

import time
from threading import RLock
from typing import Callable

from msgflux.data.stores.registry import register_store
from msgflux.runtime.approvals.base import ApprovalStore, UpdateApproval
from msgflux.runtime.approvals.records import (
    ApprovalEvent,
    ApprovalRecord,
    require_name,
)


@register_store()
class InMemoryApprovalStore(ApprovalStore):
    """Process-local approval journal for testing and prototyping."""

    provider = "in_memory"

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        super().__init__(clock=clock)
        self._lock = RLock()
        self._data: dict[tuple[str, str], ApprovalRecord] = {}
        self._events: dict[tuple[str, str], list[ApprovalEvent]] = {}

    def _update(
        self, namespace: str, request_id: str, update: UpdateApproval
    ) -> ApprovalRecord | None:
        key = (namespace, request_id)
        with self._lock:
            previous = self._data.get(key)
            record = update(previous)
            if record is not None and record != previous:
                event = ApprovalEvent.from_record(record)
                self._events.setdefault(key, []).append(event)
                self._data[key] = record
            return record

    def _records(
        self, namespace: str, thread_id: str, run_id: str
    ) -> list[ApprovalRecord]:
        with self._lock:
            return [
                record
                for record in self._data.values()
                if (
                    record.binding.namespace,
                    record.binding.thread_id,
                    record.binding.run_id,
                )
                == (namespace, thread_id, run_id)
            ]

    def events(self, namespace: str, request_id: str) -> list[ApprovalEvent]:
        with self._lock:
            return list(
                self._events.get(
                    (require_name(namespace), require_name(request_id)), ()
                )
            )
