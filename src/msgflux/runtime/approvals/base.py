"""Shared state machine for durable, single-use approval decisions."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Callable

from msgflux.data.stores.types import ApprovalStoreType
from msgflux.runtime.approvals.records import (
    ApprovalBinding,
    ApprovalEvent,
    ApprovalRecord,
    require_name,
    require_time,
)
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.permissions import require_permissions


class ApprovalConflictError(RuntimeError):
    """The request changed, expired, was denied, or has already been consumed."""


class ApprovalExpiredError(ApprovalConflictError):
    """The request's absolute deadline has been reached."""


UpdateApproval = Callable[[ApprovalRecord | None], ApprovalRecord | None]


class ApprovalStore(ABC, ApprovalStoreType):
    """Host-owned journal; decisions do not themselves grant execution authority.

    Providers must atomically commit each new revision and its audit event.
    Async wrappers preserve contextvars but cancellation cannot undo a storage
    transaction that has already committed in the worker thread.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock

    @abstractmethod
    def _update(
        self,
        namespace: str,
        request_id: str,
        update: UpdateApproval,
    ) -> ApprovalRecord | None:
        """Atomically read, transform and journal one request, including creation."""

    @abstractmethod
    def _records(
        self, namespace: str, thread_id: str, run_id: str
    ) -> list[ApprovalRecord]:
        """Read immutable snapshots in creation order."""

    @abstractmethod
    def events(self, namespace: str, request_id: str) -> list[ApprovalEvent]:
        """Read committed audit transitions in request revision order."""

    def _expire(
        self, record: ApprovalRecord | None, now: float | None = None
    ) -> ApprovalRecord | None:
        now = require_time(self._clock() if now is None else now)
        if record is not None and record.status in {"pending", "approved"}:
            if now >= record.expires_at:
                return self._transition(record, "expired", now)
        return record

    @staticmethod
    def _transition(
        record: ApprovalRecord, status: str, now: float, **kwargs
    ) -> ApprovalRecord:
        return replace(
            record,
            status=status,
            revision=record.revision + 1,
            updated_at=max(now, record.updated_at),
            **kwargs,
        )

    def request(
        self,
        binding: ApprovalBinding,
        *,
        request_id: str,
        expires_at: float,
    ) -> ApprovalRecord:
        """Idempotently create a request; reuse of its ID cannot change its binding."""
        if not isinstance(binding, ApprovalBinding):
            raise TypeError("A request requires an ApprovalBinding")
        require_name(request_id)
        require_time(expires_at)

        def create(existing):
            if existing is not None:
                if existing.binding != binding or existing.expires_at != expires_at:
                    raise ApprovalConflictError("Approval request ID is already bound")
                return self._expire(existing)
            now = require_time(self._clock())
            if expires_at <= now:
                raise ApprovalExpiredError("Approval deadline has already elapsed")
            return ApprovalRecord(
                request_id=request_id,
                binding=binding,
                expires_at=expires_at,
                created_at=now,
                updated_at=now,
            )

        return self._update(binding.namespace, request_id, create)

    def get(self, namespace: str, request_id: str) -> ApprovalRecord | None:
        """Load current state, recording expiry when the deadline has elapsed."""
        return self._update(
            require_name(namespace), require_name(request_id), self._expire
        )

    def pending(
        self, namespace: str, thread_id: str, run_id: str
    ) -> list[ApprovalRecord]:
        """Poll unresolved requests; this is not an atomic watcher snapshot."""
        for value in (namespace, thread_id, run_id):
            require_name(value)
        records = [
            self.get(namespace, item.request_id)
            for item in self._records(
                namespace,
                thread_id,
                run_id,
            )
        ]
        return [
            record
            for record in records
            if record is not None and record.status == "pending"
        ]

    def decide(
        self,
        namespace: str,
        request_id: str,
        *,
        approved: bool,
        decided_by: str,
    ) -> ApprovalRecord:
        """Record a trusted host decision. The host authenticates the reviewer."""
        if type(approved) is not bool:
            raise TypeError("Approval decisions require a bool")
        require_name(decided_by)
        status = "approved" if approved else "denied"

        def decide(current):
            now = require_time(self._clock())
            record = self._expire(current, now)
            if record is None:
                raise KeyError("Approval request not found")
            if record.status == "expired":
                return record
            if record.status == status and record.decided_by == decided_by:
                return record
            if record.status != "pending":
                raise ApprovalConflictError("Approval decision is already settled")
            return self._transition(record, status, now, decided_by=decided_by)

        record = self._update(require_name(namespace), require_name(request_id), decide)
        if record.status == "expired":
            raise ApprovalExpiredError("Approval request expired")
        return record

    def consume(self, request_id: str, *, binding: ApprovalBinding) -> ApprovalRecord:
        """Atomically use one decision after matching the call and live authority.

        A consumed decision must not be retried after a crash: this journal does
        not make an external side effect atomic with the consumption transaction.
        """
        if not isinstance(binding, ApprovalBinding):
            raise TypeError("Consumption requires an ApprovalBinding")
        require_name(request_id)
        scope = get_execution_scope()
        if (scope.namespace, scope.thread_id, scope.run_id, scope.principal) != (
            binding.namespace,
            binding.thread_id,
            binding.run_id,
            binding.principal,
        ):
            raise PermissionError(
                "Approval consumption requires the bound live execution"
            )
        require_permissions(binding.required_permissions)

        def consume(current):
            if current is None:
                raise KeyError("Approval request not found")
            if current.binding != binding:
                raise ApprovalConflictError("Approval invocation binding changed")
            now = require_time(self._clock())
            record = self._expire(current, now)
            if record.status == "expired":
                return record
            if record.status != "approved":
                raise ApprovalConflictError("Approval is not available for consumption")
            return self._transition(record, "consumed", now)

        record = self._update(binding.namespace, request_id, consume)
        if record.status == "expired":
            raise ApprovalExpiredError("Approval request expired")
        return record

    async def arequest(
        self, binding: ApprovalBinding, *, request_id: str, expires_at: float
    ) -> ApprovalRecord:
        return await asyncio.to_thread(
            self.request, binding, request_id=request_id, expires_at=expires_at
        )

    async def aget(self, namespace: str, request_id: str) -> ApprovalRecord | None:
        return await asyncio.to_thread(self.get, namespace, request_id)

    async def apending(
        self, namespace: str, thread_id: str, run_id: str
    ) -> list[ApprovalRecord]:
        return await asyncio.to_thread(self.pending, namespace, thread_id, run_id)

    async def adecide(
        self, namespace: str, request_id: str, *, approved: bool, decided_by: str
    ) -> ApprovalRecord:
        return await asyncio.to_thread(
            self.decide, namespace, request_id, approved=approved, decided_by=decided_by
        )

    async def aconsume(
        self, request_id: str, *, binding: ApprovalBinding
    ) -> ApprovalRecord:
        return await asyncio.to_thread(self.consume, request_id, binding=binding)

    async def aevents(self, namespace: str, request_id: str) -> list[ApprovalEvent]:
        return await asyncio.to_thread(self.events, namespace, request_id)

    def close(self) -> None:
        """Release provider resources."""

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)
