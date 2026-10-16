"""Service-owned manual maintenance with per-thread admission exclusion."""

import asyncio
import contextvars
from contextlib import nullcontext

from msgflux.data.stores.base import CheckpointConflictError
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.runtime.compaction import (
    CompactionBusyError,
    CompactionRecoveryRequiredError,
)
from msgflux.runtime.service.records import (
    ServiceBusyError,
    ServiceConflictError,
    ServiceRecoveryRequiredError,
)
from msgflux.runtime.service.store import validate_identifier


async def compact(service, thread_id, run_id=None):
    validate_identifier(thread_id, "thread_id")
    if run_id is not None:
        validate_identifier(run_id, "run_id")
    async with service._lock:
        service._require_open()
        service.store.thread(thread_id)
        service._require_no_maintenance(thread_id)
        if any(key[0] == thread_id for key in service._workers):
            raise ServiceBusyError("The thread still has an active worker")
        task = asyncio.create_task(
            _compact(service, thread_id, run_id), context=contextvars.Context()
        )
        service._maintenance_tasks[thread_id] = task
        task.add_done_callback(lambda _: _finished(service, thread_id, task))
    return await task


def _finished(service, thread_id, task):
    if service._maintenance_tasks.get(thread_id) is task:
        service._maintenance_tasks.pop(thread_id)
        service._cache.activity_finished(thread_id)
    if not task.cancelled():
        task.exception()  # Consume failures even when the HTTP waiter disconnected.


async def _compact(service, thread_id, run_id):
    async with service._use_session(thread_id) as session:
        async with service._lock:
            record = service.store.latest_admission(thread_id)
            if record is None:
                raise KeyError(f"Thread {thread_id} has no admitted runs")
            receipt = record.receipt
            if run_id is not None and receipt.run_id != run_id:
                raise ServiceConflictError(
                    "Only the latest admitted run can be compacted"
                )
            if receipt.status in {"accepted", "running"}:
                raise ServiceBusyError("The admitted run is still active")
            if receipt.status not in {"paused", "failed"}:
                raise ValueError("Manual compaction requires a paused or failed run")
            if record.namespace != session.namespace:
                raise ServiceConflictError("The host changed the checkpoint namespace")
            owner = _owner_guard(service, record)
            scope = session.scope(
                thread_id, run_id=receipt.run_id, account_name=record.account_name
            )
        try:
            with owner, session.context(scope):
                checkpoint = service._recovery_checkpoint(session, receipt)
                if checkpoint.get("status") != receipt.status:
                    raise ServiceRecoveryRequiredError(
                        "Admission and checkpoint statuses differ; "
                        "reconcile before compaction"
                    )
                return await session.agent.acompact(scope=scope)
        except CompactionBusyError as error:
            raise ServiceBusyError(str(error)) from error
        except (CompactionRecoveryRequiredError, TaskPauseRequestedError) as error:
            raise ServiceRecoveryRequiredError(str(error)) from error
        except CheckpointConflictError as error:
            raise ServiceConflictError(str(error)) from error


def _owner_guard(service, record):
    if record.owner_id == service._owner_id:
        return nullcontext()
    if service._owner_lease is None:
        raise ServiceRecoveryRequiredError(
            "Establish old-worker quiescence before compaction"
        )
    return service._owner_lease.reclaim(record.owner_id)
