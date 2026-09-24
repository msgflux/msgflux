"""Fencing token for one active background worker."""

import msgspec


class TaskLease(msgspec.Struct, frozen=True):
    task_id: str
    owner_id: str
    expires_at: float
