"""Results and ownership failures for durable manual compaction."""

from typing import Any, Literal

import msgspec


class CompactionResult(msgspec.Struct, frozen=True):
    thread_id: str
    run_id: str
    status: Literal["completed", "skipped"]
    reason: str
    operation_id: str | None = None
    usage: dict[str, Any] | None = None


class CompactionBusyError(RuntimeError):
    """The thread is still executing or another maintenance operation owns it."""


class CompactionRecoveryRequiredError(RuntimeError):
    """The saved execution cannot be safely compacted without reconciliation."""
