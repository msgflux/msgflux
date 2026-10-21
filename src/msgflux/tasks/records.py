from __future__ import annotations

import msgspec


class TaskSummary(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Allowlisted task fields for lightweight inspection."""

    task_id: str
    tool_name: str
    status: str
    updated_at: str
    error: str | None = None
