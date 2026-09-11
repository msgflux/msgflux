from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from msgflux.tasks.dataclasses import TaskActivity

if TYPE_CHECKING:
    from msgflux.tasks.store import TaskStore


class TaskActivityRecorder:
    """Small recorder for compact task activity entries."""

    def __init__(self, task_id: str, store: TaskStore):
        self.task_id = task_id
        self._store = store

    # --- Activity Publishing ---

    def add(
        self,
        *,
        kind: str,
        summary: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> TaskActivity | None:
        return self._store.add_activity(
            self.task_id,
            kind=kind,
            summary=summary,
            metadata=metadata,
        )

    def tool_call(self, tool_name: str, parameters: Any) -> TaskActivity | None:
        summary = f"{tool_name}({self._truncate(str(parameters))})"
        return self.add(kind="tool_call", summary=summary, metadata={"tool": tool_name})

    @staticmethod
    def _truncate(value: str, *, limit: int = 140) -> str:
        text = " ".join(value.split())
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."
