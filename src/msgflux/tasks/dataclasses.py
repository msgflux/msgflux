from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict


@dataclass
class TaskProgress:
    stage: str | None = None
    message: str | None = None
    current: int | None = None
    total: int | None = None
    percent: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskActivity:
    task_id: str
    kind: str
    summary: str
    created_at: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TaskRecord:
    task_id: str
    tool_name: str
    status: str
    created_at: str
    updated_at: str
    completed_at: str | None = None
    result: Any | None = None
    error: str | None = None
    progress: TaskProgress = field(default_factory=TaskProgress)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
