from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict
from uuid import uuid4

from msgflux.utils.time import utc_now_isoformat


@dataclass
class AgentNotification:
    notification_id: str
    source: str
    ref: str | None = None
    status: str | None = None
    hint: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dedupe_key: str | None = None
    created_at: str = field(default_factory=utc_now_isoformat)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentControlMessage:
    """Runtime control signal delivered through an AgentInbox."""

    command: str
    reason: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    control_id: str = field(default_factory=lambda: uuid4().hex[:8])
    created_at: str = field(default_factory=utc_now_isoformat)

    def to_notification(self) -> AgentNotification:
        metadata = deepcopy(self.metadata)
        if self.reason:
            metadata["reason"] = self.reason
        return AgentNotification(
            notification_id=self.control_id,
            source="control",
            status=self.command,
            hint=self.reason,
            metadata=metadata,
            dedupe_key=f"control:{self.command}",
            created_at=self.created_at,
        )
