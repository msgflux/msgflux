from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Mapping
from uuid import uuid4

from msgflux.runtime.agent_inbox.dataclasses import AgentNotification
from msgflux.utils.time import utc_now_isoformat

if TYPE_CHECKING:
    from msgflux.runtime.agent_inbox.inbox import AgentInbox


class ToolNotificationHandle:
    """Controlled notification publisher injected into tools."""

    def __init__(
        self,
        agent_inbox: AgentInbox | None,
        *,
        source: str = "tool_status",
        ref: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        self._agent_inbox = agent_inbox
        self._source = source
        self._ref = ref
        self._metadata = deepcopy(dict(metadata or {}))

    # --- Notification Publishing ---

    def publish(
        self,
        *,
        status: str,
        hint: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        dedupe_key: str | None = None,
        source: str | None = None,
        ref: str | None = None,
    ) -> AgentNotification | None:
        if self._agent_inbox is None:
            return None

        payload = deepcopy(self._metadata)
        if metadata:
            payload.update(dict(metadata))

        return self._agent_inbox.publish(
            AgentNotification(
                notification_id=uuid4().hex[:8],
                source=source or self._source,
                ref=self._ref if ref is None else ref,
                status=status,
                hint=hint,
                metadata=payload,
                dedupe_key=dedupe_key,
                created_at=utc_now_isoformat(),
            )
        )

    def notify(
        self,
        *,
        status: str,
        hint: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        dedupe_key: str | None = None,
        source: str | None = None,
        ref: str | None = None,
    ) -> AgentNotification | None:
        return self.publish(
            status=status,
            hint=hint,
            metadata=metadata,
            dedupe_key=dedupe_key,
            source=source,
            ref=ref,
        )

    def update(
        self,
        status: str,
        *,
        hint: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        dedupe_key: str | None = None,
        source: str | None = None,
        ref: str | None = None,
    ) -> AgentNotification | None:
        return self.publish(
            status=status,
            hint=hint,
            metadata=metadata,
            dedupe_key=dedupe_key,
            source=source,
            ref=ref,
        )
