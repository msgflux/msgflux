from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Mapping

from msgflux._private.store_routing import process_routing_id
from msgflux.data.stores.types import AgentInboxStoreType


class AgentInboxStore(ABC, AgentInboxStoreType):
    """Persistent storage boundary for pending agent inbox notifications."""

    @property
    def routing_id(self) -> str:
        """Identify one process-local store instance unless a provider overrides it."""
        return process_routing_id(self)

    @abstractmethod
    def load_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> List[Mapping[str, object]]:
        raise NotImplementedError

    @abstractmethod
    def save_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        notifications: Iterable[Mapping[str, object]],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear(
        self,
        namespace: str | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
        *,
        older_than: float | None = None,
    ) -> int:
        raise NotImplementedError

    def publish_notification(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        notification: Mapping[str, object],
    ) -> Mapping[str, object]:
        existing = self.load_notifications(namespace, thread_id, run_id)
        dedupe_key = notification.get("dedupe_key")
        if dedupe_key:
            for index, item in enumerate(existing):
                if item.get("dedupe_key") == dedupe_key:
                    existing[index] = notification
                    self.save_notifications(namespace, thread_id, run_id, existing)
                    return notification
        self.save_notifications(namespace, thread_id, run_id, [*existing, notification])
        return notification

    def claim_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        *,
        lease_id: str,
        lease_seconds: float,
        limit: int | None = None,
    ) -> List[Mapping[str, object]]:
        """Atomically claim pending notifications for one consumer."""
        del lease_id, lease_seconds, limit
        return self.load_notifications(namespace, thread_id, run_id)

    def ack_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        notification_ids: Iterable[str],
        *,
        lease_id: str | None = None,
    ) -> None:
        """Acknowledge notifications after their delivery is durable."""
        del lease_id
        ids = set(notification_ids)
        if not ids:
            return
        remaining = [
            item
            for item in self.load_notifications(namespace, thread_id, run_id)
            if item.get("notification_id") not in ids
        ]
        self.save_notifications(namespace, thread_id, run_id, remaining)

    def release_notifications(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        *,
        lease_id: str,
        notification_ids: Iterable[str] | None = None,
    ) -> None:
        """Release a claim so another consumer can retry it."""
        del namespace, thread_id, run_id, lease_id, notification_ids

    def move_notifications(
        self,
        previous: tuple[str, str, str],
        current: tuple[str, str, str],
    ) -> None:
        """Move pending notifications between scopes atomically when possible."""
        if previous == current:
            return
        notifications = self.load_notifications(*previous)
        if not notifications:
            return
        existing = self.load_notifications(*current)
        self.save_notifications(*current, [*notifications, *existing])
        self.clear(*previous)
