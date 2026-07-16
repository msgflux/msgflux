from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Mapping

from msgflux.data.stores.types import AgentInboxStoreType


class AgentInboxStore(ABC, AgentInboxStoreType):
    """Persistent storage boundary for pending agent inbox notifications."""

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
