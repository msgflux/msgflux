from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Mapping

_TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted"})


class CheckpointStore(ABC):
    """Unified store for agent and pipeline checkpoints.

    The key is always `(namespace, thread_id, run_id)`. State snapshots use
    UPSERT semantics while events remain append-only.
    """

    @abstractmethod
    def save_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> Mapping[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def append_event(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        event: Mapping[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_events(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        raise NotImplementedError

    def save_with_event(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        self.save_state(namespace, thread_id, run_id, state)
        self.append_event(namespace, thread_id, run_id, event)

    @abstractmethod
    def list_runs(
        self,
        namespace: str,
        thread_id: str,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> List[Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def delete_run(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> bool:
        raise NotImplementedError

    def load_latest_run(
        self,
        namespace: str,
        thread_id: str,
    ) -> Mapping[str, Any] | None:
        runs = self.list_runs(namespace, thread_id, limit=1)
        if not runs:
            return None
        return self.load_state(namespace, thread_id, runs[0]["run_id"])

    def fork_run(
        self,
        namespace: str,
        source_thread_id: str,
        source_run_id: str,
        *,
        target_thread_id: str,
        target_run_id: str,
        status: str | None = None,
    ) -> Mapping[str, Any]:
        state = self.load_state(namespace, source_thread_id, source_run_id)
        if state is None:
            raise ValueError(
                f"Checkpoint run `{source_run_id}` not found in thread "
                f"`{source_thread_id}`."
            )
        forked = dict(state)
        if status is not None:
            forked["status"] = status
        self.save_state(namespace, target_thread_id, target_run_id, forked)
        loaded = self.load_state(namespace, target_thread_id, target_run_id)
        if loaded is None:
            raise ValueError(
                f"Forked checkpoint `{target_run_id}` could not be loaded."
            )
        return loaded

    def find_incomplete_runs(
        self,
        namespace: str,
        thread_id: str,
    ) -> List[Mapping[str, Any]]:
        all_runs = self.list_runs(namespace, thread_id)
        return [r for r in all_runs if r.get("status") not in _TERMINAL_STATUSES]

    @abstractmethod
    def clear(
        self,
        namespace: str | None = None,
        thread_id: str | None = None,
        *,
        older_than: float | None = None,
    ) -> int:
        raise NotImplementedError


class AsyncCheckpointStore(ABC):
    """Async mirror of :class:`CheckpointStore`."""

    @abstractmethod
    async def asave_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def aload_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> Mapping[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    async def aappend_event(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        event: Mapping[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def aload_events(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        raise NotImplementedError

    async def asave_with_event(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        await self.asave_state(namespace, thread_id, run_id, state)
        await self.aappend_event(namespace, thread_id, run_id, event)

    @abstractmethod
    async def alist_runs(
        self,
        namespace: str,
        thread_id: str,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> List[Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    async def adelete_run(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> bool:
        raise NotImplementedError

    async def aload_latest_run(
        self,
        namespace: str,
        thread_id: str,
    ) -> Mapping[str, Any] | None:
        runs = await self.alist_runs(namespace, thread_id, limit=1)
        if not runs:
            return None
        return await self.aload_state(namespace, thread_id, runs[0]["run_id"])

    async def afind_incomplete_runs(
        self,
        namespace: str,
        thread_id: str,
    ) -> List[Mapping[str, Any]]:
        all_runs = await self.alist_runs(namespace, thread_id)
        return [r for r in all_runs if r.get("status") not in _TERMINAL_STATUSES]

    @abstractmethod
    async def aclear(
        self,
        namespace: str | None = None,
        thread_id: str | None = None,
        *,
        older_than: float | None = None,
    ) -> int:
        raise NotImplementedError
