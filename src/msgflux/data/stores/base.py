from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Literal, Mapping

_TERMINAL_STATUSES = frozenset({"completed", "failed", "interrupted"})


class CheckpointConflictError(RuntimeError):
    """Raised when a checkpoint writer uses a stale revision."""


@dataclass(frozen=True)
class CheckpointCommit:
    """Result of one revision-checked state and event commit."""

    revision: int
    state: Mapping[str, Any]
    branch_id: str | None = None
    head_item_id: str | None = None


class CheckpointStore(ABC):
    supports_atomic_commit = False
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

    def commit_state(self, *args: Any, **kwargs: Any) -> CheckpointCommit:
        """Commit state atomically; providers must implement this capability."""
        raise NotImplementedError(
            "This checkpoint provider does not support atomic revision commits"
        )

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
        at_item_id: str | None = None,
        position: Literal["before", "at"] = "at",
    ) -> Mapping[str, Any]:
        """Copy a run, optionally ending at an exact safe timeline item."""
        state = self.load_state(namespace, source_thread_id, source_run_id)
        if state is None:
            raise ValueError(
                f"Checkpoint run `{source_run_id}` not found in thread "
                f"`{source_thread_id}`."
            )
        forked = self._prepare_fork_state(
            state,
            at_item_id=at_item_id,
            position=position,
        )
        messages = forked.get("messages")
        if isinstance(messages, dict):
            messages["thread_id"] = target_thread_id
        if status is not None:
            forked["status"] = status
        checkpoint = forked.get("_checkpoint")
        if isinstance(checkpoint, Mapping):
            forked["_checkpoint"] = {
                **dict(checkpoint),
                "revision": 0,
                "branch_id": "root",
                "head_item_id": None,
                "fork_of": {
                    "namespace": namespace,
                    "thread_id": source_thread_id,
                    "run_id": source_run_id,
                    "item_id": at_item_id,
                },
            }
        self.save_state(namespace, target_thread_id, target_run_id, forked)
        loaded = self.load_state(namespace, target_thread_id, target_run_id)
        if loaded is None:
            raise ValueError(
                f"Forked checkpoint `{target_run_id}` could not be loaded."
            )
        return loaded

    @classmethod
    def _prepare_fork_state(
        cls,
        state: Mapping[str, Any],
        *,
        at_item_id: str | None,
        position: Literal["before", "at"],
    ) -> dict[str, Any]:
        """Copy a state and optionally truncate it at a safe timeline boundary."""
        if position not in {"before", "at"}:
            raise ValueError("`position` must be either 'before' or 'at'.")

        forked = deepcopy(dict(state))
        if at_item_id is None:
            return forked
        if not isinstance(at_item_id, str) or not at_item_id:
            raise ValueError("`at_item_id` must be a non-empty string.")

        messages = forked.get("messages")
        if not isinstance(messages, Mapping):
            raise ValueError("Checkpoint does not contain a ChatMessages timeline.")
        items = messages.get("items")
        if not isinstance(items, list):
            raise ValueError("Checkpoint ChatMessages timeline is corrupted.")

        matches = [
            index
            for index, item in enumerate(items)
            if isinstance(item, Mapping) and item.get("item_id") == at_item_id
        ]
        if not matches:
            raise ValueError(f"Checkpoint item `{at_item_id}` was not found.")
        if len(matches) > 1:
            raise ValueError(f"Checkpoint item_id `{at_item_id}` is not unique.")

        end_index = matches[0] + (1 if position == "at" else 0)
        selected_items = deepcopy(items[:end_index])
        cls._validate_fork_boundary(selected_items)

        forked_messages = deepcopy(dict(messages))
        forked_messages["items"] = selected_items
        forked["messages"] = forked_messages
        return forked

    @staticmethod
    def _validate_fork_boundary(items: List[Any]) -> None:
        active_turns: set[str] = set()
        open_calls: set[str] = set()

        for item in items:
            if not isinstance(item, Mapping):
                continue
            CheckpointStore._track_turn_boundary(item, active_turns)
            CheckpointStore._track_canonical_call_boundary(item, open_calls)
            CheckpointStore._track_chatml_call_boundary(item, open_calls)

        if active_turns:
            raise ValueError(
                "Cannot fork inside an active turn; choose a boundary before its "
                "start or at its terminal turn event."
            )
        if open_calls:
            raise ValueError(
                "Cannot fork between a tool call and its output; choose a boundary "
                "that keeps the pair together."
            )

    @staticmethod
    def _track_turn_boundary(item: Mapping[str, Any], active_turns: set[str]) -> None:
        if item.get("type") != "turn":
            return
        turn_id = item.get("turn_id")
        if not isinstance(turn_id, str):
            return
        event = item.get("event")
        if event in {"start", "resume"}:
            active_turns.add(turn_id)
        elif event in {"pause", "complete", "fail", "interrupt"}:
            active_turns.discard(turn_id)

    @staticmethod
    def _track_canonical_call_boundary(
        item: Mapping[str, Any], open_calls: set[str]
    ) -> None:
        item_type = item.get("type")
        if not isinstance(item_type, str):
            return
        if item_type.endswith("_output"):
            call_id = item.get("call_id") or item.get("tool_search_call_id")
            if isinstance(call_id, str):
                open_calls.discard(call_id)
        elif item_type.endswith("_call"):
            call_id = item.get("call_id") or item.get("id")
            if isinstance(call_id, str) and call_id:
                open_calls.add(call_id)

    @staticmethod
    def _track_chatml_call_boundary(
        item: Mapping[str, Any], open_calls: set[str]
    ) -> None:
        if item.get("role") == "tool":
            call_id = item.get("tool_call_id")
            if isinstance(call_id, str):
                open_calls.discard(call_id)
            return
        if item.get("role") != "assistant":
            return
        tool_calls = item.get("tool_calls")
        if not isinstance(tool_calls, list):
            return
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                continue
            call_id = tool_call.get("id")
            if isinstance(call_id, str) and call_id:
                open_calls.add(call_id)

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

    async def acommit_state(self, *args: Any, **kwargs: Any) -> CheckpointCommit:
        raise NotImplementedError(
            "This checkpoint provider does not support atomic revision commits"
        )

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
