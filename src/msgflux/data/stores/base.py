from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Literal, Mapping
from uuid import uuid4

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
    """Snapshots and append-only events keyed by namespace, thread and run."""

    supports_atomic_commit = False

    def read_commits(self, namespace, thread_id, run_id, *, after=None, limit=100):
        """Atomically read a snapshot or bounded durable transitions after a cursor."""
        raise NotImplementedError("This provider has no durable observation support")

    async def aread_commits(
        self, namespace, thread_id, run_id, *, after=None, limit=100
    ):
        return await asyncio.to_thread(
            self.read_commits, namespace, thread_id, run_id, after=after, limit=limit
        )

    @staticmethod
    def _validate_checkpoint_envelope(checkpoint: Any) -> None:
        """Validate metadata while retaining compatibility with legacy states."""
        if checkpoint is None:
            return
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Checkpoint envelope must be a mapping")
        schema_version = checkpoint.get("schema_version", 1)
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ValueError("Checkpoint schema_version must be an integer")
        if schema_version != 1:
            raise ValueError(
                f"Unsupported checkpoint schema version `{schema_version}`"
            )
        revision = checkpoint.get("revision", 0)
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise ValueError("Checkpoint revision must be a non-negative integer")
        stream_id = checkpoint.get("stream_id")
        if stream_id is not None and (not isinstance(stream_id, str) or not stream_id):
            raise ValueError("Checkpoint stream_id must be a non-empty string or null")
        branch_id = checkpoint.get("branch_id")
        if branch_id is not None and (not isinstance(branch_id, str) or not branch_id):
            raise ValueError("Checkpoint branch_id must be a non-empty string or null")
        head_item_id = checkpoint.get("head_item_id")
        if head_item_id is not None and (
            not isinstance(head_item_id, str) or not head_item_id
        ):
            raise ValueError(
                "Checkpoint head_item_id must be a non-empty string or null"
            )
        extensions = checkpoint.get("extensions", {})
        if not isinstance(extensions, Mapping):
            raise ValueError("Checkpoint extensions must be a mapping")

    @classmethod
    def _prepare_revision_state(
        cls,
        state,
        current,
        *,
        expected_revision,
        branch_id,
        head_item_id,
        extension_state,
    ) -> tuple[dict[str, Any], int]:
        """Prepare and validate a new snapshot without publishing any writes."""
        checkpoint = current.get("_checkpoint", {})
        cls._validate_checkpoint_envelope(checkpoint)
        cls._validate_checkpoint_envelope(state.get("_checkpoint"))
        revision = checkpoint.get("revision", 0)
        if expected_revision is not None:
            if (
                isinstance(expected_revision, bool)
                or not isinstance(expected_revision, int)
                or expected_revision < 0
            ):
                raise ValueError("expected_revision must be a non-negative integer")
            if expected_revision != revision:
                raise CheckpointConflictError(
                    f"Checkpoint revision conflict: expected {expected_revision}, "
                    f"found {revision}."
                )
        committed = deepcopy(dict(state))
        runtime = committed.get("runtime", {})
        if not isinstance(runtime, Mapping):
            raise ValueError("Checkpoint runtime must be a mapping")
        envelope = {
            **deepcopy(dict(checkpoint)),
            "schema_version": 1,
            "revision": revision + 1,
            "stream_id": checkpoint.get("stream_id") or uuid4().hex,
            "branch_id": branch_id
            if branch_id is not None
            else runtime.get("branch_id", checkpoint.get("branch_id")),
            "head_item_id": head_item_id
            if head_item_id is not None
            else runtime.get("head_item_id", checkpoint.get("head_item_id")),
            "extensions": deepcopy(
                extension_state
                if extension_state is not None
                else runtime.get("extensions", checkpoint.get("extensions", {}))
            ),
        }
        cls._validate_checkpoint_envelope(envelope)
        committed["_checkpoint"] = envelope
        if "runtime" in committed:
            committed["runtime"] = {
                **runtime,
                "revision": envelope["revision"],
                "branch_id": envelope["branch_id"],
                "head_item_id": envelope["head_item_id"],
                "extensions": deepcopy(envelope["extensions"]),
            }
        return committed, revision + 1

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
        self._set_fork_metadata(
            forked,
            namespace=namespace,
            source_thread_id=source_thread_id,
            source_run_id=source_run_id,
            target_thread_id=target_thread_id,
            target_run_id=target_run_id,
            at_item_id=at_item_id,
        )
        self.save_state(namespace, target_thread_id, target_run_id, forked)
        loaded = self.load_state(namespace, target_thread_id, target_run_id)
        if loaded is None:
            raise ValueError(
                f"Forked checkpoint `{target_run_id}` could not be loaded."
            )
        return loaded

    @classmethod
    def _set_fork_metadata(
        cls,
        state,
        *,
        namespace,
        source_thread_id,
        source_run_id,
        target_thread_id,
        target_run_id,
        at_item_id,
    ) -> None:
        source = state.get("_checkpoint", {})
        cls._validate_checkpoint_envelope(source)
        messages = state.get("messages", {})
        items = messages.get("items", [])
        scopes = (
            messages.get("metadata", {}).get("runtime", {}).get("context_scopes", {})
        )
        branch = scopes.get("active", "root")
        head = items[-1].get("item_id") if items else None
        state["_checkpoint"] = {
            "schema_version": 1,
            "revision": 0,
            "branch_id": branch,
            "head_item_id": head,
            "extensions": source.get("extensions", {}),
            "fork_of": {
                "namespace": namespace,
                "thread_id": source_thread_id,
                "run_id": source_run_id,
                "item_id": at_item_id,
                "branch_id": source.get("branch_id"),
                "head_item_id": source.get("head_item_id"),
            },
        }
        for key in ("runtime", "scope"):
            if isinstance(state.get(key), Mapping):
                value = dict(state[key])
                value.update(
                    namespace=namespace,
                    thread_id=target_thread_id,
                    run_id=target_run_id,
                    parent_run_id=source_run_id,
                    root_run_id=value.get("root_run_id") or source_run_id,
                )
                if key == "runtime":
                    value.update(revision=0, branch_id=branch, head_item_id=head)
                state[key] = value

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
