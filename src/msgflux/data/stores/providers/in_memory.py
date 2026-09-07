# Checkpoints are split into small run records and immutable message items.
# A run belongs to `(namespace, thread_id, run_id)` and stores the non-message
# state plus the ChatMessages container metadata and an ordered list of item
# references. The message items themselves are stored once per thread by a
# content hash, so continuing a conversation with a new run_id only writes a new
# run record that points at the existing items. Loading reverses that shape and
# returns the public state with `messages.items` restored. Forking to another
# thread copies only the referenced items needed by the fork; deleting a run
# removes message items when no remaining run in that thread references them.
#
# In memory this is represented by two nested dictionaries:
# `_data[namespace][thread_id][run_id]` keeps the normalized run state and
# events, while `_message_items[namespace][thread_id][item_ref]` keeps each
# frozen ChatMessages item payload encoded as msgpack bytes. The normalized run
# state replaces
# `messages.items` with `_messages = {"state": <messages without items>,
# "item_entries": [...]}`. Entries preserve ordering and keep the sparse
# per-occurrence ``item_id`` outside the deduplicated item payload.

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from threading import RLock
from typing import Any, Dict, List, Literal, Mapping

import msgspec

from msgflux._private.chat_items import (
    restore_item_occurrence,
    split_item_occurrence,
)
from msgflux.data.stores.base import (
    CheckpointCommit,
    CheckpointConflictError,
    CheckpointStore,
)
from msgflux.data.stores.registry import register_store
from msgflux.data.stores.types import CheckpointStoreType


@register_store()
class InMemoryCheckpointStore(CheckpointStore, CheckpointStoreType):
    """In-memory checkpoint store for tests and local prototyping."""

    provider = "in_memory"
    supports_atomic_commit = True

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        self._message_items: Dict[str, Dict[str, Dict[str, bytes]]] = {}
        self._lock = RLock()

    def _get_run(
        self, namespace: str, thread_id: str, run_id: str
    ) -> Dict[str, Any] | None:
        return self._data.get(namespace, {}).get(thread_id, {}).get(run_id)

    def _ensure_run(
        self, namespace: str, thread_id: str, run_id: str
    ) -> Dict[str, Any]:
        ns = self._data.setdefault(namespace, {})
        thread = ns.setdefault(thread_id, {})
        run = thread.get(run_id)
        if run is None:
            run = {"state": {}, "events": [], "updated_at": time.time()}
            thread[run_id] = run
        return run

    def _normalize_state(
        self,
        namespace: str,
        thread_id: str,
        state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        normalized = deepcopy(dict(state))
        messages = normalized.pop("messages", None)
        if not isinstance(messages, Mapping):
            return normalized

        items = messages.get("items")
        if not isinstance(items, list):
            return normalized

        item_store = self._message_items.setdefault(namespace, {}).setdefault(
            thread_id, {}
        )
        item_entries = []
        item_ids: set[str] = set()
        for index, item in enumerate(items):
            payload, entry = split_item_occurrence(
                namespace=namespace,
                thread_id=thread_id,
                index=index,
                item=item,
            )
            item_id = entry["item_id"]
            if item_id in item_ids:
                raise ValueError(f"Duplicate ChatMessages item_id `{item_id}`.")
            item_ids.add(item_id)
            item_ref = self._item_ref(payload)
            item_store.setdefault(item_ref, msgspec.msgpack.encode(payload))
            entry["item_ref"] = item_ref
            item_entries.append(entry)

        message_state = deepcopy(dict(messages))
        message_state.pop("items", None)
        normalized["_messages"] = {
            "state": message_state,
            "item_entries": item_entries,
        }
        return normalized

    def _denormalize_state(
        self,
        namespace: str,
        thread_id: str,
        state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        restored = deepcopy(dict(state))
        normalized_messages = restored.pop("_messages", None)
        if not isinstance(normalized_messages, Mapping):
            return restored

        message_state = normalized_messages.get("state")
        item_entries = normalized_messages.get("item_entries")
        if not isinstance(message_state, Mapping) or not isinstance(item_entries, list):
            return restored

        item_store = self._message_items.get(namespace, {}).get(thread_id, {})
        messages = deepcopy(dict(message_state))
        messages["items"] = []
        for entry in item_entries:
            if not isinstance(entry, Mapping):
                raise ValueError("Checkpoint message entry is corrupted")
            item_ref = entry.get("item_ref")
            if not isinstance(item_ref, str) or item_ref not in item_store:
                raise ValueError(
                    "Checkpoint message item is missing or corrupted: "
                    f"{namespace}/{thread_id}/{item_ref!r}"
                )
            item = restore_item_occurrence(
                msgspec.msgpack.decode(item_store[item_ref]),
                entry,
            )
            messages["items"].append(item)
        restored["messages"] = messages
        return restored

    @staticmethod
    def _item_ref(item: Any) -> str:
        payload = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        return "item_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _collect_live_item_refs(self, namespace: str, thread_id: str) -> set[str]:
        live: set[str] = set()
        thread = self._data.get(namespace, {}).get(thread_id, {})
        for run in thread.values():
            live.update(self._collect_refs_from_state(run.get("state", {})))
        return live

    @staticmethod
    def _collect_refs_from_state(state: Any) -> set[str]:
        if not isinstance(state, Mapping):
            return set()
        messages = state.get("_messages")
        if not isinstance(messages, Mapping):
            return set()
        item_entries = messages.get("item_entries")
        if not isinstance(item_entries, list):
            return set()
        return {
            entry["item_ref"]
            for entry in item_entries
            if isinstance(entry, Mapping) and isinstance(entry.get("item_ref"), str)
        }

    def _cleanup_orphaned_items(self, namespace: str, thread_id: str) -> None:
        item_store = self._message_items.get(namespace, {}).get(thread_id)
        if item_store is None:
            return
        live = self._collect_live_item_refs(namespace, thread_id)
        for item_ref in list(item_store):
            if item_ref not in live:
                del item_store[item_ref]
        if not item_store:
            ns_items = self._message_items.get(namespace)
            if ns_items is not None:
                ns_items.pop(thread_id, None)
                if not ns_items:
                    self._message_items.pop(namespace, None)

    def save_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
    ) -> None:
        with self._lock:
            run = self._ensure_run(namespace, thread_id, run_id)
            run["state"] = self._normalize_state(namespace, thread_id, state)
            run["updated_at"] = time.time()

    def load_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> Mapping[str, Any] | None:
        with self._lock:
            run = self._get_run(namespace, thread_id, run_id)
            if run is None:
                return None
            return self._denormalize_state(namespace, thread_id, run["state"])

    def append_event(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        event: Mapping[str, Any],
    ) -> None:
        with self._lock:
            run = self._ensure_run(namespace, thread_id, run_id)
            run["events"].append(deepcopy(dict(event)))

    def load_events(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> List[Mapping[str, Any]]:
        with self._lock:
            run = self._get_run(namespace, thread_id, run_id)
            if run is None:
                return []
            return deepcopy(run["events"])

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
        with self._lock:
            source = self._get_run(namespace, source_thread_id, source_run_id)
            if source is None:
                raise ValueError(
                    f"Checkpoint run `{source_run_id}` not found in thread "
                    f"`{source_thread_id}`."
                )
            source_state = self._denormalize_state(
                namespace,
                source_thread_id,
                source["state"],
            )
            forked = self._prepare_fork_state(
                source_state,
                at_item_id=at_item_id,
                position=position,
            )
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
                        "thread_id": source_thread_id,
                        "run_id": source_run_id,
                        "item_id": at_item_id,
                    },
                }

            messages = forked.get("messages")
            if isinstance(messages, dict):
                messages["thread_id"] = target_thread_id

            target = self._ensure_run(namespace, target_thread_id, target_run_id)
            target["state"] = self._normalize_state(
                namespace,
                target_thread_id,
                forked,
            )
            target["updated_at"] = time.time()
            self._cleanup_orphaned_items(namespace, target_thread_id)
            return self._denormalize_state(namespace, target_thread_id, target["state"])

    def save_with_event(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
        event: Mapping[str, Any],
    ) -> None:
        with self._lock:
            run = self._ensure_run(namespace, thread_id, run_id)
            run["state"] = self._normalize_state(namespace, thread_id, state)
            run["updated_at"] = time.time()
            run["events"].append(deepcopy(dict(event)))

    def commit_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
        event: Mapping[str, Any] | None = None,
        branch_id: str | None = None,
        head_item_id: str | None = None,
        extension_state: Mapping[str, Any] | None = None,
    ) -> CheckpointCommit:
        with self._lock:
            run = self._get_run(namespace, thread_id, run_id)
            current = (
                self._denormalize_state(namespace, thread_id, run["state"])
                if run is not None
                else {}
            )
            checkpoint = current.get("_checkpoint", {})
            revision = checkpoint.get("revision", 0)
            if expected_revision is not None and revision != expected_revision:
                raise CheckpointConflictError(
                    f"Checkpoint revision conflict: expected {expected_revision}, "
                    f"found {revision}."
                )
            next_revision = revision + 1
            if run is None:
                run = self._ensure_run(namespace, thread_id, run_id)
            committed = deepcopy(dict(state))
            committed["_checkpoint"] = {
                "schema_version": 1,
                "revision": next_revision,
                "branch_id": branch_id
                if branch_id is not None
                else checkpoint.get("branch_id"),
                "head_item_id": head_item_id
                if head_item_id is not None
                else checkpoint.get("head_item_id"),
                "extensions": deepcopy(dict(extension_state or {})),
            }
            run["state"] = self._normalize_state(namespace, thread_id, committed)
            run["updated_at"] = time.time()
            if event is not None:
                run["events"].append(deepcopy(dict(event)))
            return CheckpointCommit(
                next_revision,
                committed,
                committed["_checkpoint"]["branch_id"],
                committed["_checkpoint"]["head_item_id"],
            )

    def list_runs(
        self,
        namespace: str,
        thread_id: str,
        *,
        status: str | None = None,
        limit: int | None = None,
    ) -> List[Mapping[str, Any]]:
        with self._lock:
            thread_runs = self._data.get(namespace, {}).get(thread_id, {})
            entries: List[Dict[str, Any]] = []
            for run_id, run in thread_runs.items():
                run_status = run["state"].get("status")
                if status is not None and run_status != status:
                    continue
                entries.append(
                    {
                        "run_id": run_id,
                        "status": run_status,
                        "updated_at": run["updated_at"],
                    }
                )
            entries.sort(key=lambda e: e["updated_at"], reverse=True)
            if limit is not None:
                entries = entries[:limit]
            return entries

    def delete_run(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
    ) -> bool:
        with self._lock:
            thread_runs = self._data.get(namespace, {}).get(thread_id, {})
            if run_id in thread_runs:
                del thread_runs[run_id]
                self._cleanup_orphaned_items(namespace, thread_id)
                return True
            return False

    def clear(
        self,
        namespace: str | None = None,
        thread_id: str | None = None,
        *,
        older_than: float | None = None,
    ) -> int:
        cutoff = time.time() - older_than if older_than is not None else None
        removed = 0
        with self._lock:
            namespaces = (
                [namespace] if namespace is not None else list(self._data.keys())
            )
            for ns in namespaces:
                ns_data = self._data.get(ns)
                if ns_data is None:
                    continue
                threads = [thread_id] if thread_id is not None else list(ns_data.keys())
                for sid in threads:
                    thread = ns_data.get(sid)
                    if thread is None:
                        continue
                    to_delete = []
                    for rid, run in thread.items():
                        if cutoff is not None and run["updated_at"] >= cutoff:
                            continue
                        to_delete.append(rid)
                    for rid in to_delete:
                        del thread[rid]
                        removed += 1
                    self._cleanup_orphaned_items(ns, sid)
                    if not thread:
                        del ns_data[sid]
                if not ns_data:
                    del self._data[ns]
        return removed
