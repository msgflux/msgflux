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
# In memory this is represented by three nested dictionaries:
# `_data[namespace][thread_id][run_id]` keeps the normalized run state and
# events, `_message_items[namespace][thread_id][item_ref]` keeps each frozen
# ChatMessages item payload encoded as msgpack bytes, and
# `_manifest_pages[namespace][thread_id][segment_ref]` keeps immutable
# content-addressed manifest pages. The normalized run state replaces
# `messages.items` with `_messages = {"state": <messages without items>, ...}`.
# Manifests up to `MANIFEST_PAGE_SIZE` entries keep their ordered list inline as
# `item_entries`; larger manifests store only a
# `manifest = {"format": "segments_v1", "head": <page ref>, "count": <N>}`
# descriptor whose linked pages hold the occurrences.
#
# Replacing an existing run's segmented head preflights the post-replacement
# survivor refs (including staged candidate pages) before publishing, then
# prunes unreachable pages/items. New runs and same-head updates skip that GC,
# and pages referenced by other retained runs remain live.

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from itertools import islice
from threading import RLock
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping
from uuid import uuid4

import msgspec

from msgflux._private.chat_items import (
    legacy_item_id,
    restore_item_occurrence,
    split_item_occurrence,
)
from msgflux.data.stores._manifest import (
    MANIFEST_PAGE_SIZE,
    append_manifest,
    build_manifest,
    iter_manifest_entries,
    iter_manifest_pages,
    replacement_needs_segment_gc,
    resolve_manifest_layout,
)
from msgflux.data.stores.base import (
    CheckpointCommit,
    CheckpointConflictError,
    CheckpointStore,
)
from msgflux.data.stores.incremental import (
    CheckpointBase,
    CheckpointTail,
    append_control_state,
    compose_append_state,
    migrate_inline_entries,
    optional_snapshot_base,
    recover_tail,
    same_payload_value,
    snapshot_base,
    validate_tail_limit,
)
from msgflux.data.stores.observation import make_page, validate_read
from msgflux.data.stores.registry import register_store
from msgflux.data.stores.types import CheckpointStoreType


@register_store()
class InMemoryCheckpointStore(CheckpointStore, CheckpointStoreType):
    """In-memory checkpoint store for tests and local prototyping."""

    provider = "in_memory"
    supports_atomic_commit = True
    supports_incremental = True

    @property
    def supports_agent_incremental(self) -> bool:
        return self._has_default_agent_commit(
            InMemoryCheckpointStore._agent_commit_implementation
        )

    def _stage_manifest_page(
        self,
        namespace: str,
        thread_id: str,
        segment_ref: str,
        data: bytes,
        page_store: Dict[str, bytes],
    ) -> None:
        """Stage one page and reject reuse of differing stored or staged bytes."""
        existing = (
            self._manifest_pages.get(namespace, {}).get(thread_id, {}).get(segment_ref)
        )
        if existing is not None and existing != data:
            raise ValueError(f"Checkpoint manifest page is corrupted: {segment_ref!r}")
        staged = page_store.get(segment_ref)
        if staged is not None and staged != data:
            raise ValueError(
                f"Checkpoint manifest page changed while staging: {segment_ref!r}"
            )
        page_store[segment_ref] = data

    def _tail_from_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
        limit: int,
    ) -> CheckpointTail:
        self._validate_checkpoint_envelope(state.get("_checkpoint"))
        normalized_messages = state.get("_messages")
        if not isinstance(normalized_messages, Mapping):
            raise ValueError(
                "Incremental checkpoints require a normalized message history"
            )
        if not isinstance(normalized_messages.get("state"), Mapping):
            raise ValueError("Checkpoint manifest is missing its message state")
        base = snapshot_base(self.routing_id, namespace, thread_id, run_id, state)
        return recover_tail(
            state,
            base,
            limit,
            lambda segment_ref: self._read_manifest_segment(
                namespace, thread_id, segment_ref
            ),
            lambda entries: self._iter_message_items(namespace, thread_id, entries),
        )

    def load_tail(
        self, namespace: str, thread_id: str, run_id: str, *, limit: int = 128
    ) -> CheckpointTail | None:
        validate_tail_limit(limit)
        with self._lock:
            run = self._get_run(namespace, thread_id, run_id)
            if run is None:
                return None
            return self._tail_from_state(
                namespace, thread_id, run_id, run["state"], limit
            )

    def _prepare_append_items(
        self,
        namespace: str,
        thread_id: str,
        items: List[Mapping[str, Any]],
        *,
        start_index: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, bytes]]:
        """Split only new occurrences and reject payload-ref collisions."""
        entries: List[Dict[str, Any]] = []
        payloads: Dict[str, bytes] = {}
        for offset, item in enumerate(items):
            payload, entry = split_item_occurrence(
                namespace=namespace,
                thread_id=thread_id,
                index=start_index + offset,
                item=item,
            )
            item_ref = self._item_ref(payload)
            encoded = msgspec.msgpack.encode(payload)
            staged = payloads.get(item_ref)
            if staged is not None and not same_payload_value(
                msgspec.msgpack.decode(staged), msgspec.msgpack.decode(encoded)
            ):
                raise ValueError(
                    "Checkpoint message item payload reference collision: "
                    f"{namespace}/{thread_id}/{item_ref!r}"
                )
            payloads[item_ref] = encoded
            entry["item_ref"] = item_ref
            entries.append(entry)
        return entries, payloads

    def append_state(  # noqa: C901 - keeps validation, staging and CAS in one lock
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        items: list[Mapping[str, Any]],
        *,
        base: CheckpointBase,
        state_updates: Mapping[str, Any] | None = None,
        message_updates: Mapping[str, Any] | None = None,
        event: Mapping[str, Any] | None = None,
        branch_id: str | None = None,
        head_item_id: str | None = None,
        extension_state: Mapping[str, Any] | None = None,
    ) -> CheckpointTail:
        self._validate_incremental_base(
            base, namespace, thread_id, run_id, self.routing_id
        )
        with self._lock:
            run = self._get_run(namespace, thread_id, run_id)
            if run is None:
                raise CheckpointConflictError(
                    f"Checkpoint run `{run_id}` not found in thread `{thread_id}`."
                )
            previous = run["state"]
            self._validate_checkpoint_envelope(previous.get("_checkpoint"))
            old_base = snapshot_base(
                self.routing_id, namespace, thread_id, run_id, previous
            )
            if old_base != base:
                raise CheckpointConflictError(
                    "Checkpoint base is stale or does not match the persisted snapshot"
                )
            previous_messages = previous.get("_messages")
            if not isinstance(previous_messages, Mapping):
                raise ValueError(
                    "Incremental checkpoints require a normalized message history"
                )
            if not isinstance(previous_messages.get("state"), Mapping):
                raise ValueError("Checkpoint manifest is missing its message state")
            inline, manifest = resolve_manifest_layout(previous_messages)
            control_state = append_control_state(
                previous, items, state_updates, message_updates
            )
            # Freeze the delta before the potentially long strict archive scan.
            # A caller mutating its inputs cannot desynchronize the ack/payloads.
            new_items = control_state["messages"]["items"]

            # Strictly validate every prior payload exactly once and retain only
            # occurrence identities, never the previous timeline.
            old_item_ids: List[str] = []
            prior = self._commit_envelope_from_state(
                namespace,
                thread_id,
                previous,
                collect_item_ids=old_item_ids,
            )
            if len(old_item_ids) != old_base.item_count:
                raise ValueError("Checkpoint manifest occurrence count is corrupted")
            incoming_ids = [item["item_id"] for item in new_items]
            old_id_set = set(old_item_ids)
            if len(old_id_set) != len(old_item_ids):
                raise ValueError("Checkpoint archive contains duplicate item IDs")
            for item_id in incoming_ids:
                if item_id in old_id_set:
                    raise ValueError(f"Duplicate ChatMessages item_id `{item_id}`.")

            effective_head = (
                head_item_id
                if head_item_id is not None
                else (incoming_ids[-1] if incoming_ids else None)
            )
            committed, next_revision = self._prepare_revision_state(
                control_state,
                prior,
                expected_revision=base.revision,
                branch_id=branch_id,
                head_item_id=effective_head,
                extension_state=extension_state,
            )
            prepared_event = deepcopy(dict(event)) if event is not None else None

            new_entries, item_store = self._prepare_append_items(
                namespace,
                thread_id,
                new_items,
                start_index=old_base.item_count,
            )
            for item_ref, data in item_store.items():
                existing = (
                    self._message_items.get(namespace, {})
                    .get(thread_id, {})
                    .get(item_ref)
                )
                if existing is None:
                    continue
                try:
                    stored = msgspec.msgpack.decode(existing)
                    expected = msgspec.msgpack.decode(data)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "Checkpoint message item is corrupted or could not be "
                        f"published: {namespace}/{thread_id}/{item_ref!r}"
                    ) from error
                if not isinstance(stored, Mapping) or not same_payload_value(
                    stored, expected
                ):
                    raise ValueError(
                        "Checkpoint message item is corrupted or could not be "
                        f"published: {namespace}/{thread_id}/{item_ref!r}"
                    )

            page_store: Dict[str, bytes] = {}

            def write_page(segment_ref: str, data: bytes) -> None:
                self._stage_manifest_page(
                    namespace,
                    thread_id,
                    segment_ref,
                    data,
                    page_store,
                )

            if inline is not None:
                prefix = (
                    migrate_inline_entries(inline, old_item_ids)
                    if new_entries
                    else [dict(entry) for entry in inline]
                )
                combined = prefix + new_entries
                # A control-only update preserves even a large legacy inline
                # layout; no migration or page rewrite is needed without a delta.
                if not new_entries or len(combined) <= MANIFEST_PAGE_SIZE:
                    normalized = compose_append_state(committed, item_entries=combined)
                else:
                    normalized = compose_append_state(
                        committed,
                        manifest=build_manifest(combined, write_page),
                    )
            else:
                updated_manifest = (
                    append_manifest(
                        manifest,
                        new_entries,
                        lambda segment_ref: self._read_manifest_segment(
                            namespace, thread_id, segment_ref
                        ),
                        write_page,
                    )
                    if new_entries
                    else dict(manifest)
                )
                normalized = compose_append_state(committed, manifest=updated_manifest)

            new_base = snapshot_base(
                self.routing_id, namespace, thread_id, run_id, normalized
            )
            live_refs = None
            if replacement_needs_segment_gc(previous, normalized):
                # Preflight the post-append survivors, including staged pages,
                # before publishing anything to the in-memory maps.
                live_refs = self._replacement_gc_live_refs(
                    namespace,
                    thread_id,
                    run_id,
                    normalized,
                    page_store,
                )

            self._publish_staged(namespace, thread_id, item_store, page_store)
            run["state"] = normalized
            run["updated_at"] = time.time()
            run.setdefault("commits", []).append(
                (
                    committed["_checkpoint"]["stream_id"],
                    next_revision,
                    prepared_event
                    if prepared_event is not None
                    else {
                        "event_type": "checkpoint",
                        "status": committed.get("status"),
                    },
                )
            )
            if prepared_event is not None:
                run["events"].append(prepared_event)
            if live_refs is not None:
                self._prune_live_refs(namespace, thread_id, live_refs)
            return CheckpointTail(new_base, old_base.item_count, deepcopy(committed))

    def read_commits(self, namespace, thread_id, run_id, *, after=None, limit=100):
        with self._lock:
            state = self.load_state(namespace, thread_id, run_id)
            latest = validate_read(state, namespace, thread_id, run_id, after, limit)
            run = self._get_run(namespace, thread_id, run_id)
            records = (
                []
                if after is None
                else list(
                    islice(
                        (
                            (revision, data)
                            for stream_id, revision, data in run.get("commits", [])
                            if stream_id == latest.stream_id
                            and revision > after.revision
                        ),
                        limit,
                    )
                )
            )
            return make_page(state, latest, after, records)

    def __init__(self) -> None:
        self._routing_id = uuid4().hex
        self._data: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        self._message_items: Dict[str, Dict[str, Dict[str, bytes]]] = {}
        self._manifest_pages: Dict[str, Dict[str, Dict[str, bytes]]] = {}
        self._lock = RLock()

    @property
    def routing_id(self) -> str:
        return f"memory:{self._routing_id}"

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

    def _stage_state(
        self,
        namespace: str,
        thread_id: str,
        state: Mapping[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, bytes], Dict[str, bytes]]:
        """Build a normalized state without publishing any durable rows."""
        normalized = deepcopy(dict(state))
        messages = normalized.pop("messages", None)
        if not isinstance(messages, Mapping):
            return normalized, {}, {}

        items = messages.get("items")
        if not isinstance(items, list):
            return normalized, {}, {}

        item_store = {}
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
        page_store: Dict[str, bytes] = {}
        if len(item_entries) <= MANIFEST_PAGE_SIZE:
            normalized["_messages"] = {
                "state": message_state,
                "item_entries": item_entries,
            }
            return normalized, item_store, page_store

        def write_page(segment_ref: str, data: bytes) -> None:
            self._stage_manifest_page(
                namespace,
                thread_id,
                segment_ref,
                data,
                page_store,
            )

        normalized["_messages"] = {
            "state": message_state,
            "manifest": build_manifest(item_entries, write_page),
        }
        return normalized, item_store, page_store

    def _publish_staged(
        self,
        namespace: str,
        thread_id: str,
        item_store: Mapping[str, bytes],
        page_store: Mapping[str, bytes],
    ) -> None:
        if item_store:
            self._message_items.setdefault(namespace, {}).setdefault(
                thread_id, {}
            ).update(item_store)
        if page_store:
            self._manifest_pages.setdefault(namespace, {}).setdefault(
                thread_id, {}
            ).update(page_store)

    def _denormalize_state(
        self,
        namespace: str,
        thread_id: str,
        state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        restored = deepcopy(dict(state))
        self._validate_checkpoint_envelope(restored.get("_checkpoint"))
        normalized_messages = restored.pop("_messages", None)
        if not isinstance(normalized_messages, Mapping):
            return restored

        message_state = normalized_messages.get("state")
        item_entries = self._stored_manifest_entries(
            namespace, thread_id, normalized_messages
        )
        if not isinstance(message_state, Mapping):
            raise ValueError("Checkpoint manifest is missing its message state")

        messages = deepcopy(dict(message_state))
        messages["items"] = list(
            self._iter_message_items(namespace, thread_id, item_entries)
        )
        restored["messages"] = messages
        return restored

    def _stored_manifest_entries(
        self,
        namespace: str,
        thread_id: str,
        normalized_messages: Mapping[str, Any],
    ) -> Iterable[Any]:
        """Choose the inline or segmented manifest layout for one snapshot."""
        item_entries, manifest = resolve_manifest_layout(normalized_messages)
        if item_entries is not None:
            return item_entries
        return iter_manifest_entries(
            manifest,
            lambda segment_ref: self._read_manifest_segment(
                namespace, thread_id, segment_ref
            ),
        )

    def _replacement_gc_live_refs(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        normalized: Mapping[str, Any],
        page_store: Mapping[str, bytes],
    ) -> tuple[set[str], set[str]]:
        """Preflight post-replacement liveness including staged candidate pages."""
        existing_runs = self._data.get(namespace, {}).get(thread_id, {})
        runs = [
            run for current_id, run in existing_runs.items() if current_id != run_id
        ]
        runs.append({"state": normalized})
        return self._collect_live_refs(
            namespace,
            thread_id,
            runs,
            page_overlay=page_store,
        )

    def _read_manifest_segment(
        self,
        namespace: str,
        thread_id: str,
        segment_ref: str,
    ) -> bytes:
        return self._manifest_pages.get(namespace, {}).get(thread_id, {})[segment_ref]

    def _iter_message_items(
        self,
        namespace: str,
        thread_id: str,
        item_entries: Iterable[Any],
    ) -> Iterator[Dict[str, Any]]:
        """Yield restored items while validating stored payload bytes.

        Commit validation consumes the iterator without retaining the timeline,
        but it still decodes and restores every manifest occurrence so malformed
        bytes and wrong payload shapes fail before publication.
        """
        item_store = self._message_items.get(namespace, {}).get(thread_id, {})
        for entry in item_entries:
            if not isinstance(entry, Mapping):
                raise ValueError("Checkpoint message entry is corrupted")
            item_ref = entry.get("item_ref")
            if (
                not isinstance(item_ref, str)
                or not item_ref
                or item_ref not in item_store
            ):
                raise ValueError(
                    "Checkpoint message item is missing or corrupted: "
                    f"{namespace}/{thread_id}/{item_ref!r}"
                )
            payload = msgspec.msgpack.decode(item_store[item_ref])
            if not isinstance(payload, Mapping):
                raise ValueError(
                    "Checkpoint message item is missing or corrupted: "
                    f"{namespace}/{thread_id}/{item_ref!r}"
                )
            yield restore_item_occurrence(payload, entry)

    def _load_commit_envelope(
        self,
        namespace: str,
        thread_id: str,
        state: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Return only the previous envelope needed by a revision commit.

        The revision preparation reads ``_checkpoint`` from the current state and
        inherits its branch, head and extensions. A full previous timeline is not
        retained, but payload bytes are still decoded and validated so
        corruption cannot be published.
        """
        return self._commit_envelope_from_state(namespace, thread_id, state)

    def _commit_envelope_from_state(
        self,
        namespace: str,
        thread_id: str,
        state: Mapping[str, Any],
        *,
        collect_item_ids: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Validate prior payloads once and optionally collect occurrence ids."""
        self._validate_checkpoint_envelope(state.get("_checkpoint"))
        normalized_messages = state.get("_messages")
        if isinstance(normalized_messages, Mapping):
            item_entries = self._stored_manifest_entries(
                namespace, thread_id, normalized_messages
            )
            if not isinstance(normalized_messages.get("state"), Mapping):
                raise ValueError("Checkpoint manifest is missing its message state")
            for index, item in enumerate(
                self._iter_message_items(namespace, thread_id, item_entries)
            ):
                if collect_item_ids is not None:
                    item_id = item.get("item_id")
                    if not isinstance(item_id, str) or not item_id:
                        item_id = legacy_item_id(
                            namespace=namespace,
                            thread_id=thread_id,
                            index=index,
                            item=item,
                        )
                    collect_item_ids.append(item_id)
        if "_checkpoint" not in state:
            return {}
        return {"_checkpoint": deepcopy(state["_checkpoint"])}

    @staticmethod
    def _item_ref(item: Any) -> str:
        payload = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        return "item_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _collect_live_refs(
        self,
        namespace: str,
        thread_id: str,
        runs: Iterable[Mapping[str, Any]] | None = None,
        page_overlay: Mapping[str, bytes] | None = None,
    ) -> tuple[set[str], set[str]]:
        live_items: set[str] = set()
        live_segments: set[str] = set()
        seen_segments: set[str] = set()
        if runs is None:
            runs = self._data.get(namespace, {}).get(thread_id, {}).values()

        def read_page(segment_ref: str) -> bytes:
            if page_overlay is not None:
                staged = page_overlay.get(segment_ref)
                if staged is not None:
                    return staged
            return self._read_manifest_segment(namespace, thread_id, segment_ref)

        for run in runs:
            self._collect_run_refs(
                run,
                read_page,
                live_items,
                live_segments,
                seen_segments,
            )
        return live_items, live_segments

    def _collect_run_refs(
        self,
        run: Mapping[str, Any],
        read_page,
        live_items: set[str],
        live_segments: set[str],
        seen_segments: set[str],
    ) -> None:
        messages = run.get("state", {}).get("_messages")
        if not isinstance(messages, Mapping):
            return
        item_entries, manifest = resolve_manifest_layout(messages)
        if not isinstance(messages.get("state"), Mapping):
            raise ValueError("Checkpoint manifest is missing its message state")
        if item_entries is not None:
            live_items.update(self._collect_entry_refs(item_entries))
        if manifest is None:
            return
        for segment_ref, entries in iter_manifest_pages(manifest, read_page):
            if segment_ref in seen_segments:
                # A seen page implies its whole parent chain is live already.
                break
            seen_segments.add(segment_ref)
            live_segments.add(segment_ref)
            live_items.update(self._collect_entry_refs(entries))

    @staticmethod
    def _collect_entry_refs(entries: Iterable[Any]) -> set[str]:
        refs: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise ValueError(f"Checkpoint message entry {index} is corrupted")
            item_ref = entry.get("item_ref")
            if not isinstance(item_ref, str) or not item_ref:
                raise ValueError(
                    f"Checkpoint message entry {index} is missing or corrupted"
                )
            refs.add(item_ref)
        return refs

    def _cleanup_orphaned_items(self, namespace: str, thread_id: str) -> None:
        self._prune_live_refs(
            namespace,
            thread_id,
            self._collect_live_refs(namespace, thread_id),
        )

    def _prune_live_refs(
        self,
        namespace: str,
        thread_id: str,
        live_refs: tuple[set[str], set[str]],
    ) -> None:
        live_items, live_segments = live_refs
        self._prune_store(self._message_items, namespace, thread_id, live_items)
        self._prune_store(self._manifest_pages, namespace, thread_id, live_segments)

    def _prune_store(
        self,
        store: Dict[str, Dict[str, Dict[str, bytes]]],
        namespace: str,
        thread_id: str,
        live: set[str],
    ) -> None:
        values = store.get(namespace, {}).get(thread_id)
        if values is None:
            return
        for ref in list(values):
            if ref not in live:
                del values[ref]
        if not values:
            ns_values = store.get(namespace)
            if ns_values is not None:
                ns_values.pop(thread_id, None)
                if not ns_values:
                    store.pop(namespace, None)

    def save_state(
        self,
        namespace: str,
        thread_id: str,
        run_id: str,
        state: Mapping[str, Any],
    ) -> None:
        with self._lock:
            self._validate_checkpoint_envelope(state.get("_checkpoint"))
            previous = self._get_run(namespace, thread_id, run_id)
            normalized, item_store, page_store = self._stage_state(
                namespace, thread_id, state
            )
            live_refs = None
            if previous is not None and replacement_needs_segment_gc(
                previous["state"], normalized
            ):
                live_refs = self._replacement_gc_live_refs(
                    namespace,
                    thread_id,
                    run_id,
                    normalized,
                    page_store,
                )
            self._publish_staged(namespace, thread_id, item_store, page_store)
            run = self._ensure_run(namespace, thread_id, run_id)
            run["state"] = normalized
            run["updated_at"] = time.time()
            if live_refs is not None:
                self._prune_live_refs(namespace, thread_id, live_refs)

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
            self._set_fork_metadata(
                forked,
                namespace=namespace,
                source_thread_id=source_thread_id,
                source_run_id=source_run_id,
                target_thread_id=target_thread_id,
                target_run_id=target_run_id,
                at_item_id=at_item_id,
            )

            messages = forked.get("messages")
            if isinstance(messages, dict):
                messages["thread_id"] = target_thread_id

            normalized, item_store, page_store = self._stage_state(
                namespace,
                target_thread_id,
                forked,
            )
            previous_target = self._get_run(namespace, target_thread_id, target_run_id)
            live_refs = None
            if previous_target is not None and replacement_needs_segment_gc(
                previous_target["state"], normalized
            ):
                live_refs = self._replacement_gc_live_refs(
                    namespace,
                    target_thread_id,
                    target_run_id,
                    normalized,
                    page_store,
                )
            else:
                # Fail before publishing when another run in the target thread
                # has a corrupt manifest that post-publication GC must read.
                existing_runs = self._data.get(namespace, {}).get(target_thread_id, {})
                self._collect_live_refs(
                    namespace,
                    target_thread_id,
                    (
                        run
                        for target_id, run in existing_runs.items()
                        if target_id != target_run_id
                    ),
                )
            self._publish_staged(namespace, target_thread_id, item_store, page_store)
            target = self._ensure_run(namespace, target_thread_id, target_run_id)
            target["state"] = normalized
            target["updated_at"] = time.time()
            if live_refs is not None:
                self._prune_live_refs(namespace, target_thread_id, live_refs)
            else:
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
            prepared_event = deepcopy(dict(event))
            previous = self._get_run(namespace, thread_id, run_id)
            normalized, item_store, page_store = self._stage_state(
                namespace, thread_id, state
            )
            live_refs = None
            if previous is not None and replacement_needs_segment_gc(
                previous["state"], normalized
            ):
                live_refs = self._replacement_gc_live_refs(
                    namespace,
                    thread_id,
                    run_id,
                    normalized,
                    page_store,
                )
            self._publish_staged(namespace, thread_id, item_store, page_store)
            run = self._ensure_run(namespace, thread_id, run_id)
            run["state"] = normalized
            run["updated_at"] = time.time()
            run["events"].append(prepared_event)
            if live_refs is not None:
                self._prune_live_refs(namespace, thread_id, live_refs)

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
                self._load_commit_envelope(namespace, thread_id, run["state"])
                if run is not None
                else {}
            )
            committed, next_revision = self._prepare_revision_state(
                state,
                current,
                expected_revision=expected_revision,
                branch_id=branch_id,
                head_item_id=head_item_id,
                extension_state=extension_state,
            )
            prepared_event = deepcopy(dict(event)) if event is not None else None
            durable_event = (
                deepcopy(dict(event))
                if event is not None
                else {"event_type": "checkpoint", "status": committed.get("status")}
            )
            normalized, item_store, page_store = self._stage_state(
                namespace, thread_id, committed
            )
            live_refs = None
            if run is not None and replacement_needs_segment_gc(
                run["state"], normalized
            ):
                live_refs = self._replacement_gc_live_refs(
                    namespace,
                    thread_id,
                    run_id,
                    normalized,
                    page_store,
                )
            commit_base = optional_snapshot_base(
                self.routing_id, namespace, thread_id, run_id, normalized
            )
            self._publish_staged(namespace, thread_id, item_store, page_store)
            if run is None:
                run = self._ensure_run(namespace, thread_id, run_id)
            run["state"] = normalized
            run["updated_at"] = time.time()
            run.setdefault("commits", []).append(
                (committed["_checkpoint"]["stream_id"], next_revision, durable_event)
            )
            if prepared_event is not None:
                run["events"].append(prepared_event)
            if live_refs is not None:
                self._prune_live_refs(namespace, thread_id, live_refs)
            return CheckpointCommit(
                next_revision,
                committed,
                committed["_checkpoint"]["branch_id"],
                committed["_checkpoint"]["head_item_id"],
                base=commit_base,
            )

    # Freeze the original identity so a later class-level wrapper is honored.
    _agent_commit_implementation = commit_state

    def load_latest_run(
        self,
        namespace: str,
        thread_id: str,
    ) -> Mapping[str, Any] | None:
        with self._lock:
            return super().load_latest_run(namespace, thread_id)

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
            if run_id not in thread_runs:
                return False
            remaining = [run for rid, run in thread_runs.items() if rid != run_id]
            live_items, live_segments = self._collect_live_refs(
                namespace, thread_id, remaining
            )
            del thread_runs[run_id]
            self._prune_store(self._message_items, namespace, thread_id, live_items)
            self._prune_store(self._manifest_pages, namespace, thread_id, live_segments)
            return True

    def _plan_clear(
        self,
        namespaces: List[str],
        thread_id: str | None,
        cutoff: float | None,
    ) -> tuple[
        List[tuple[str, str, List[str]]],
        List[tuple[str, str, set[str], set[str]]],
    ]:
        """Validate and stage every deletion before mutating durable state."""
        removals: List[tuple[str, str, List[str]]] = []
        prunes: List[tuple[str, str, set[str], set[str]]] = []
        for ns in namespaces:
            ns_data = self._data.get(ns)
            if ns_data is None:
                continue
            threads = [thread_id] if thread_id is not None else list(ns_data.keys())
            for sid in threads:
                thread = ns_data.get(sid)
                if thread is None:
                    continue
                keep = {
                    rid: run
                    for rid, run in thread.items()
                    if cutoff is not None and run["updated_at"] >= cutoff
                }
                to_delete = [rid for rid in thread if rid not in keep]
                if not to_delete:
                    continue
                live_items, live_segments = self._collect_live_refs(
                    ns, sid, keep.values()
                )
                removals.append((ns, sid, to_delete))
                prunes.append((ns, sid, live_items, live_segments))
        return removals, prunes

    def clear(
        self,
        namespace: str | None = None,
        thread_id: str | None = None,
        *,
        older_than: float | None = None,
    ) -> int:
        cutoff = time.time() - older_than if older_than is not None else None
        with self._lock:
            namespaces = (
                [namespace] if namespace is not None else list(self._data.keys())
            )
            removals, prunes = self._plan_clear(namespaces, thread_id, cutoff)
            removed = 0
            for ns, sid, run_ids in removals:
                thread = self._data[ns][sid]
                for rid in run_ids:
                    del thread[rid]
                    removed += 1
            for ns, sid, live_items, live_segments in prunes:
                self._prune_store(self._message_items, ns, sid, live_items)
                self._prune_store(self._manifest_pages, ns, sid, live_segments)
            for ns, sid, _run_ids in removals:
                ns_data = self._data.get(ns)
                if ns_data is not None and not ns_data.get(sid):
                    ns_data.pop(sid, None)
                    if not ns_data:
                        self._data.pop(ns, None)
        return removed
