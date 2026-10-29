"""Explicit suffix recovery and append provenance for checkpoint stores.

A suffix is a storage window, not a complete ChatMessages conversation. The
base identifies the exact persisted snapshot; it is a concurrency token, not an
authorization credential. Full checkpoint APIs keep their existing semantics.
"""

from __future__ import annotations

import hashlib
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from msgflux.data.stores._manifest import (
    _canonical_json,
    _copy_json_value,
    _parse_descriptor,
    iter_manifest_suffix,
    resolve_manifest_layout,
)


@dataclass(frozen=True)
class CheckpointBase:
    """Identity and fingerprint of the complete snapshot behind a suffix."""

    routing_id: str
    namespace: str
    thread_id: str
    run_id: str
    revision: int
    item_count: int
    digest: str

    def __post_init__(self) -> None:
        for name in ("routing_id", "namespace", "thread_id", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Checkpoint base {name} must be a non-empty string")
        for name in ("revision", "item_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Checkpoint base {name} must be a non-negative integer"
                )
        if (
            not isinstance(self.digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.digest) is None
        ):
            raise ValueError("Checkpoint base digest must be a SHA-256 hex string")


@dataclass(frozen=True)
class CheckpointTail:
    """Isolated suffix state with absolute offset and complete-base provenance.

    ``state["messages"]["items"]`` contains only occurrences at ``start`` and
    later. It must not be passed to ``save_state`` or ``commit_state`` as a full
    replacement. ``base.item_count`` is the canonical count, not suffix length.
    """

    base: CheckpointBase
    start: int
    state: Mapping[str, Any]


def snapshot_base(
    routing_id: str,
    namespace: str,
    thread_id: str,
    run_id: str,
    normalized: Mapping[str, Any],
) -> CheckpointBase:
    messages = normalized.get("_messages")
    if not isinstance(messages, Mapping) or not isinstance(
        messages.get("state"), Mapping
    ):
        raise ValueError("Incremental checkpoints require a normalized message history")
    inline, manifest = resolve_manifest_layout(messages)
    count = len(inline) if inline is not None else _parse_descriptor(manifest)[1]
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("Checkpoint manifest count must be a non-negative integer")
    checkpoint = normalized.get("_checkpoint", {})
    revision = checkpoint.get("revision", 0)
    # Unlike str-based serialization, a portable JSON snapshot cannot confuse
    # bytes with their textual representation or tuples with lists on a later
    # same-revision replacement. Full store APIs keep their broader support.
    try:
        portable = _copy_json_value(normalized, "checkpoint")
        encoded = _canonical_json(portable)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Incremental checkpoint control metadata must be JSON-compatible"
        ) from error
    digest = hashlib.sha256(encoded).hexdigest()
    return CheckpointBase(
        routing_id, namespace, thread_id, run_id, revision, count, digest
    )


def recover_tail(
    normalized: Mapping[str, Any],
    base: CheckpointBase,
    limit: int,
    read_page: Callable[[str], bytes],
    read_items: Callable[[Iterable[Any]], Iterable[Mapping[str, Any]]],
) -> CheckpointTail:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("Checkpoint tail limit must be a non-negative integer")
    messages = normalized["_messages"]
    inline, manifest = resolve_manifest_layout(messages)
    start = max(0, base.item_count - limit)
    entries = (
        inline[start:]
        if inline is not None
        else iter_manifest_suffix(manifest, read_page, start=start)
    )
    state = deepcopy(dict(normalized))
    state.pop("_messages")
    state["messages"] = deepcopy(dict(messages["state"]))
    state["messages"]["items"] = list(read_items(entries))
    return CheckpointTail(base, start, state)


def append_control_state(
    normalized: Mapping[str, Any],
    items: list[Mapping[str, Any]],
    state_updates: Mapping[str, Any] | None,
    message_updates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Prepare a bounded public tail/control state for revision preparation."""
    if not isinstance(items, list):
        raise ValueError("Checkpoint append items must be a list")
    ids = set()
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("Checkpoint append items must be mappings")
        item_id = item.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("Checkpoint append requires explicit non-empty item_id")
        if item_id in ids:
            raise ValueError(f"Duplicate ChatMessages item_id `{item_id}`.")
        ids.add(item_id)
    updates = {} if state_updates is None else state_updates
    message_updates = {} if message_updates is None else message_updates
    if not isinstance(updates, Mapping) or not isinstance(message_updates, Mapping):
        raise ValueError("Checkpoint append updates must be mappings")
    if {"messages", "_messages", "_checkpoint"}.intersection(updates):
        raise ValueError(
            "Checkpoint append cannot replace history or checkpoint envelope"
        )
    if "items" in message_updates:
        raise ValueError("Checkpoint message updates cannot replace items")
    state = {
        key: deepcopy(value) for key, value in normalized.items() if key != "_messages"
    }
    state.update(deepcopy(dict(updates)))
    messages = deepcopy(dict(normalized["_messages"]["state"]))
    messages.update(deepcopy(dict(message_updates)))
    messages["items"] = deepcopy(items)
    state["messages"] = messages
    return state


def compose_append_state(
    committed: Mapping[str, Any],
    *,
    item_entries: list[Any] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    messages = committed.get("messages")
    if not isinstance(messages, Mapping):
        raise ValueError("Checkpoint append requires message state")
    normalized = {key: value for key, value in committed.items() if key != "messages"}
    message_state = {key: value for key, value in messages.items() if key != "items"}
    normalized_messages: dict[str, Any] = {"state": message_state}
    if item_entries is not None:
        normalized_messages["item_entries"] = item_entries
    if manifest is not None:
        normalized_messages["manifest"] = manifest
    normalized["_messages"] = normalized_messages
    return normalized


def migrate_inline_entries(
    entries: list[Any],
    old_item_ids: list[str],
) -> list[dict[str, Any]]:
    """Copy inline occurrences, adding deterministic ids for id-less legacy."""
    migrated: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError("Checkpoint message entry is corrupted")
        copied = dict(entry)
        item_id = copied.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            if index >= len(old_item_ids):
                raise ValueError(
                    "Checkpoint message entry is missing its occurrence id"
                )
            copied["item_id"] = old_item_ids[index]
        migrated.append(copied)
    return migrated


def validate_tail_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("Checkpoint tail limit must be a non-negative integer")


def same_payload_value(left: Any, right: Any) -> bool:
    """Compare decoded payloads without conflating JSON scalar types."""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            same_payload_value(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            same_payload_value(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def is_plain_json(value: Any) -> bool:
    """Qualify optional optimizations without copying or invoking user hooks.

    Full APIs accept opaque objects and Mapping subclasses. Only exact native
    JSON containers/scalars are eligible for automatic fingerprints, so an
    optional optimization never calls an opaque object's iteration/conversion
    methods. The active set is bounded by nesting depth and rejects cycles.
    """
    try:
        return _plain_json_value(value, set())
    except RecursionError:
        return False


def _plain_json_value(value: Any, active: set[int]) -> bool:
    value_type = type(value)
    if value is None or value_type is str or value_type is bool or value_type is int:
        return True
    if value_type is float:
        return math.isfinite(value)
    if value_type is not list and value_type is not dict:
        return False
    identity = id(value)
    if identity in active:
        return False
    active.add(identity)
    try:
        if value_type is list:
            return all(_plain_json_value(item, active) for item in value)
        return all(
            type(key) is str and _plain_json_value(item, active)
            for key, item in value.items()
        )
    finally:
        active.remove(identity)


def optional_snapshot_base(
    routing_id: str,
    namespace: str,
    thread_id: str,
    run_id: str,
    normalized: Mapping[str, Any],
) -> CheckpointBase | None:
    """Derive optional append provenance without narrowing full-commit inputs.

    Full APIs accept control objects beyond the incremental JSON contract. This
    pure, best-effort derivation performs no storage access: qualification
    failures disable incremental use, while transaction/integrity errors in the
    actual commit path remain outside this handler and propagate normally.
    """
    if not is_plain_json(normalized):
        return None
    try:
        return snapshot_base(routing_id, namespace, thread_id, run_id, normalized)
    except (TypeError, ValueError, RecursionError):
        return None
