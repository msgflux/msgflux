"""Ephemeral planning for automatic Agent checkpoint appends.

The tracker built here is run-local, immutable and intentionally small: it holds
the committed base, the item count, one digest over the already-copied canonical
checkpoint items and the control/message key sets. It never keeps a second
copy of the conversation history and is never written into durable state.

Every save revalidates the tracker against the current store, run scope and
full checkpoint state. Any mismatch, unsupported payload or stale revision
falls back to the existing full commit path; only an exact unchanged prefix is
appended. Append failures always propagate so callers cannot mask CAS or
corruption errors with an implicit full retry.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from msgflux.data.stores._manifest import _canonical_json
from msgflux.data.stores.base import AsyncCheckpointStore
from msgflux.data.stores.incremental import (
    CheckpointBase,
    CheckpointTail,
    is_plain_json,
)

_HISTORY_KEYS = frozenset({"messages", "_messages", "_checkpoint"})
_CONTAINER_KEYS = frozenset({"items"})
_QUALIFICATION_ERRORS = (TypeError, ValueError, RecursionError)


@dataclass(frozen=True)
class CheckpointTracker:
    """Opaque validation token for the last atomically committed full state."""

    base: CheckpointBase
    count: int
    digest: str
    control_keys: frozenset[str]
    message_keys: frozenset[str]


@dataclass(frozen=True)
class AppendPlan:
    """Delta and control updates ready to be appended to ``base``."""

    base: CheckpointBase
    count: int
    items: list[Mapping[str, Any]]
    state_updates: dict[str, Any]
    message_updates: dict[str, Any]
    digest: str
    control_keys: frozenset[str]
    message_keys: frozenset[str]


def _messages_and_items(
    state: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[Any]] | None:
    messages = state.get("messages")
    if not isinstance(messages, Mapping):
        return None
    items = messages.get("items")
    if not isinstance(items, list):
        return None
    return messages, items


def _fingerprint(items: list[Any], prefix_count: int) -> tuple[str, str]:
    """Hash canonical typed JSON items in one pass with O(max item) memory.

    Each item is framed with its byte length so concatenations cannot alias.
    The prefix digest covers the first ``prefix_count`` items; the whole digest
    covers every item and is stored in the next tracker.
    """
    prefix = hashlib.sha256()
    whole = hashlib.sha256()
    for index, item in enumerate(items):
        if not is_plain_json(item):
            raise ValueError("Automatic checkpoint hashing requires plain JSON items")
        encoded = _canonical_json(item)
        frame = len(encoded).to_bytes(8, "big") + encoded
        whole.update(frame)
        if index < prefix_count:
            prefix.update(frame)
    return prefix.hexdigest(), whole.hexdigest()


def _keysets(
    state: Mapping[str, Any], messages: Mapping[str, Any]
) -> tuple[frozenset[str], frozenset[str]]:
    control_keys = frozenset(key for key in state if key not in _HISTORY_KEYS)
    message_keys = frozenset(key for key in messages if key not in _CONTAINER_KEYS)
    return control_keys, message_keys


def _is_json_supported(value: Any) -> bool:
    """Check native JSON without copying values or invoking opaque hooks."""
    return is_plain_json(value)


def _updates_are_json_native(
    state: Mapping[str, Any],
    messages: Mapping[str, Any],
    extension_state: Mapping[str, Any],
) -> bool:
    if not all(
        _is_json_supported(value)
        for key, value in state.items()
        if key not in _HISTORY_KEYS
    ):
        return False
    if not all(
        _is_json_supported(value)
        for key, value in messages.items()
        if key not in _CONTAINER_KEYS
    ):
        return False
    return _is_json_supported(extension_state)


def seed_tracker(
    base: CheckpointBase | None, state: Mapping[str, Any]
) -> CheckpointTracker | None:
    """Build a tracker from an actual committed state, or disable tracking."""
    try:
        if type(base) is not CheckpointBase:
            return None
        resolved = _messages_and_items(state)
        if resolved is None:
            return None
        messages, items = resolved
        if len(items) != base.item_count:
            return None
        _, digest = _fingerprint(items, 0)
        control_keys, message_keys = _keysets(state, messages)
        return CheckpointTracker(
            base=base,
            count=len(items),
            digest=digest,
            control_keys=control_keys,
            message_keys=message_keys,
        )
    except Exception:
        return None


def tracker_from_commit(
    committed: Any, store: Any | None = None
) -> CheckpointTracker | None:
    """Seed optional provenance without failing an already committed write."""
    try:
        if (
            store is not None
            and getattr(store, "supports_agent_incremental", False) is not True
        ):
            return None
        return seed_tracker(getattr(committed, "base", None), committed.state)
    except Exception:
        return None


def tracker_snapshot(run: Any) -> tuple[CheckpointTracker | None, int]:
    """Read the tracker and revision together before planning a save."""
    with run._command_receipt_lock:
        return run._checkpoint_tracker, run.revision


def advance_revision(run: Any, revision: int) -> None:
    """Record a successful write's revision without waiting for the seed hash."""
    if run is None:
        return
    with run._command_receipt_lock:
        run.revision = max(run.revision, revision)


def invalidate_tracker(run: Any, expected: CheckpointTracker | None) -> bool:
    """Clear the tracker only if no newer save already replaced it."""
    if run is None:
        return False
    with run._command_receipt_lock:
        if run._checkpoint_tracker is expected:
            run._checkpoint_tracker = None
            return True
        return False


def commit_tracker(
    run: Any,
    expected: CheckpointTracker | None,
    tracker: CheckpointTracker | None,
    revision: int,
) -> bool:
    """Install a commit/append tracker with identity and revision CAS.

    The revision is advanced monotonically even when a newer tracker won the
    identity race. A tracker whose base revision disagrees with the committed
    revision is never installed, so a stale seed cannot outlive its write.
    """
    if run is None:
        return False
    with run._command_receipt_lock:
        if tracker is not None and tracker.base.revision != revision:
            tracker = None
        run.revision = max(run.revision, revision)
        if run._checkpoint_tracker is not expected:
            return False
        if tracker is not None and tracker.base.revision < run.revision:
            run._checkpoint_tracker = None
            return False
        run._checkpoint_tracker = tracker
        return True


def plan_append(
    store: Any,
    run: Any,
    state: Mapping[str, Any],
    tracker: CheckpointTracker | None,
    *,
    namespace: str,
    thread_id: str,
    run_id: str,
    expected_revision: int | None = None,
) -> AppendPlan | None:
    """Return an append plan only when the whole prior history is unchanged.

    The capability and revision are checked on every save. The
    ``supports_agent_incremental`` flag is re-read dynamically so a subclass,
    instance wrapper or ``acommit_state`` override installed after the seed is
    honored.
    """
    if run is None:
        return None
    revision = run.revision if expected_revision is None else expected_revision
    if not _eligible(store, tracker, revision, namespace, thread_id, run_id):
        return None
    try:
        resolved = _matching_state(tracker, state)
        if resolved is None:
            return None
        messages, items, digest = resolved
        if not _updates_are_json_native(state, messages, run.extension_state):
            return None
        updates = {
            key: value for key, value in state.items() if key not in _HISTORY_KEYS
        }
        message_updates = {
            key: value for key, value in messages.items() if key not in _CONTAINER_KEYS
        }
        control_keys, message_keys = _keysets(state, messages)
        return AppendPlan(
            base=tracker.base,
            count=len(items),
            items=list(items[tracker.count :]),
            state_updates=updates,
            message_updates=message_updates,
            digest=digest,
            control_keys=control_keys,
            message_keys=message_keys,
        )
    except Exception:
        return None


def _eligible(
    store: Any,
    tracker: CheckpointTracker | None,
    revision: int,
    namespace: str,
    thread_id: str,
    run_id: str,
) -> bool:
    if store is None or tracker is None:
        return False
    try:
        if getattr(store, "supports_agent_incremental", False) is not True:
            return False
        if getattr(store, "supports_atomic_commit", False) is not True:
            return False
        base = tracker.base
        if base.routing_id != store.routing_id:
            return False
        if (
            base.namespace != namespace
            or base.thread_id != thread_id
            or base.run_id != run_id
        ):
            return False
        return base.revision == revision and base.item_count == tracker.count
    except Exception:
        return False


def _matching_state(
    tracker: CheckpointTracker, state: Mapping[str, Any]
) -> tuple[Mapping[str, Any], list[Any], str] | None:
    resolved = _messages_and_items(state)
    if resolved is None:
        return None
    messages, items = resolved
    if len(items) < tracker.count:
        return None
    control_keys, message_keys = _keysets(state, messages)
    if control_keys != tracker.control_keys or message_keys != tracker.message_keys:
        return None
    try:
        prefix_digest, digest = _fingerprint(items, tracker.count)
    except _QUALIFICATION_ERRORS:
        return None
    if prefix_digest != tracker.digest:
        return None
    # Fingerprinting first qualifies exact native containers, so these lookups
    # cannot invoke an opaque Mapping subclass's callbacks during planning.
    for item in items[tracker.count :]:
        if type(item) is not dict:
            return None
        item_id = item.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            # The full path assigns legacy occurrence ids; append requires them.
            return None
    return messages, items, digest


def accept_append(plan: AppendPlan, tail: CheckpointTail) -> CheckpointTracker | None:
    """Build the next tracker from an append ack, or disable tracking."""
    base = tail.base
    if base.revision != plan.base.revision + 1:
        return None
    if base.item_count != plan.count:
        return None
    if (
        base.routing_id != plan.base.routing_id
        or base.namespace != plan.base.namespace
        or base.thread_id != plan.base.thread_id
        or base.run_id != plan.base.run_id
    ):
        return None
    return CheckpointTracker(
        base=base,
        count=plan.count,
        digest=plan.digest,
        control_keys=plan.control_keys,
        message_keys=plan.message_keys,
    )


def select_async_append(store: Any) -> Any | None:
    """Return an awaitable append callable, or ``None`` to use the sync path.

    Built-in stores inherit the thread-offloading base implementation. Stores
    that replace ``aappend_state`` keep their override; stores without a usable
    async append fall back to the same synchronous append the async commit path
    already uses.
    """
    append = getattr(store, "aappend_state", None)
    implementation = getattr(type(store), "aappend_state", None)
    if callable(append) and implementation is not AsyncCheckpointStore.aappend_state:
        return append
    if hasattr(store, "append_state"):
        return None
    return append
