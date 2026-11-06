from __future__ import annotations

import hashlib
import itertools
import json
import os
from typing import Any, Mapping

# A random per-process prefix plus a counter keeps ids unique without one
# ``getrandom`` syscall per item; each syscall drops the GIL, and on a busy
# event loop re-acquiring it competes with the checkpoint writer thread.
_item_id_prefix = os.urandom(8).hex()
_item_id_counter = itertools.count()


def _reseed_item_ids() -> None:
    global _item_id_prefix, _item_id_counter  # noqa: PLW0603
    _item_id_prefix = os.urandom(8).hex()
    _item_id_counter = itertools.count()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reseed_item_ids)


def new_item_id() -> str:
    """Return a new logical identity for one timeline occurrence."""
    return f"itm_{_item_id_prefix}{next(_item_id_counter) & 0xFFFFFFFFFFFFFFFF:016x}"


def legacy_item_id(
    *,
    namespace: str | None,
    thread_id: str | None,
    index: int,
    item: Mapping[str, Any],
) -> str:
    """Derive a stable occurrence id for an item written before item ids existed."""
    payload = {key: value for key, value in item.items() if key != "item_id"}
    encoded = json.dumps(
        {
            "namespace": namespace,
            "thread_id": thread_id,
            "index": index,
            "item": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
    return f"itm_{digest}"


def item_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable payload stored independently of occurrence state."""
    return {key: value for key, value in item.items() if key != "item_id"}


def split_item_occurrence(
    *,
    namespace: str,
    thread_id: str,
    index: int,
    item: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate one logical occurrence from its deduplicated payload."""
    payload = item_payload(item)
    item_id = item.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        item_id = legacy_item_id(
            namespace=namespace,
            thread_id=thread_id,
            index=index,
            item=payload,
        )
    occurrence = {"item_id": item_id}
    return payload, occurrence


def restore_item_occurrence(
    payload: Mapping[str, Any],
    occurrence: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild one public timeline item from payload and occurrence state."""
    item = dict(payload)
    item_id = occurrence.get("item_id")
    if isinstance(item_id, str) and item_id:
        item["item_id"] = item_id
    return item
