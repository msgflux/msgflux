from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping
from uuid import uuid4


def new_item_id() -> str:
    """Return a new logical identity for one timeline occurrence."""
    return f"itm_{uuid4().hex}"


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
