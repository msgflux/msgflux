"""Shared immutable checkpoint manifest segments.

The checkpoint stores keep the ordered occurrence manifest outside the run state
once a timeline grows beyond one page.  A manifest is a linked chain of
immutable, content-addressed pages.  Each page stores at most
``MANIFEST_PAGE_SIZE`` occurrence entries, the cumulative entry count through
that page, and the reference of its parent page.  The page reference is the
SHA-256 digest of the page's canonical JSON bytes, so the parent reference is
part of the child digest.  Appending to a history therefore reuses every
complete prefix page and only rewrites the rightmost partial page.

The codec is deliberately storage agnostic: callers provide ``write_page`` and
``read_page`` callbacks.  Both SQLite and the in-memory store share this module
so the on-disk and in-memory formats cannot drift.  This module uses only the
standard library and never mutates the entries supplied to
:func:`build_manifest`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Tuple

MANIFEST_PAGE_SIZE = 128
"""Maximum number of occurrence entries stored in one manifest page."""

MANIFEST_FORMAT = "segments_v1"
"""Descriptor format stored at ``_messages.manifest["format"]``."""

_REF_PREFIX = "seg_"
_REF_PATTERN = re.compile(r"seg_[0-9a-f]{64}")
_PAGE_KEYS = frozenset({"parent", "count", "entries"})

ReadPage = Callable[[str], bytes]
WritePage = Callable[[str, bytes], None]


def resolve_manifest_layout(
    normalized_messages: Mapping[str, Any],
) -> Tuple[List[Any] | None, Mapping[str, Any] | None]:
    """Resolve the private ``_messages`` layout shared by providers.

    Exactly one of an inline ``item_entries`` list or a segmented ``manifest``
    mapping is valid. Mixed, missing, or wrongly typed layout keys are
    corruption: readers, revision commits and garbage collection must all fail
    closed instead of silently dropping message history.

    Returns ``(item_entries, manifest)`` with one side always ``None``.
    """
    has_inline = "item_entries" in normalized_messages
    has_manifest = "manifest" in normalized_messages
    if has_inline and has_manifest:
        raise ValueError("Checkpoint manifest mixes inline item_entries with segments")
    if has_inline:
        item_entries = normalized_messages["item_entries"]
        if not isinstance(item_entries, list):
            raise ValueError("Checkpoint manifest item_entries must be a list")
        return item_entries, None
    if has_manifest:
        manifest = normalized_messages["manifest"]
        if not isinstance(manifest, Mapping):
            raise ValueError("Checkpoint manifest descriptor must be a mapping")
        return None, manifest
    raise ValueError("Checkpoint manifest is missing item_entries or manifest")


def build_manifest(
    entries: Iterable[Mapping[str, Any]],
    write_page: WritePage,
) -> Dict[str, Any]:
    """Write immutable pages for ``entries`` and return their descriptor.

    ``write_page`` is called once per page with ``(ref, data)``.  Callers should
    write only pages that do not already exist; existing pages are reused by
    reference.  Pages are emitted from oldest to newest while the input is
    consumed, so memory stays bounded by one page.  If validation fails after
    earlier pages were written, the caller must roll back or discard staged
    pages before publishing the run state.

    The returned descriptor has the exact shape::

        {"format": "segments_v1", "head": <ref or None>, "count": <N>}

    An empty manifest returns ``head=None`` and ``count=0`` and writes no page.
    Stores that keep manifests of up to ``MANIFEST_PAGE_SIZE`` entries inline
    should call this function only when the entry count exceeds that limit.
    """
    if not callable(write_page):
        raise TypeError("write_page must be callable")

    parent: str | None = None
    count = 0
    chunk: List[Dict[str, Any]] = []
    for index, entry in enumerate(entries):
        chunk.append(_copy_entry(entry, index))
        if len(chunk) == MANIFEST_PAGE_SIZE:
            parent = _write_page(parent, count + len(chunk), chunk, write_page)
            count += len(chunk)
            chunk = []

    if chunk:
        parent = _write_page(parent, count + len(chunk), chunk, write_page)
        count += len(chunk)

    return {"format": MANIFEST_FORMAT, "head": parent, "count": count}


def iter_manifest_entries(
    descriptor: Mapping[str, Any],
    read_page: ReadPage,
) -> Iterator[Dict[str, Any]]:
    """Yield occurrence entries in timeline order (oldest first).

    The page chain is validated head-to-parent first.  Only page references are
    retained, so memory is O(number of pages) rather than O(number of entries).
    Pages are then fetched oldest-first and their entries yielded one at a time.
    Every yielded entry is a fresh JSON object; mutating it cannot affect the
    input, the descriptor, or later iterations.
    """
    refs = [ref for ref, _page in _walk_pages(descriptor, read_page)]
    for ref in reversed(refs):
        page = _read_page(ref, read_page)
        yield from page["entries"]


def iter_manifest_pages(
    descriptor: Mapping[str, Any],
    read_page: ReadPage,
) -> Iterator[Tuple[str, List[Dict[str, Any]]]]:
    """Yield ``(ref, entries)`` from head to parent for garbage collection.

    Each page is read and validated once.  The chain linkage, cumulative counts
    and oldest-page count are checked before a page is yielded.
    """
    for ref, page in _walk_pages(descriptor, read_page):
        yield ref, page["entries"]


def _walk_pages(
    descriptor: Mapping[str, Any],
    read_page: ReadPage,
) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Validate and yield pages head-to-parent without retaining the chain."""
    head, count = _parse_descriptor(descriptor)
    if count == 0:
        return

    if head is None:  # _parse_descriptor guarantees a head when count > 0
        raise ValueError("Checkpoint manifest is missing a head page")
    ref = head
    page = _read_page(ref, read_page)
    if page["count"] != count:
        raise ValueError(
            "Checkpoint manifest head count does not match descriptor: "
            f"expected {count}, found {page['count']}"
        )

    seen = {ref}
    pages = 1
    while True:
        parent_ref = page["parent"]
        if parent_ref is None:
            if page["count"] != len(page["entries"]):
                raise ValueError(
                    "Checkpoint manifest oldest page count does not match "
                    f"its entry count: {page['count']} != "
                    f"{len(page['entries'])}"
                )
            yield ref, page
            return

        if parent_ref in seen:
            raise ValueError(
                f"Checkpoint manifest page chain is cyclic at {parent_ref!r}"
            )

        parent = _read_page(parent_ref, read_page)
        expected = parent["count"] + len(page["entries"])
        if page["count"] != expected:
            raise ValueError(
                "Checkpoint manifest page count is not cumulative: "
                f"expected {expected}, found {page['count']} at {ref!r}"
            )

        yield ref, page
        seen.add(parent_ref)
        pages += 1
        if pages > count:
            raise ValueError(
                "Checkpoint manifest page chain is longer than its entry count"
            )
        ref = parent_ref
        page = parent


def _parse_descriptor(descriptor: Mapping[str, Any]) -> Tuple[str | None, int]:
    """Validate a manifest descriptor and return ``(head, count)``."""
    if not isinstance(descriptor, Mapping):
        raise ValueError("Checkpoint manifest descriptor must be a mapping")
    if descriptor.get("format") != MANIFEST_FORMAT:
        raise ValueError(
            "Checkpoint manifest descriptor has unsupported format: "
            f"{descriptor.get('format')!r}"
        )
    if "count" not in descriptor:
        raise ValueError("Checkpoint manifest descriptor is missing count")
    count = descriptor["count"]
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError("Checkpoint manifest count must be an integer")
    if count < 0:
        raise ValueError("Checkpoint manifest count must not be negative")
    if "head" not in descriptor:
        raise ValueError("Checkpoint manifest descriptor is missing head")
    head = descriptor["head"]
    if count == 0:
        if head is not None:
            raise ValueError("Checkpoint manifest with no entries must not have a head")
        return None, 0
    if not isinstance(head, str):
        raise ValueError("Checkpoint manifest head must be a page reference")
    _validate_ref(head)
    return head, count


def _write_page(
    parent: str | None,
    count: int,
    entries: List[Dict[str, Any]],
    write_page: WritePage,
) -> str:
    """Encode, address and write one page, returning its reference."""
    page = {"parent": parent, "count": count, "entries": entries}
    data = _canonical_json(page)
    ref = _ref_for(data)
    write_page(ref, data)
    return ref


def _canonical_json(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for ``value``."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _ref_for(data: bytes) -> str:
    """Return the content address for canonical page bytes."""
    return _REF_PREFIX + hashlib.sha256(data).hexdigest()


def _validate_ref(ref: Any) -> str:
    """Validate a page reference and return it."""
    if not isinstance(ref, str) or _REF_PATTERN.fullmatch(ref) is None:
        raise ValueError(f"Checkpoint manifest page reference is invalid: {ref!r}")
    return ref


def _read_page(ref: str, read_page: ReadPage) -> Dict[str, Any]:
    """Read, address-check, decode and validate one page."""
    try:
        data = read_page(ref)
    except KeyError as error:
        raise ValueError(f"Checkpoint manifest page is missing: {ref!r}") from error

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ValueError(
            f"Checkpoint manifest page {ref!r} is not bytes: {type(data).__name__}"
        )
    raw = bytes(data)
    actual = _ref_for(raw)
    if actual != ref:
        raise ValueError(
            "Checkpoint manifest page digest mismatch: "
            f"expected {ref!r}, computed {actual!r}"
        )

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Checkpoint manifest page {ref!r} is not valid UTF-8"
        ) from error

    try:
        page = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Checkpoint manifest page {ref!r} is not valid JSON"
        ) from error

    _validate_page(page, ref)
    return page


def _object_without_duplicate_keys(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Build a JSON object while rejecting duplicate keys."""
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Checkpoint manifest page contains duplicate JSON key {key!r}"
            )
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    """Reject NaN and infinities, which are not valid portable JSON."""
    raise ValueError(
        f"Checkpoint manifest page contains non-finite JSON number {value!r}"
    )


def _validate_page(page: Any, ref: str) -> None:
    """Validate the shape and contents of a decoded page."""
    if not isinstance(page, Mapping):
        raise ValueError(f"Checkpoint manifest page {ref!r} must be a JSON object")
    if set(page) != _PAGE_KEYS:
        raise ValueError(
            f"Checkpoint manifest page {ref!r} has unexpected keys: {sorted(page)}"
        )

    parent = page["parent"]
    if parent is not None:
        _validate_ref(parent)

    count = page["count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(
            f"Checkpoint manifest page {ref!r} count must be a positive integer"
        )

    entries = page["entries"]
    if not isinstance(entries, list):
        raise ValueError(f"Checkpoint manifest page {ref!r} entries must be a list")
    if not 1 <= len(entries) <= MANIFEST_PAGE_SIZE:
        raise ValueError(
            f"Checkpoint manifest page {ref!r} has an invalid entry count: "
            f"{len(entries)}"
        )
    for index, entry in enumerate(entries):
        _validate_entry(entry, ref, index)


def _validate_entry(entry: Any, ref: str, index: int) -> None:
    """Validate one decoded occurrence entry."""
    if not isinstance(entry, Mapping):
        raise ValueError(
            f"Checkpoint manifest page {ref!r} entry {index} must be a mapping"
        )
    item_ref = entry.get("item_ref")
    item_id = entry.get("item_id")
    if not isinstance(item_ref, str) or not item_ref:
        raise ValueError(
            f"Checkpoint manifest page {ref!r} entry {index} has an invalid item_ref"
        )
    if not isinstance(item_id, str) or not item_id:
        raise ValueError(
            f"Checkpoint manifest page {ref!r} entry {index} has an invalid item_id"
        )


def _copy_entry(entry: Any, index: int) -> Dict[str, Any]:
    """Validate an input entry and return an isolated JSON-native copy."""
    if not isinstance(entry, Mapping):
        raise ValueError(f"Checkpoint manifest entry {index} must be a mapping")
    item_ref = entry.get("item_ref")
    item_id = entry.get("item_id")
    if not isinstance(item_ref, str) or not item_ref:
        raise ValueError(f"Checkpoint manifest entry {index} has an invalid item_ref")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError(f"Checkpoint manifest entry {index} has an invalid item_id")

    copied = _copy_json_value(entry, f"entry[{index}]")
    if not isinstance(copied, dict):  # a mapping always copies to a dict
        raise ValueError(f"Checkpoint manifest entry {index} must be a mapping")
    return copied


def _copy_json_value(value: Any, path: str) -> Any:
    """Validate JSON representability and return an isolated copy of ``value``."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Checkpoint manifest {path} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [
            _copy_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        copied: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"Checkpoint manifest {path} contains a non-string key"
                )
            copied[key] = _copy_json_value(item, f"{path}.{key}")
        return copied
    raise ValueError(
        f"Checkpoint manifest {path} is not JSON-serializable: {type(value).__name__}"
    )


def replacement_needs_segment_gc(
    old_state: Mapping[str, Any],
    new_state: Mapping[str, Any],
) -> bool:
    """Return whether replacing a run can orphan its old segment head.

    Only a previously segmented snapshot can leave segment pages behind.
    Legacy inline state has no pages of its own, so replacing it never needs
    page GC. A malformed old manifest is treated as changed when the run is
    being replaced, so its unreachable pages are still collected.
    """
    old_messages = old_state.get("_messages")
    if not isinstance(old_messages, Mapping) or "manifest" not in old_messages:
        return False
    old_manifest = old_messages["manifest"]
    if not isinstance(old_manifest, Mapping):
        return True
    old_head = old_manifest.get("head")
    if not isinstance(old_head, str):
        return True
    new_messages = new_state.get("_messages")
    if not isinstance(new_messages, Mapping):
        return True
    new_manifest = new_messages.get("manifest")
    if not isinstance(new_manifest, Mapping):
        return True
    return new_manifest.get("head") != old_head
