import copy
import hashlib
import json

import pytest

from msgflux.data.stores import _manifest
from msgflux.data.stores._manifest import (
    MANIFEST_FORMAT,
    MANIFEST_PAGE_SIZE,
    build_manifest,
    iter_manifest_entries,
    iter_manifest_pages,
)


class PageStore:
    """Small in-memory page store used by the codec tests."""

    def __init__(self):
        self.pages = {}
        self.writes = []

    def write(self, ref, data):
        self.writes.append((ref, data))
        self.pages.setdefault(ref, data)

    def read(self, ref):
        return self.pages[ref]


def _entry(index, **metadata):
    entry = {
        "item_id": f"itm_{index:05d}",
        "item_ref": f"item_{index:05d}",
    }
    entry.update(metadata)
    return entry


def _entries(count, **metadata):
    return [_entry(index, **metadata) for index in range(count)]


def _build(count, **metadata):
    store = PageStore()
    descriptor = build_manifest(_entries(count, **metadata), store.write)
    return descriptor, store


def _descriptor(head, count):
    return {"format": MANIFEST_FORMAT, "head": head, "count": count}


def _store_raw(store, data):
    ref = "seg_" + hashlib.sha256(data).hexdigest()
    store.pages[ref] = data
    return ref


def _store_page(store, *, parent=None, count=1, entries=None):
    entries = [_entry(0)] if entries is None else entries
    data = json.dumps(
        {"parent": parent, "count": count, "entries": entries},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _store_raw(store, data)


def _page_refs(store):
    return [ref for ref, _data in store.writes]


def test_constants():
    assert MANIFEST_PAGE_SIZE == 128
    assert MANIFEST_FORMAT == "segments_v1"


def test_empty_manifest_has_no_head_and_writes_no_pages():
    store = PageStore()
    descriptor = build_manifest([], store.write)

    assert descriptor == {"format": MANIFEST_FORMAT, "head": None, "count": 0}
    assert store.writes == []
    assert list(iter_manifest_entries(descriptor, store.read)) == []
    assert list(iter_manifest_pages(descriptor, store.read)) == []


@pytest.mark.parametrize("count", [1, 2, 127, 128])
def test_single_page_round_trip(count):
    descriptor, store = _build(count)

    assert descriptor["count"] == count
    assert len(store.writes) == 1
    ref, data = store.writes[0]
    assert descriptor["head"] == ref
    assert isinstance(data, bytes)
    assert ref == "seg_" + hashlib.sha256(data).hexdigest()

    page = json.loads(data)
    assert set(page) == {"parent", "count", "entries"}
    assert page["parent"] is None
    assert page["count"] == count
    assert page["entries"] == _entries(count)

    assert list(iter_manifest_entries(descriptor, store.read)) == _entries(count)
    assert list(iter_manifest_pages(descriptor, store.read)) == [(ref, _entries(count))]


@pytest.mark.parametrize("count", [129, 130, 256, 300, 1000])
def test_multiple_pages_round_trip(count):
    descriptor, store = _build(count)
    expected_pages = (count + MANIFEST_PAGE_SIZE - 1) // MANIFEST_PAGE_SIZE

    assert descriptor["count"] == count
    assert len(store.writes) == expected_pages
    assert len(_page_refs(store)) == len(set(_page_refs(store)))

    for index, (ref, data) in enumerate(store.writes):
        page = json.loads(data)
        start = index * MANIFEST_PAGE_SIZE
        stop = min(start + MANIFEST_PAGE_SIZE, count)
        assert page["count"] == stop
        assert page["entries"] == _entries(count)[start:stop]
        assert page["parent"] == (None if index == 0 else store.writes[index - 1][0])
        assert ref == "seg_" + hashlib.sha256(data).hexdigest()

    entries = list(iter_manifest_entries(descriptor, store.read))
    assert entries == _entries(count)
    assert [entry["item_id"] for entry in entries] == [
        f"itm_{index:05d}" for index in range(count)
    ]

    pages = list(iter_manifest_pages(descriptor, store.read))
    assert [ref for ref, _entries_in_page in pages] == list(reversed(_page_refs(store)))
    assert [entries_in_page for _ref, entries_in_page in pages] == list(
        reversed(
            [
                _entries(count)[
                    index * MANIFEST_PAGE_SIZE : min(
                        (index + 1) * MANIFEST_PAGE_SIZE, count
                    )
                ]
                for index in range(expected_pages)
            ]
        )
    )


def test_page_bytes_are_canonical_and_contain_parent():
    descriptor, store = _build(300)

    for ref, data in store.writes:
        parsed = json.loads(data)
        canonical = json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        assert data == canonical
        assert ref == "seg_" + hashlib.sha256(data).hexdigest()

    assert json.loads(store.writes[0][1])["parent"] is None
    assert json.loads(store.writes[1][1])["parent"] == store.writes[0][0]
    assert descriptor["head"] == store.writes[-1][0]


def test_appended_histories_reuse_complete_prefix_pages():
    short_descriptor, short_store = _build(129)
    long_descriptor, long_store = _build(130)

    assert short_descriptor["head"] != long_descriptor["head"]
    assert short_store.writes[0][0] == long_store.writes[0][0]
    assert short_store.writes[0][1] == long_store.writes[0][1]
    assert short_store.writes[1][0] != long_store.writes[1][0]

    refs_400 = _page_refs(_build(400)[1])
    refs_500 = _page_refs(_build(500)[1])
    assert refs_400[:3] == refs_500[:3]
    assert refs_400[3] != refs_500[3]


def test_partial_tail_pages_are_not_shared_across_different_lengths():
    # Every non-final page is full, so a partial tail page is unique to the
    # manifest length that produced it.  Only complete pages are reusable.
    refs_by_count = {
        count: set(_page_refs(_build(count)[1])) for count in (200, 250, 300, 350, 400)
    }
    first_page_ref = set(_page_refs(_build(128)[1]))
    two_page_refs = set(_page_refs(_build(256)[1]))

    all_refs = set().union(*refs_by_count.values())
    assert len(all_refs) == 8
    assert refs_by_count[200] & refs_by_count[250] == first_page_ref
    assert refs_by_count[300] & refs_by_count[350] & refs_by_count[400] == two_page_refs


def test_divergent_histories_share_only_complete_prefix_pages():
    first_refs = set(_page_refs(_build(300)[1]))

    # Replace everything from entry 200 onward while keeping the same prefix.
    second_store = PageStore()
    second_entries = [*_entries(200), *_entries(100)]
    second_entries[200:] = [_entry(index, divergent=True) for index in range(200, 300)]
    build_manifest(second_entries, second_store.write)
    divergent_refs = set(_page_refs(second_store))

    assert len(first_refs) == 3
    assert len(divergent_refs) == 3
    assert len(first_refs | divergent_refs) == 5
    assert first_refs & divergent_refs == set(_page_refs(_build(128)[1]))


def test_page_digest_includes_parent_reference():
    parent_ref = _store_raw(
        PageStore(),
        json.dumps(
            {"parent": None, "count": 1, "entries": [_entry(0)]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    )
    entries = [_entry(1)]
    no_parent = json.dumps(
        {"parent": None, "count": 1, "entries": entries},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with_parent = json.dumps(
        {"parent": parent_ref, "count": 2, "entries": entries},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    assert (
        hashlib.sha256(no_parent).hexdigest() != hashlib.sha256(with_parent).hexdigest()
    )

    # Identical entries on different pages never collapse to one reference.
    store = PageStore()
    build_manifest([_entry(0)] * 256, store.write)
    assert store.writes[0][0] != store.writes[1][0]


def test_deterministic_refs_across_builds_and_key_order():
    first_store = PageStore()
    second_store = PageStore()
    first_entries = _entries(300, metadata={"nested": {"b": 2, "a": 1}})
    second_entries = [
        {
            "metadata": {"nested": {"a": 1, "b": 2}},
            "item_ref": entry["item_ref"],
            "item_id": entry["item_id"],
        }
        for entry in first_entries
    ]

    first = build_manifest(first_entries, first_store.write)
    second = build_manifest(second_entries, second_store.write)

    assert first == second
    assert first_store.writes == second_store.writes


def test_build_does_not_mutate_or_retain_input_entries():
    entries = _entries(2, metadata={"nested": {"values": [1, 2]}})
    snapshot = copy.deepcopy(entries)
    store = PageStore()

    descriptor = build_manifest(entries, store.write)
    assert entries == snapshot
    assert "entries" not in descriptor

    entries[0]["item_id"] = "mutated"
    entries[0]["metadata"]["nested"]["values"].append(3)
    assert entries != snapshot

    stored = list(iter_manifest_entries(descriptor, store.read))
    assert stored == snapshot


def test_build_accepts_a_one_shot_iterable():
    source = _entries(129)
    store = PageStore()

    descriptor = build_manifest(iter(source), store.write)

    assert descriptor["count"] == 129
    assert list(iter_manifest_entries(descriptor, store.read)) == source


@pytest.mark.parametrize(
    "bad_entry",
    [
        "not-a-mapping",
        {"item_id": "itm_1"},
        {"item_ref": "item_1"},
        {"item_id": "", "item_ref": "item_1"},
        {"item_id": "itm_1", "item_ref": ""},
        {"item_id": 1, "item_ref": "item_1"},
        {"item_id": "itm_1", "item_ref": 1},
        {"item_id": "itm_1", "item_ref": "item_1", "bad": object()},
        {"item_id": "itm_1", "item_ref": "item_1", "bad": (1, 2)},
        {"item_id": "itm_1", "item_ref": "item_1", "bad": {1: "x"}},
        {"item_id": "itm_1", "item_ref": "item_1", "bad": float("nan")},
        {"item_id": "itm_1", "item_ref": "item_1", "bad": float("inf")},
    ],
)
def test_build_rejects_invalid_entries(bad_entry):
    store = PageStore()

    with pytest.raises(ValueError):
        build_manifest([bad_entry], store.write)
    assert store.writes == []


def test_build_rejects_non_callable_writer():
    with pytest.raises(TypeError):
        build_manifest([], None)


def test_invalid_late_entry_leaves_earlier_pages_for_caller_rollback():
    store = PageStore()
    entries = [*_entries(128), {"item_id": "itm_x"}]

    with pytest.raises(ValueError):
        build_manifest(entries, store.write)

    assert len(store.writes) == 1
    assert json.loads(store.writes[0][1])["count"] == 128


@pytest.mark.parametrize(
    "descriptor",
    [
        None,
        [],
        "segments_v1",
        {},
        {"format": "segments_v0", "head": None, "count": 0},
        {"head": None, "count": 0},
        {"format": MANIFEST_FORMAT, "count": 0},
        {"format": MANIFEST_FORMAT, "head": None},
        {"format": MANIFEST_FORMAT, "head": None, "count": True},
        {"format": MANIFEST_FORMAT, "head": None, "count": False},
        {"format": MANIFEST_FORMAT, "head": None, "count": "0"},
        {"format": MANIFEST_FORMAT, "head": None, "count": 0.0},
        {"format": MANIFEST_FORMAT, "head": None, "count": -1},
        {"format": MANIFEST_FORMAT, "head": "seg_" + "a" * 64, "count": 0},
        {"format": MANIFEST_FORMAT, "head": None, "count": 1},
        {"format": MANIFEST_FORMAT, "head": 1, "count": 1},
        {"format": MANIFEST_FORMAT, "head": "bad", "count": 1},
        {"format": MANIFEST_FORMAT, "head": "seg_" + "A" * 64, "count": 1},
        {"format": MANIFEST_FORMAT, "head": "seg_" + "a" * 63, "count": 1},
        {"format": MANIFEST_FORMAT, "head": "seg_" + "a" * 65, "count": 1},
    ],
)
def test_descriptor_validation(descriptor):
    with pytest.raises(ValueError):
        list(iter_manifest_entries(descriptor, lambda ref: b""))


def test_descriptor_extra_keys_are_ignored():
    descriptor, store = _build(2)
    descriptor = dict(descriptor, page_size=MANIFEST_PAGE_SIZE)

    assert list(iter_manifest_entries(descriptor, store.read)) == _entries(2)


def test_missing_head_page_is_a_value_error():
    descriptor, store = _build(1)
    del store.pages[descriptor["head"]]

    with pytest.raises(ValueError, match="missing"):
        list(iter_manifest_entries(descriptor, store.read))


def test_missing_parent_page_is_a_value_error():
    descriptor, store = _build(129)
    del store.pages[store.writes[0][0]]

    with pytest.raises(ValueError, match="missing"):
        list(iter_manifest_entries(descriptor, store.read))
    with pytest.raises(ValueError, match="missing"):
        list(iter_manifest_pages(descriptor, store.read))


def test_digest_mismatch_is_rejected():
    descriptor, store = _build(129)
    store.pages[descriptor["head"]] = store.writes[0][1]

    with pytest.raises(ValueError, match="digest"):
        list(iter_manifest_entries(descriptor, store.read))


def test_digest_mismatch_in_parent_is_rejected_before_child_is_yielded():
    descriptor, store = _build(300)
    store.pages[store.writes[1][0]] = b"{}"

    iterator = iter_manifest_pages(descriptor, store.read)
    with pytest.raises(ValueError, match="digest"):
        next(iterator)


def test_read_page_accepts_bytes_like_objects():
    descriptor, store = _build(2)
    data = store.pages[descriptor["head"]]

    assert list(iter_manifest_entries(descriptor, lambda ref: bytearray(data))) == (
        _entries(2)
    )
    assert list(iter_manifest_entries(descriptor, lambda ref: memoryview(data))) == (
        _entries(2)
    )


@pytest.mark.parametrize("bad_value", [None, "text", 1, []])
def test_read_page_rejects_non_bytes(bad_value):
    descriptor, _store = _build(1)

    with pytest.raises(ValueError, match="not bytes"):
        list(iter_manifest_entries(descriptor, lambda ref: bad_value))


def test_read_page_rejects_invalid_utf8():
    store = PageStore()
    ref = _store_raw(store, b'{"parent":null,"count":1,"entries":[]}\xff')

    with pytest.raises(ValueError, match="UTF-8"):
        list(iter_manifest_entries(_descriptor(ref, 1), store.read))


def test_read_page_rejects_invalid_json():
    store = PageStore()
    ref = _store_raw(store, b"{not json")

    with pytest.raises(ValueError, match="JSON"):
        list(iter_manifest_entries(_descriptor(ref, 1), store.read))


def test_read_page_rejects_json_array():
    store = PageStore()
    ref = _store_raw(store, b"[]")

    with pytest.raises(ValueError, match="JSON object"):
        list(iter_manifest_entries(_descriptor(ref, 1), store.read))


def test_read_page_rejects_duplicate_json_keys():
    store = PageStore()
    ref = _store_raw(
        store,
        b'{"parent":null,"count":1,"entries":[{"item_id":"itm_1",'
        b'"item_ref":"item_1","meta":1,"meta":2}]}',
    )

    with pytest.raises(ValueError, match="duplicate"):
        list(iter_manifest_entries(_descriptor(ref, 1), store.read))


def test_read_page_rejects_non_finite_json_numbers():
    store = PageStore()
    ref = _store_raw(
        store,
        b'{"parent":null,"count":1,"entries":[{"item_id":"itm_1",'
        b'"item_ref":"item_1","score":NaN}]}',
    )

    with pytest.raises(ValueError, match="non-finite"):
        list(iter_manifest_entries(_descriptor(ref, 1), store.read))


@pytest.mark.parametrize(
    ("page", "count"),
    [
        ({"parent": None, "count": 1, "entries": [_entry(0)], "extra": 1}, 1),
        ({"count": 1, "entries": [_entry(0)]}, 1),
        ({"parent": None, "entries": [_entry(0)]}, 1),
        ({"parent": None, "count": 1}, 1),
        ({"parent": "bad", "count": 1, "entries": [_entry(0)]}, 1),
        ({"parent": 1, "count": 1, "entries": [_entry(0)]}, 1),
        ({"parent": None, "count": True, "entries": [_entry(0)]}, 1),
        ({"parent": None, "count": 0, "entries": [_entry(0)]}, 1),
        ({"parent": None, "count": -1, "entries": [_entry(0)]}, 1),
        ({"parent": None, "count": "1", "entries": [_entry(0)]}, 1),
        ({"parent": None, "count": 1, "entries": "bad"}, 1),
        ({"parent": None, "count": 1, "entries": []}, 1),
        ({"parent": None, "count": 129, "entries": _entries(129)}, 129),
        ({"parent": None, "count": 1, "entries": ["bad"]}, 1),
        ({"parent": None, "count": 1, "entries": [{"item_id": "itm_1"}]}, 1),
        ({"parent": None, "count": 1, "entries": [{"item_ref": "item_1"}]}, 1),
        (
            {"parent": None, "count": 1, "entries": [{"item_id": "", "item_ref": "r"}]},
            1,
        ),
        (
            {"parent": None, "count": 1, "entries": [{"item_id": "i", "item_ref": ""}]},
            1,
        ),
        (
            {
                "parent": None,
                "count": 1,
                "entries": [{"item_id": 1, "item_ref": "item_1"}],
            },
            1,
        ),
        (
            {
                "parent": None,
                "count": 1,
                "entries": [{"item_id": "itm_1", "item_ref": 1}],
            },
            1,
        ),
    ],
)
def test_page_shape_corruption_is_rejected(page, count):
    store = PageStore()
    ref = _store_raw(
        store,
        json.dumps(
            page,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    )

    with pytest.raises(ValueError):
        list(iter_manifest_entries(_descriptor(ref, count), store.read))


def test_head_count_must_match_descriptor():
    descriptor, store = _build(129)
    bad_descriptor = dict(descriptor, count=130)

    with pytest.raises(ValueError, match="head count"):
        list(iter_manifest_entries(bad_descriptor, store.read))


def test_parent_count_must_be_cumulative():
    store = PageStore()
    first = _store_page(store, parent=None, count=128, entries=_entries(128))
    second = _store_page(store, parent=first, count=200, entries=_entries(2))
    descriptor = _descriptor(second, 200)

    with pytest.raises(ValueError, match="cumulative"):
        list(iter_manifest_entries(descriptor, store.read))


def test_parent_count_cannot_be_larger_than_child_count():
    store = PageStore()
    first = _store_page(store, parent=None, count=128, entries=_entries(128))
    second = _store_page(store, parent=first, count=127, entries=_entries(1))
    descriptor = _descriptor(second, 127)

    with pytest.raises(ValueError, match="cumulative"):
        list(iter_manifest_entries(descriptor, store.read))


def test_oldest_page_count_must_match_its_entry_count():
    store = PageStore()
    ref = _store_page(store, parent=None, count=2, entries=_entries(1))

    with pytest.raises(ValueError, match="oldest page count"):
        list(iter_manifest_entries(_descriptor(ref, 2), store.read))


def test_corrupt_tail_fails_before_any_entry_is_yielded():
    descriptor, store = _build(300)
    oldest_ref = store.writes[0][0]
    store.pages[oldest_ref] = b"{}"

    iterator = iter_manifest_entries(descriptor, store.read)
    with pytest.raises(ValueError):
        next(iterator)


def test_cyclic_page_chain_is_rejected(monkeypatch):
    first_ref = "seg_" + "a" * 64
    second_ref = "seg_" + "b" * 64
    pages = {
        first_ref: {
            "parent": second_ref,
            "count": 2,
            "entries": [_entry(0)],
        },
        second_ref: {
            "parent": first_ref,
            "count": 1,
            "entries": [_entry(1)],
        },
    }
    monkeypatch.setattr(
        _manifest,
        "_read_page",
        lambda ref, read_page: pages[ref],
    )

    with pytest.raises(ValueError, match="cyclic"):
        list(iter_manifest_pages(_descriptor(first_ref, 2), lambda ref: b""))


def test_self_parent_page_is_rejected(monkeypatch):
    ref = "seg_" + "a" * 64
    pages = {
        ref: {
            "parent": ref,
            "count": 1,
            "entries": [_entry(0)],
        }
    }
    monkeypatch.setattr(
        _manifest,
        "_read_page",
        lambda page_ref, read_page: pages[page_ref],
    )

    with pytest.raises(ValueError, match="cyclic"):
        list(iter_manifest_pages(_descriptor(ref, 1), lambda page_ref: b""))


def test_entries_are_isolated_between_reads():
    descriptor, store = _build(
        300,
        metadata={"nested": {"values": [1, 2]}},
    )

    first = list(iter_manifest_entries(descriptor, store.read))
    second = list(iter_manifest_entries(descriptor, store.read))

    assert first == second
    assert first[0] is not second[0]
    first[0]["item_id"] = "mutated"
    first[0]["metadata"]["nested"]["values"].append(3)
    assert second[0]["item_id"] == "itm_00000"
    assert second[0]["metadata"] == {"nested": {"values": [1, 2]}}

    third = list(iter_manifest_entries(descriptor, store.read))
    assert third == second


def test_pages_are_isolated_between_reads():
    descriptor, store = _build(300, metadata={"nested": {"values": [1, 2]}})

    first = list(iter_manifest_pages(descriptor, store.read))
    second = list(iter_manifest_pages(descriptor, store.read))

    assert first == second
    assert first[0][1][0] is not second[0][1][0]
    first[0][1][0]["item_id"] = "mutated"
    first[0][1][0]["metadata"]["nested"]["values"].append(3)
    assert second[0][1][0]["item_id"] == "itm_00256"
    assert second[0][1][0]["metadata"] == {"nested": {"values": [1, 2]}}


def test_iteration_reads_each_page_a_bounded_number_of_times():
    descriptor, store = _build(1000)
    page_count = len(store.writes)
    reads = 0

    def read(ref):
        nonlocal reads
        reads += 1
        return store.pages[ref]

    entries = list(iter_manifest_entries(descriptor, read))
    assert len(entries) == 1000
    assert reads <= 2 * page_count

    reads = 0
    pages = list(iter_manifest_pages(descriptor, read))
    assert len(pages) == page_count
    assert reads == page_count


def test_non_sequential_ids_and_duplicate_ids_are_preserved():
    entries = [
        {"item_id": "itm_z", "item_ref": "item_z"},
        {"item_id": "itm_a", "item_ref": "item_a"},
        {"item_id": "itm_z", "item_ref": "item_z_2"},
    ]
    store = PageStore()

    descriptor = build_manifest(entries, store.write)

    assert list(iter_manifest_entries(descriptor, store.read)) == entries


def test_entry_metadata_is_preserved_exactly():
    entries = [
        _entry(
            0,
            metadata={
                "nested": [1, None, True, {"b": 2.5, "a": "text"}],
                "empty": {},
                "unicode": "olá 🚀",
            },
        )
    ]
    store = PageStore()

    descriptor = build_manifest(entries, store.write)

    assert list(iter_manifest_entries(descriptor, store.read)) == entries
