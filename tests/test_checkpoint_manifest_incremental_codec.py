from copy import deepcopy

import pytest

from msgflux.data.stores._manifest import (
    append_manifest,
    build_manifest,
    iter_manifest_entries,
    iter_manifest_suffix,
)


def entries(n, start=0):
    return [
        {"item_id": f"itm_{i}", "item_ref": f"ref_{i}", "extra": {"i": [i]}}
        for i in range(start, start + n)
    ]


@pytest.mark.parametrize("size", [0, 1, 127, 128, 129, 255, 256, 1025])
@pytest.mark.parametrize("limit", [0, 1, 127, 128, 129, 2048])
def test_suffix_matches_full_manifest_and_bounds_reads(size, limit):
    pages = {}
    original = entries(size)
    descriptor = build_manifest(original, pages.__setitem__)
    reads = []

    def read(ref):
        reads.append(ref)
        return pages[ref]

    start = max(0, size - limit)
    assert list(iter_manifest_suffix(descriptor, read, start=start)) == original[start:]
    assert len(reads) <= (min(size, limit) + 127) // 128 + 2
    if limit == 0:
        assert not reads


@pytest.mark.parametrize("size", [0, 1, 127, 128, 129, 255, 256, 1025])
@pytest.mark.parametrize("added", [0, 1, 127, 128, 129, 300])
def test_append_matches_rebuild_and_preserves_base(size, added):
    pages = {}
    old = entries(size)
    tail = entries(added, size)
    original = deepcopy(tail)
    descriptor = build_manifest(old, pages.__setitem__)
    writes = []

    def write(ref, data):
        writes.append(ref)
        pages[ref] = data

    updated = append_manifest(descriptor, tail, pages.__getitem__, write)
    assert list(iter_manifest_entries(updated, pages.__getitem__)) == old + tail
    assert list(iter_manifest_entries(descriptor, pages.__getitem__)) == old
    expected = build_manifest(old + tail, lambda *_: None)
    assert updated == expected
    assert tail == original
    assert len(writes) <= (added + 127) // 128 + 1
    if added == 0:
        assert not writes


@pytest.mark.parametrize("start", [-1, 4, True, 1.5, None])
def test_invalid_suffix_start(start):
    pages = {}
    descriptor = build_manifest(entries(3), pages.__setitem__)
    with pytest.raises(ValueError):
        list(iter_manifest_suffix(descriptor, pages.__getitem__, start=start))


def test_corrupt_head_rejects_suffix_and_append():
    pages = {}
    descriptor = build_manifest(entries(300), pages.__setitem__)
    pages[descriptor["head"]] = b"corrupt"
    with pytest.raises(ValueError, match="digest"):
        list(iter_manifest_suffix(descriptor, pages.__getitem__, start=299))
    with pytest.raises(ValueError, match="digest"):
        append_manifest(descriptor, entries(1, 300), pages.__getitem__, lambda *_: None)
