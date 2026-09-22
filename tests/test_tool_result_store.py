"""Incremental durable result storage without a model or workspace grant."""

import hashlib
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import msgspec
import pytest

from msgflux.runtime import (
    LocalToolResultStore,
    ToolResultIntegrityError,
    ToolResultRef,
    ToolResultTooLargeError,
)

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX local store")


def _crash_writer(root):
    store = LocalToolResultStore(root)

    def chunks():
        yield b"partial"
        os._exit(23)

    store.put(chunks())


def test_incremental_roundtrip_and_byte_ranges(tmp_path):
    store = LocalToolResultStore(tmp_path)
    payload = "olá 🌍\n".encode()
    ref = store.put(
        (payload[i : i + 2] for i in range(0, len(payload), 2)), media_type="text/plain"
    )
    assert ref.sha256 == hashlib.sha256(payload).hexdigest()
    assert ref.size_bytes == len(payload)
    assert ref.uri == f"tool-result://{ref.result_id}"
    assert store.get(ref.result_id) == ref
    assert store.read(ref, offset=2, limit=4) == payload[2:6]
    assert store.read(ref, offset=100) == b""
    assert b"".join(store.iter_bytes(ref, chunk_size=3)) == payload
    assert max(map(len, store.iter_bytes(ref, chunk_size=3))) <= 3
    assert msgspec.json.decode(msgspec.json.encode(ref), type=ToolResultRef) == ref
    assert str(tmp_path) not in str(ref.to_dict())
    store.verify(ref)
    reopened = LocalToolResultStore(tmp_path)
    assert reopened.read(ref) == payload


def test_empty_output_is_a_valid_immutable_result(tmp_path):
    store = LocalToolResultStore(tmp_path)
    ref = store.put([])
    assert ref.size_bytes == 0
    assert store.read(ref) == b""
    store.verify(ref)
    with pytest.raises(AttributeError):
        ref.size_bytes = 10


@pytest.mark.parametrize("result_id", ["../x", "/absolute/x", "res_", "a/b", "", None])
def test_invalid_ids_are_rejected(tmp_path, result_id):
    with pytest.raises(ValueError, match="ID"):
        LocalToolResultStore(tmp_path).get(result_id)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budgets_do_not_create_directories(tmp_path, budget):
    root = tmp_path / "absent"
    with pytest.raises(ValueError):
        LocalToolResultStore(root, max_result_bytes=budget)
    assert not root.exists()


def test_failed_generator_and_quota_never_publish_partial_results(tmp_path):
    store = LocalToolResultStore(tmp_path, max_result_bytes=3)

    def broken():
        yield b"ok"
        raise RuntimeError("producer failed")

    with pytest.raises(RuntimeError, match="producer failed"):
        store.put(broken())
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ToolResultTooLargeError):
        store.put([b"ok", b"no"])
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(TypeError, match="bytes"):
        store.put(["not bytes"])
    assert list(tmp_path.iterdir()) == []
    assert store.put([b"123"]).size_bytes == 3


def test_forced_id_collision_cannot_replace_existing_result(tmp_path, monkeypatch):
    monkeypatch.setattr("msgflux.runtime.tool_results.uuid4", lambda: UUID(int=1))
    store = LocalToolResultStore(tmp_path)
    ref = store.put([b"original"])
    with pytest.raises(OSError):
        store.put([b"replacement"])
    assert store.read(ref) == b"original"
    assert len(list(tmp_path.iterdir())) == 1


def test_concurrent_writers_have_independent_ids(tmp_path):
    store = LocalToolResultStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        refs = list(pool.map(lambda i: store.put([str(i).encode()]), range(20)))
    assert len({ref.result_id for ref in refs}) == 20
    assert [store.read(ref) for ref in refs] == [str(i).encode() for i in range(20)]


def test_fsync_failure_does_not_publish_a_reference(tmp_path, monkeypatch):
    store = LocalToolResultStore(tmp_path)

    def fail(_fd):
        raise OSError("storage sync failed")

    monkeypatch.setattr("msgflux.runtime.tool_results.os.fsync", fail)
    with pytest.raises(OSError, match="storage sync failed"):
        store.put([b"never published"])
    assert list(tmp_path.iterdir()) == []


def test_cleanup_failure_preserves_original_producer_error(tmp_path, monkeypatch):
    store = LocalToolResultStore(tmp_path)

    def fail(*args, **kwargs):
        raise PermissionError("cleanup denied")

    with monkeypatch.context() as patch:
        patch.setattr("msgflux.runtime.tool_results.os.unlink", fail)
        with pytest.raises(TypeError, match="must be bytes") as caught:
            store.put(["invalid"])
    assert any("cleanup denied" in note for note in caught.value.__notes__)
    assert all(path.name.startswith(".pending-") for path in tmp_path.iterdir())


def test_replaced_root_is_rejected_by_an_existing_store(tmp_path):
    root = tmp_path / "root"
    store = LocalToolResultStore(root)
    root.rename(tmp_path / "original")
    root.mkdir()
    with pytest.raises(PermissionError, match="replaced"):
        store.put([b"not written into replacement"])
    assert list(root.iterdir()) == []


def test_result_directory_symlink_is_rejected(tmp_path):
    store = LocalToolResultStore(tmp_path)
    ref = store.put([b"abc"])
    alias = "res_" + "1" * 32
    (tmp_path / alias).symlink_to(tmp_path / ref.result_id, target_is_directory=True)
    with pytest.raises(OSError):
        store.get(alias)


def test_missing_and_corrupt_content_are_not_empty_results(tmp_path):
    store = LocalToolResultStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get("res_" + "0" * 32)
    ref = store.put([b"original"])
    path = tmp_path / ref.result_id / "content"
    path.write_bytes(b"modified")
    with pytest.raises(ToolResultIntegrityError, match="content"):
        store.verify(ref)
    path.write_bytes(b"short")
    with pytest.raises(ToolResultIntegrityError, match="size"):
        store.read(ref)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        store.read(ref)


def test_corrupt_metadata_and_reference_mismatch(tmp_path):
    store = LocalToolResultStore(tmp_path)
    ref = store.put([b"abc"])
    altered = msgspec.structs.replace(ref, sha256="0" * 64)
    with pytest.raises(ToolResultIntegrityError, match="metadata"):
        store.read(altered)
    metadata = tmp_path / ref.result_id / "metadata.json"
    metadata.write_bytes(b"bad JSON")
    with pytest.raises(ToolResultIntegrityError, match="metadata"):
        store.get(ref.result_id)
    metadata.write_bytes(b"x" * 4097)
    with pytest.raises(ToolResultIntegrityError, match="too large"):
        store.get(ref.result_id)


@pytest.mark.parametrize("kind", ["symlink", "fifo", "hardlink"])
def test_nonregular_and_linked_content_is_rejected(tmp_path, kind):
    store = LocalToolResultStore(tmp_path)
    ref = store.put([b"abc"])
    path = tmp_path / ref.result_id / "content"
    if kind == "hardlink":
        os.link(path, tmp_path / "other-link")
    else:
        path.unlink()
        if kind == "symlink":
            path.symlink_to(tmp_path / ref.result_id / "metadata.json")
        else:
            os.mkfifo(path)
    with pytest.raises(OSError):
        store.read(ref)


def test_abrupt_process_death_does_not_publish_a_partial_reference(tmp_path):
    process = multiprocessing.get_context("spawn").Process(
        target=_crash_writer, args=(str(tmp_path),)
    )
    process.start()
    try:
        process.join(timeout=20)
        assert process.exitcode == 23
        entries = list(tmp_path.iterdir())
        assert entries and all(path.name.startswith(".pending-") for path in entries)
        store = LocalToolResultStore(tmp_path)
        ref = store.put([b"complete after restart"])
        store.verify(ref)
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


@pytest.mark.parametrize(
    "kwargs", [{"offset": -1}, {"offset": True}, {"limit": 0}, {"chunk_size": 0}]
)
def test_invalid_read_ranges(tmp_path, kwargs):
    store = LocalToolResultStore(tmp_path)
    ref = store.put([b"abc"])
    with pytest.raises(ValueError):
        list(store.iter_bytes(ref, **kwargs))
