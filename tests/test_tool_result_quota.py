import multiprocessing

import pytest

from msgflux.runtime import LocalToolResultStore, ToolResultQuotaError


def _write_with_quota(root, start, outcomes):
    store = LocalToolResultStore(root, max_store_bytes=1000)
    start.wait(5)
    try:
        store.put([b"x" * 700])
        outcomes.put("ok")
    except ToolResultQuotaError:
        outcomes.put("quota")


def test_quota_serializes_real_process_writers(tmp_path):
    context = multiprocessing.get_context("spawn")
    start, outcomes = context.Event(), context.Queue()
    processes = [
        context.Process(target=_write_with_quota, args=(str(tmp_path), start, outcomes))
        for _ in range(2)
    ]
    try:
        for process in processes:
            process.start()
        start.set()
        assert sorted(outcomes.get(timeout=10) for _ in processes) == ["ok", "quota"]
        for process in processes:
            process.join(10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join(5)
        outcomes.close()
    usage = LocalToolResultStore(tmp_path).usage()
    assert usage.results == 1 and usage.pending == 0 and usage.size_bytes <= 1000


def test_offline_gc_preserves_references_and_reclaims_interrupted_writes(tmp_path):
    store = LocalToolResultStore(tmp_path)
    retained = store.put([b"must survive"])
    orphan = store.put([b"orphan"])
    pending = tmp_path / (".pending-" + "a" * 32)
    pending.mkdir()
    (pending / "content").write_bytes(b"partial")
    before = store.usage()
    assert before.results == 2 and before.pending == 1
    with pytest.raises(ValueError, match="quiescent"):
        store.collect_garbage([retained])
    candidates = store.collect_garbage([retained], quiescent=True)
    assert set(candidates) == {orphan.result_id, pending.name}
    assert store.get(orphan.result_id) == orphan
    assert (
        store.collect_garbage([retained], quiescent=True, dry_run=False) == candidates
    )
    store.verify(retained)
    assert store.read(retained) == b"must survive"
    assert store.usage().size_bytes < before.size_bytes
    assert store.usage().results == 1 and store.usage().pending == 0


def test_interrupted_bytes_count_toward_quota(tmp_path):
    pending = tmp_path / (".pending-" + "b" * 32)
    pending.mkdir()
    (pending / "content").write_bytes(b"x" * 1000)
    store = LocalToolResultStore(tmp_path, max_store_bytes=1000)
    with pytest.raises(ToolResultQuotaError):
        store.put([b"small"])
    assert store.usage().size_bytes == 1000


def test_missing_retained_reference_prevents_all_deletions(tmp_path):
    store = LocalToolResultStore(tmp_path / "one")
    retained = LocalToolResultStore(tmp_path / "other").put([b"other"])
    orphan = store.put([b"keep until valid inventory"])
    with pytest.raises(FileNotFoundError):
        store.collect_garbage([retained], quiescent=True, dry_run=False)
    store.verify(orphan)


def test_sqlite_reference_survives_cleanup_and_reopen(tmp_path):
    import msgspec
    from msgflux.runtime import RuntimeResources, ToolResultRef

    resources = RuntimeResources(tmp_path).initialize()
    store = resources.tool_result_store()
    reference = store.put([b"durable"])
    orphan = store.put([b"unreferenced"])
    checkpoint = resources.checkpoint_store("thread")
    checkpoint.save_state("agent", "thread", "run", {"result": reference.to_dict()})
    checkpoint.close()
    checkpoint = resources.checkpoint_store("thread")
    state = checkpoint.load_state("agent", "thread", "run")
    checkpoint.close()
    retained = msgspec.convert(state["result"], type=ToolResultRef)
    removed = store.collect_garbage([retained], quiescent=True, dry_run=False)
    assert removed == (orphan.result_id,)
    reopened = resources.tool_result_store()
    assert reopened.read(retained) == b"durable"
