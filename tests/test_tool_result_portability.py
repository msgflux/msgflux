"""Portable checkpoint/result bundles and publication crash boundaries."""

import multiprocessing
import os
import shutil

import msgspec
import pytest

from msgflux.runtime import RuntimeResources, ToolResultIntegrityError, ToolResultRef


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX local store")


def _crash_before_publication(root: str) -> None:
    store = RuntimeResources(root).tool_result_store()

    def chunks():
        yield b"partial output"
        os._exit(23)

    store.put(chunks())


def _crash_after_publication(root: str) -> None:
    store = RuntimeResources(root).tool_result_store()
    store.put([b"published output"])
    os._exit(24)


def _crash_between_publication_and_checkpoint(root: str) -> None:
    resources = RuntimeResources(root).initialize()
    resources.tool_result_store().put([b"orphaned output"])
    os._exit(25)


def _crash_after_checkpoint(root: str) -> None:
    resources = RuntimeResources(root).initialize()
    reference = resources.tool_result_store().put([b"checkpointed output"])
    checkpoint = resources.checkpoint_store("thread_main")
    checkpoint.save_state(
        "agent",
        "thread_main",
        "run_main",
        {"status": "completed", "result": reference.to_dict()},
    )
    os._exit(26)


def _run_crashing_writer(target, root) -> int:
    process = multiprocessing.get_context("spawn").Process(
        target=target, args=(str(root),)
    )
    process.start()
    process.join(timeout=20)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
    return process.exitcode


def _make_closed_bundle(root):
    resources = RuntimeResources(root).initialize()
    result_store = resources.tool_result_store()
    reference = result_store.put(
        [b"portable ", b"tool result"], media_type="text/plain"
    )
    checkpoint = resources.checkpoint_store("thread_main")
    checkpoint.save_state(
        "agent",
        "thread_main",
        "run_main",
        {"status": "completed", "result": reference.to_dict()},
    )
    checkpoint.close()
    return reference


def test_closed_runtime_bundle_relocates_with_original_reference(tmp_path):
    reference = _make_closed_bundle(tmp_path / "origin")
    shutil.copytree(tmp_path / "origin", tmp_path / "relocated")

    relocated = RuntimeResources(tmp_path / "relocated")
    checkpoint = relocated.checkpoint_store("thread_main")
    try:
        state = checkpoint.load_state("agent", "thread_main", "run_main")
    finally:
        checkpoint.close()

    restored = msgspec.convert(state["result"], type=ToolResultRef)
    assert restored == reference
    assert restored.uri == reference.uri
    results = relocated.tool_result_store()
    results.verify(restored)
    assert results.read(restored) == b"portable tool result"


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_relocated_checkpoint_exposes_missing_or_corrupt_result(tmp_path, damage):
    reference = _make_closed_bundle(tmp_path / "origin")
    shutil.copytree(tmp_path / "origin", tmp_path / "relocated")
    result_path = tmp_path / "relocated" / "tool-results" / reference.result_id
    if damage == "missing":
        shutil.rmtree(result_path)
    else:
        (result_path / "content").write_bytes(b"tampered")

    relocated = RuntimeResources(tmp_path / "relocated")
    checkpoint = relocated.checkpoint_store("thread_main")
    try:
        state = checkpoint.load_state("agent", "thread_main", "run_main")
    finally:
        checkpoint.close()
    restored = msgspec.convert(state["result"], type=ToolResultRef)
    with pytest.raises(
        (FileNotFoundError, ToolResultIntegrityError),
    ):
        relocated.tool_result_store().verify(restored)


def test_spawned_death_before_publication_leaves_no_published_reference(tmp_path):
    root = tmp_path / "before"
    assert _run_crashing_writer(_crash_before_publication, root) == 23
    entries = list((root / "tool-results").iterdir())
    assert entries and all(path.name.startswith(".pending-") for path in entries)
    assert not any(path.name.startswith("res_") for path in entries)


def test_spawned_death_after_publication_preserves_complete_reference(tmp_path):
    root = tmp_path / "after"
    assert _run_crashing_writer(_crash_after_publication, root) == 24
    entries = [
        path
        for path in (root / "tool-results").iterdir()
        if path.name.startswith("res_")
    ]
    assert len(entries) == 1
    store = RuntimeResources(root).tool_result_store()
    reference = store.get(entries[0].name)
    store.verify(reference)
    assert store.read(reference) == b"published output"


def test_spawned_death_between_publication_and_checkpoint_leaves_orphan(tmp_path):
    root = tmp_path / "between"
    assert _run_crashing_writer(_crash_between_publication_and_checkpoint, root) == 25

    result_entries = [
        path
        for path in (root / "tool-results").iterdir()
        if path.name.startswith("res_")
    ]
    assert len(result_entries) == 1
    checkpoint = RuntimeResources(root).checkpoint_store("thread_main")
    try:
        assert checkpoint.load_state("agent", "thread_main", "run_main") is None
    finally:
        checkpoint.close()


def test_spawned_death_after_checkpoint_keeps_reference_resolvable(tmp_path):
    root = tmp_path / "checkpointed"
    assert _run_crashing_writer(_crash_after_checkpoint, root) == 26

    checkpoint = RuntimeResources(root).checkpoint_store("thread_main")
    try:
        state = checkpoint.load_state("agent", "thread_main", "run_main")
    finally:
        checkpoint.close()
    reference = msgspec.convert(state["result"], type=ToolResultRef)
    results = RuntimeResources(root).tool_result_store()
    results.verify(reference)
    assert results.read(reference) == b"checkpointed output"
