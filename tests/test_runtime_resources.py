"""Runtime layout and portable references in the existing checkpoint adapter."""

import shutil

import msgspec
import pytest

from msgflux.runtime import RuntimeResources, ToolResultRef


def test_layout_is_explicit_and_checkpoint_factory_is_lazy(tmp_path):
    resources = RuntimeResources(tmp_path / "app")
    assert not resources.root.exists()
    resources.initialize()
    assert {p.name for p in resources.root.iterdir()} == {"threads", "tool-results"}
    assert not resources.plans_path.exists()
    store = resources.checkpoint_store("thd_main")
    try:
        # Parent and child use the same thread/database, distinct namespace/run.
        store.save_state("main", "thd_main", "parent", {"status": "running"})
        store.save_state("child", "thd_main", "child_run", {"parent_run_id": "parent"})
        assert (
            store.load_state("child", "thd_main", "child_run")["parent_run_id"]
            == "parent"
        )
        assert store.load_state("main", "thd_main", "parent")["status"] == "running"
    finally:
        store.close()
    assert (resources.root / "threads/thd_main/checkpoint.sqlite").is_file()


def test_optional_directories_are_explicit_and_initialization_is_idempotent(tmp_path):
    resources = RuntimeResources(tmp_path / "app")
    assert resources.initialize(extra_dirs=("plans", "skills", "skills")) is resources
    resources.initialize()
    assert {p.name for p in resources.root.iterdir()} == {
        "plans",
        "skills",
        "threads",
        "tool-results",
    }


@pytest.mark.parametrize(
    "names",
    [("skills", "../escape"), ("/absolute",), ("a/b",), ("..",), (None,), "skills"],
)
def test_invalid_extra_directories_have_no_side_effects(tmp_path, names):
    resources = RuntimeResources(tmp_path / "absent")
    with pytest.raises((ValueError, TypeError)):
        resources.initialize(extra_dirs=names)
    assert not resources.root.exists()


@pytest.mark.parametrize("thread_id", ["../escape", "/absolute", "", "a/b", ".", None])
def test_invalid_thread_paths_have_no_side_effects(tmp_path, thread_id):
    resources = RuntimeResources(tmp_path / "absent")
    with pytest.raises(ValueError):
        resources.checkpoint_store(thread_id)
    assert not resources.root.exists()


def test_checkpoint_and_result_survive_root_relocation(tmp_path):
    original = RuntimeResources(tmp_path / "original").initialize()
    results = original.tool_result_store()
    ref = results.put([b"persisted ", b"tool output"], media_type="text/plain")
    checkpoint = original.checkpoint_store("thd_main")
    try:
        checkpoint.save_state(
            "main",
            "thd_main",
            "run_1",
            {"result": ref.to_dict(), "tool_call_id": "call_1"},
        )
    finally:
        checkpoint.close()
    # Copy only closed SQLite storage; live backup/WAL coordination is separate.
    shutil.copytree(original.root, tmp_path / "moved")
    relocated = RuntimeResources(tmp_path / "moved")
    reopened = relocated.checkpoint_store("thd_main")
    try:
        state = reopened.load_state("main", "thd_main", "run_1")
    finally:
        reopened.close()
    restored = msgspec.convert(state["result"], type=ToolResultRef)
    assert restored == ref
    target = relocated.tool_result_store()
    target.verify(restored)
    assert target.read(restored) == b"persisted tool output"
