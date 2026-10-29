"""Full commits carry atomic optional provenance for subsequent append."""

from copy import deepcopy

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import (
    CheckpointCommit,
    CheckpointConflictError,
    SQLiteCheckpointStore,
)

NS, THREAD, RUN = "commit-provenance", "thread", "run"


def _state(count):
    return {
        "status": "running",
        "messages": ChatMessages(
            [{"role": "user", "content": str(i)} for i in range(count)],
            namespace=NS,
            thread_id=THREAD,
        )._to_state(),
    }


@pytest.mark.parametrize("count", [0, 2, 129, 300])
def test_full_commit_base_matches_exact_persisted_snapshot(checkpoint_store, count):
    store = checkpoint_store
    state = _state(count)
    original = deepcopy(state)
    committed = store.commit_state(NS, THREAD, RUN, state, expected_revision=0)
    assert committed.base is not None
    assert committed.base == store.load_tail(NS, THREAD, RUN, limit=0).base
    assert committed.base.revision == committed.revision == 1
    assert committed.base.item_count == count
    assert state == original
    assert store.load_state(NS, THREAD, RUN) == committed.state
    assert "base" not in committed.state


def test_non_message_commit_keeps_legacy_full_behavior(checkpoint_store):
    committed = checkpoint_store.commit_state(
        NS, THREAD, RUN, {"status": "running"}, expected_revision=0
    )
    assert committed.base is None
    assert checkpoint_store.load_state(NS, THREAD, RUN) == committed.state


def test_non_json_control_commit_keeps_legacy_full_behavior(checkpoint_store):
    state = _state(1)
    state["opaque"] = b"kept by the original full API"
    committed = checkpoint_store.commit_state(
        NS, THREAD, RUN, state, expected_revision=0
    )
    assert committed.base is None
    assert committed.state["opaque"] == state["opaque"]
    assert checkpoint_store.load_state(NS, THREAD, RUN)["status"] == "running"


def test_auxiliary_base_preserves_commit_equality_and_repr(checkpoint_store):
    committed = checkpoint_store.commit_state(
        NS, THREAD, RUN, _state(1), expected_revision=0
    )
    old_style = CheckpointCommit(
        committed.revision, committed.state, committed.branch_id, committed.head_item_id
    )
    assert old_style.base is None
    assert committed == old_style
    assert repr(committed) == repr(old_style)


def test_commit_base_cannot_adopt_later_same_revision_writer(tmp_path):
    path = str(tmp_path / "atomic-provenance.db")
    first, second = SQLiteCheckpointStore(path), SQLiteCheckpointStore(path)
    try:
        committed = first.commit_state(NS, THREAD, RUN, _state(1), expected_revision=0)
        replacement = second.load_state(NS, THREAD, RUN)
        replacement["messages"]["items"][0]["content"] = "other writer"
        second.save_state(NS, THREAD, RUN, replacement)
        fresh = first.load_tail(NS, THREAD, RUN, limit=0)
        assert fresh.base.revision == committed.base.revision
        assert fresh.base != committed.base
        with pytest.raises(CheckpointConflictError):
            first.append_state(
                NS,
                THREAD,
                RUN,
                [{"role": "assistant", "content": "append", "item_id": "itm_new"}],
                base=committed.base,
            )
        assert first.load_state(NS, THREAD, RUN) == replacement
    finally:
        first.close()
        second.close()


def test_automatic_append_eligibility_honors_instance_commit_wrappers(
    checkpoint_store, monkeypatch
):
    store = checkpoint_store
    assert store.supports_agent_incremental is True
    original = store.commit_state

    def wrapped(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "commit_state", wrapped)
    assert store.supports_agent_incremental is False
    monkeypatch.setattr(store, "commit_state", original)
    assert store.supports_agent_incremental is True

    async def async_commit(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "acommit_state", async_commit, raising=False)
    assert store.supports_agent_incremental is False


def test_automatic_append_eligibility_honors_subclass_commit_hooks(
    checkpoint_store, tmp_path
):
    class CustomStore(type(checkpoint_store)):
        def commit_state(self, *args, **kwargs):
            return super().commit_state(*args, **kwargs)

    store = (
        CustomStore(str(tmp_path / "custom.db"))
        if isinstance(checkpoint_store, SQLiteCheckpointStore)
        else CustomStore()
    )
    try:
        assert store.supports_incremental is True
        assert store.supports_agent_incremental is False
        assert (
            store.commit_state(NS, THREAD, RUN, _state(1), expected_revision=0).base
            is not None
        )
    finally:
        if isinstance(store, SQLiteCheckpointStore):
            store.close()


class _OpaqueControlMapping:
    """Full serializers accept opaque controls through their existing fallback."""

    def __str__(self):
        return "legacy opaque control"


def test_optional_provenance_does_not_narrow_opaque_full_controls(checkpoint_store):
    from collections.abc import Mapping

    iterations = []

    class LegacyControl(_OpaqueControlMapping, Mapping):
        def __iter__(self):
            iterations.append(True)
            raise RuntimeError("this control intentionally has no JSON iteration")

        def __len__(self):
            return 0

        def __getitem__(self, key):
            raise KeyError(key)

    state = _state(1)
    state["opaque"] = LegacyControl()
    committed = checkpoint_store.commit_state(
        NS, THREAD, RUN, state, expected_revision=0
    )
    assert committed.base is None
    assert committed.state["status"] == "running"
    assert checkpoint_store.load_state(NS, THREAD, RUN)["status"] == "running"
    assert iterations == []


def test_automatic_append_eligibility_honors_class_level_commit_wrappers(
    checkpoint_store, monkeypatch
):
    store = checkpoint_store
    original = type(store).commit_state
    calls = []

    def class_wrapper(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    assert store.supports_agent_incremental is True
    monkeypatch.setattr(type(store), "commit_state", class_wrapper)
    assert store.supports_agent_incremental is False
    store.commit_state(NS, THREAD, RUN, _state(1), expected_revision=0)
    assert calls == [True]
