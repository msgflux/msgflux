"""Adversarial Agent write selection and coding-receipt integration."""

import asyncio
from contextlib import contextmanager
from copy import deepcopy

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import CheckpointConflictError
from msgflux.nn import Agent
from msgflux.nn.modules.agent.context import _persist_command_receipt
from msgflux.runtime.agent_run import AgentRun, agent_run_context
from msgflux.runtime.context import ExecutionScope, execution_context
from msgflux.runtime.workspace.receipts import new_command_receipt

NS, THREAD, RUN = "incremental_integrity", "thread", "run"


class _UnusedModel:
    model_type = "chat_completion"

    def __call__(self, **kwargs):
        raise AssertionError("Checkpoint integrity tests must not invoke the model")


@contextmanager
def _checkpoint_context(store):
    agent = Agent(name=NS, model=_UnusedModel(), checkpoint_store=store)
    messages = ChatMessages(thread_id=THREAD, namespace=NS)
    messages.begin_turn(turn_id=RUN)
    messages.add_user("work")
    run = AgentRun(namespace=NS, thread_id=THREAD, run_id=RUN)
    scope = ExecutionScope(namespace=NS, thread_id=THREAD, run_id=RUN)
    try:
        with (
            execution_context(scope=scope, checkpoint_store=store),
            agent_run_context(run),
        ):
            yield agent, messages, run
    finally:
        asyncio.run(agent.aclose())


def _save(agent, messages, *, asynchronous=False, status="running"):
    if asynchronous:
        return asyncio.run(agent._acheckpoint_save(messages, {}, status=status))
    return agent._checkpoint_save(messages, {}, status=status)


def _observe_append(store, monkeypatch):
    calls = []
    original = store.append_state

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "append_state", record)
    return calls


def test_agent_uses_delta_without_replacing_canonical_prefix(
    checkpoint_store, monkeypatch
):
    store = checkpoint_store
    calls = _observe_append(store, monkeypatch)
    with _checkpoint_context(store) as (agent, messages, run):
        _save(agent, messages)
        before = store.load_state(NS, THREAD, RUN)
        messages.add_assistant("new")
        _save(agent, messages)
        assert len(calls) == 1
        assert len(calls[0][0][3]) == 1
        assert run.revision == 2
        assert (
            store.load_state(NS, THREAD, RUN)["messages"]["items"]
            == before["messages"]["items"] + messages.to_items()[-1:]
        )
        assert "_checkpoint_write_state" not in run.durable_state()
        assert "CheckpointBase(" not in repr(run)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_append_conflict_is_not_retried_inline_and_cleanup_can_full_save(
    checkpoint_store, monkeypatch, asynchronous
):
    store = checkpoint_store
    calls = _observe_append(store, monkeypatch)
    with _checkpoint_context(store) as (agent, messages, run):
        _save(agent, messages, asynchronous=asynchronous)
        foreign = deepcopy(store.load_state(NS, THREAD, RUN))
        foreign["metadata"]["foreign"] = "same-revision overwrite"
        store.save_state(NS, THREAD, RUN, foreign)
        messages.add_assistant("local pending result")
        with pytest.raises(CheckpointConflictError):
            _save(agent, messages, asynchronous=asynchronous)
        assert len(calls) == 1
        assert run.revision == 1
        assert store.load_state(NS, THREAD, RUN) == foreign
        # The separate existing failure-settlement path may save full state;
        # it must not repeat the failed append with the stale cached base.
        _save(agent, messages, asynchronous=asynchronous, status="failed")
        assert len(calls) == 1
        assert store.load_state(NS, THREAD, RUN)["status"] == "failed"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_late_full_commit_wrapper_is_honored_after_seed(
    checkpoint_store, monkeypatch, asynchronous
):
    store = checkpoint_store
    calls = _observe_append(store, monkeypatch)
    with _checkpoint_context(store) as (agent, messages, _run):
        _save(agent, messages, asynchronous=asynchronous)
        original = store.commit_state
        failure = OSError("custom terminal commit hook")

        def custom_commit(*args, **kwargs):
            if args[3]["status"] == "completed":
                raise failure
            return original(*args, **kwargs)

        monkeypatch.setattr(store, "commit_state", custom_commit)
        messages.add_assistant("complete")
        with pytest.raises(OSError) as raised:
            _save(agent, messages, asynchronous=asynchronous, status="completed")
        assert raised.value is failure
        assert calls == []
        assert store.load_state(NS, THREAD, RUN)["status"] == "running"


def test_typed_prefix_change_forces_full_checkpoint(checkpoint_store, monkeypatch):
    store = checkpoint_store
    calls = _observe_append(store, monkeypatch)
    with _checkpoint_context(store) as (agent, messages, _run):
        messages._items[1]["metadata"] = {"flag": 1}
        _save(agent, messages)
        messages._items[1]["metadata"]["flag"] = True
        messages.add_assistant("new")
        _save(agent, messages)
        assert calls == []
        flag = store.load_state(NS, THREAD, RUN)["messages"]["items"][1]["metadata"][
            "flag"
        ]
        assert flag is True


def test_removed_control_key_does_not_survive_append_merge(
    checkpoint_store, monkeypatch
):
    store = checkpoint_store
    calls = _observe_append(store, monkeypatch)
    with _checkpoint_context(store) as (agent, messages, _run):
        original = agent._build_checkpoint_state
        include = [True]

        def state_with_optional_control(*args, **kwargs):
            state = original(*args, **kwargs)
            if include[0]:
                state["obsolete_control"] = {"value": "remove on next save"}
            return state

        monkeypatch.setattr(
            agent, "_build_checkpoint_state", state_with_optional_control
        )
        _save(agent, messages)
        include[0] = False
        messages.add_assistant("new")
        _save(agent, messages)
        assert calls == []
        assert "obsolete_control" not in store.load_state(NS, THREAD, RUN)


def test_coding_receipt_reseeds_provenance_for_next_agent_append(
    checkpoint_store, monkeypatch
):
    store = checkpoint_store
    calls = _observe_append(store, monkeypatch)
    with _checkpoint_context(store) as (agent, messages, run):
        _save(agent, messages)
        receipt = new_command_receipt(
            workspace_reference={"version": 1},
            backend="local",
            tool_call_id="call-command",
            message_offset=len(messages),
        )
        _persist_command_receipt(
            {"agent": agent, "run": run, "messages": messages}, receipt
        )
        assert run.revision == 2
        messages.add_assistant("after receipt")
        _save(agent, messages)
        assert len(calls) == 1
        saved = store.load_state(NS, THREAD, RUN)
        assert saved["_checkpoint"]["revision"] == 3
        assert (
            saved["runtime"]["extensions"]["command_receipts"][0]["execution_id"]
            == receipt.execution_id
        )


def test_new_non_json_container_metadata_keeps_full_checkpoint_behavior(
    checkpoint_store, monkeypatch
):
    store = checkpoint_store
    calls = _observe_append(store, monkeypatch)
    with _checkpoint_context(store) as (agent, messages, _run):
        _save(agent, messages)
        messages.metadata["opaque"] = b"legacy bytes metadata"
        messages.add_assistant("new")
        _save(agent, messages)
        assert calls == []
        saved = store.load_state(NS, THREAD, RUN)
        assert saved["status"] == "running"
        assert "opaque" in saved["messages"]["metadata"]


@pytest.mark.parametrize("position", ["container", "item"])
def test_optional_planning_never_iterates_opaque_legacy_mappings(
    checkpoint_store, monkeypatch, position
):
    from collections.abc import Mapping

    iterations = []

    class Opaque(Mapping):
        def __iter__(self):
            iterations.append(True)
            raise RuntimeError("opaque legacy value is not a JSON container")

        def __len__(self):
            return 0

        def __getitem__(self, key):
            raise KeyError(key)

        def __str__(self):
            return "opaque legacy value"

    store = checkpoint_store
    calls = _observe_append(store, monkeypatch)
    with _checkpoint_context(store) as (agent, messages, _run):
        _save(agent, messages)
        if position == "container":
            messages.metadata["opaque"] = Opaque()
        else:
            messages._items[1]["metadata"] = {"opaque": Opaque()}
        messages.add_assistant("new")
        from msgflux.data.stores import InMemoryCheckpointStore

        if position == "item" and isinstance(store, InMemoryCheckpointStore):
            # The legacy msgpack item codec already rejects opaque mappings;
            # planning must preserve that full-path error without invoking them.
            before = store.load_state(NS, THREAD, RUN)
            with pytest.raises(TypeError, match="unsupported"):
                _save(agent, messages)
            assert store.load_state(NS, THREAD, RUN) == before
        else:
            _save(agent, messages)
        assert calls == []
        assert iterations == []
        assert store.load_state(NS, THREAD, RUN)["status"] == "running"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_state_prepared_before_newer_commit_cannot_adopt_its_revision(
    checkpoint_store, monkeypatch, asynchronous
):
    store = checkpoint_store
    with _checkpoint_context(store) as (agent, messages, run):
        _save(agent, messages)
        original = agent._build_checkpoint_state
        newer = messages.copy()
        newer.add_assistant("newer committed item")
        injected = [False]

        def interleave_newer_write(*args, **kwargs):
            state = original(*args, **kwargs)
            if not injected[0]:
                injected[0] = True
                _save(agent, newer)
            return state

        monkeypatch.setattr(agent, "_build_checkpoint_state", interleave_newer_write)
        with pytest.raises(CheckpointConflictError):
            _save(agent, messages, asynchronous=asynchronous)
        persisted = store.load_state(NS, THREAD, RUN)
        assert persisted["_checkpoint"]["revision"] == run.revision == 2
        assert persisted["messages"]["items"][-1]["content"] == "newer committed item"
        newer.add_assistant("next item")
        _save(agent, newer, asynchronous=asynchronous)
        persisted = store.load_state(NS, THREAD, RUN)
        assert persisted["_checkpoint"]["revision"] == 3
        assert [
            item.get("content") for item in persisted["messages"]["items"][-2:]
        ] == ["newer committed item", "next item"]
