"""Agent automatic incremental checkpoint writes.

These tests exercise the public Agent flow and the conversation save path with
the built-in append-capable stores. They intentionally avoid instrumenting
``commit_state`` in positive cases because that correctly disables automatic
appends; ``append_state`` instrumentation does not.
"""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from itertools import pairwise
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux._private.checkpoint_write import (
    commit_tracker,
    invalidate_tracker,
    plan_append,
    seed_tracker,
)
from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import (
    CheckpointConflictError,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import (
    AgentRun,
    ExecutionScope,
    agent_run_context,
    execution_context,
)

NS, THREAD, RUN = "main", "thread", "run"


def _tool_call(call_id: str, value: int) -> ModelResponse:
    calls = ToolCallAggregator()
    calls.process(0, call_id, "echo", f'{{"value":{value}}}')
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    response.reasoning = None
    response.metadata = {}
    return response


def _text(value: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(value)
    response.reasoning = None
    response.metadata = {}
    return response


def _make_agent(store, effects):
    def echo(value: int) -> str:
        """Echo one value through the tool loop."""
        effects.append(value)
        return f"echo-{value}"

    model = Mock()
    model.model_type = "chat_completion"
    return Agent(name=NS, model=model, tools=[echo], checkpoint_store=store)


def _record_appends(store):
    calls = []
    original = store.append_state

    def record(namespace, thread_id, run_id, items, **kwargs):
        calls.append(
            {
                "run_id": run_id,
                "delta": len(items),
                "count": kwargs["base"].item_count,
                "revision": kwargs["base"].revision,
            }
        )
        return original(namespace, thread_id, run_id, items, **kwargs)

    store.append_state = record
    return calls


def _scope(run_id: str = RUN) -> ExecutionScope:
    return ExecutionScope(namespace=NS, thread_id=THREAD, run_id=run_id)


def _history() -> ChatMessages:
    messages = ChatMessages(namespace=NS, thread_id=THREAD)
    messages.begin_turn(turn_id=RUN)
    messages.append({"role": "user", "content": "begin"})
    return messages


def _task_handle(run_id: str = RUN):
    return SimpleNamespace(
        task_id=run_id,
        forward_messages=lambda *_: None,
        ack_messages=lambda *_: None,
    )


class _ManualRun:
    def __init__(self, store, *, asynchronous=False, task_handle=None):
        self.store = store
        self.asynchronous = asynchronous
        self.task_handle = task_handle
        self.agent = _make_agent(store, [])
        self.messages = _history()
        self.run = AgentRun(namespace=NS, thread_id=THREAD, run_id=RUN)
        self.scope = _scope()
        self.original_append = store.append_state
        self.appends = _record_appends(store)

    def _save(self, status="running", **kwargs):
        if self.asynchronous:
            asyncio.run(
                self.agent._acheckpoint_save(self.messages, {}, status=status, **kwargs)
            )
        else:
            self.agent._checkpoint_save(self.messages, {}, status=status, **kwargs)

    def save(self, status="running", **kwargs):
        with (
            execution_context(scope=self.scope, task_handle=self.task_handle),
            agent_run_context(self.run),
        ):
            self._save(status=status, **kwargs)

    def state(self):
        return self.store.load_state(NS, THREAD, RUN)

    def revision(self):
        return self.state()["_checkpoint"]["revision"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_agent_appends_after_a_single_full_seed(checkpoint_store, asynchronous):
    rounds = 12
    effects = []
    agent = _make_agent(checkpoint_store, effects)
    appends = _record_appends(checkpoint_store)
    history = ChatMessages()
    responses = [_tool_call(f"call_{i}", i) for i in range(rounds)] + [_text("done")]
    if asynchronous:
        agent.generator.aforward = AsyncMock(side_effect=responses)
        result = asyncio.run(agent.acall("go", messages=history, scope=_scope()))
    else:
        agent.generator.forward = Mock(side_effect=responses)
        result = agent("go", messages=history, scope=_scope())

    assert result == "done"
    assert effects == list(range(rounds))
    state = checkpoint_store.load_state(NS, THREAD, RUN)
    revision = state["_checkpoint"]["revision"]
    item_count = len(state["messages"]["items"])
    assert state["messages"]["items"] == history._to_state()["items"]
    assert (
        sum(
            item.get("type") == "function_call_output"
            for item in state["messages"]["items"]
        )
        == rounds
    )

    # One full seed, then only deltas. The completed checkpoint may fall back
    # once when a terminal control key such as task_result first appears.
    assert len(appends) >= rounds
    full_commits = revision - len(appends)
    assert 1 <= full_commits <= 2
    assert appends[0]["count"] < item_count
    assert all(0 < call["delta"] < item_count for call in appends)
    assert all(call["run_id"] == RUN for call in appends)
    # Each append owns exactly the suffix after the previous committed count.
    assert all(
        previous["count"] + previous["delta"] == following["count"]
        for previous, following in pairwise(appends)
    )
    assert appends[-1]["count"] + appends[-1]["delta"] == item_count

    if isinstance(checkpoint_store, SQLiteCheckpointStore):
        with sqlite3.connect(checkpoint_store.path) as conn:
            rows = conn.execute(
                "SELECT COUNT(*) FROM checkpoint_commits "
                "WHERE namespace=? AND thread_id=? AND run_id=?",
                (NS, THREAD, RUN),
            ).fetchone()[0]
        assert rows == revision


@pytest.mark.parametrize("asynchronous", [False, True])
def test_prefix_mutation_falls_back_and_reseeds(checkpoint_store, asynchronous):
    manual = _ManualRun(checkpoint_store, asynchronous=asynchronous)
    manual.save()
    assert manual.run._checkpoint_tracker is not None
    seed_revision = manual.revision()

    manual.messages.append({"role": "assistant", "content": "first"})
    manual.save()
    assert len(manual.appends) == 1
    assert manual.revision() == seed_revision + 1

    manual.messages[1]["content"] = "mutated prefix"
    manual.messages.append({"role": "user", "content": "next"})
    mutation_revision = manual.revision()
    manual.save()
    assert len(manual.appends) == 1  # full fallback, no append
    assert manual.revision() == mutation_revision + 1
    assert manual.state()["messages"]["items"][1]["content"] == "mutated prefix"
    assert manual.run._checkpoint_tracker is not None

    manual.messages.append({"role": "assistant", "content": "after reseed"})
    manual.save()
    assert len(manual.appends) == 2
    assert manual.appends[-1]["count"] >= 3


def test_plan_rejects_scope_routing_and_revision_mismatch():
    store = InMemoryCheckpointStore()
    state = {
        "status": "running",
        "messages": {
            "items": [{"item_id": "itm_1", "role": "user", "content": "one"}],
            "metadata": {},
            "thread_id": THREAD,
            "namespace": NS,
        },
    }
    committed = store.commit_state(NS, THREAD, RUN, state, expected_revision=0)
    tracker = seed_tracker(committed.base, committed.state)
    run = AgentRun(namespace=NS, thread_id=THREAD, run_id=RUN, revision=1)

    def plan(**overrides):
        values = {"namespace": NS, "thread_id": THREAD, "run_id": RUN}
        values.update(overrides)
        return plan_append(store, run, state, tracker, **values)

    assert plan() is not None
    assert plan(namespace="other") is None
    assert plan(thread_id="other") is None
    assert plan(run_id="other") is None
    assert (
        plan_append(
            InMemoryCheckpointStore(),
            run,
            state,
            tracker,
            namespace=NS,
            thread_id=THREAD,
            run_id=RUN,
        )
        is None
    )
    run.revision = 2
    assert plan() is None


def test_removed_control_key_falls_back_to_full_commit(checkpoint_store):
    manual = _ManualRun(checkpoint_store, task_handle=_task_handle())
    manual.messages.append({"role": "assistant", "content": "work"})
    manual.save(status="completed", task_result={"ok": True})
    assert "task_result" in manual.state()
    assert manual.run._checkpoint_tracker is not None

    manual.messages.append({"role": "user", "content": "follow up"})
    before_revision = manual.revision()
    before_appends = len(manual.appends)
    manual.save(status="running")
    assert len(manual.appends) == before_appends
    assert manual.revision() == before_revision + 1
    assert "task_result" not in manual.state()
    assert manual.run._checkpoint_tracker is not None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_non_json_item_uses_full_commit_and_disables_tracking(
    checkpoint_store, asynchronous
):
    manual = _ManualRun(checkpoint_store, asynchronous=asynchronous)
    manual.save()
    manual.messages.append({"role": "assistant", "content": "plain"})
    manual.save()
    assert len(manual.appends) == 1

    manual.messages.append({"role": "user", "content": b"opaque"})
    before_revision = manual.revision()
    before_appends = len(manual.appends)
    manual.save()
    assert len(manual.appends) == before_appends
    assert manual.revision() == before_revision + 1
    assert manual.run._checkpoint_tracker is None

    # Tracking stays disabled, so the next save also keeps the full contract.
    manual.messages.append({"role": "assistant", "content": "after"})
    revision = manual.revision()
    manual.save()
    assert len(manual.appends) == before_appends
    assert manual.revision() == revision + 1


@pytest.mark.parametrize("case", ["metadata", "control", "scope", "runtime_extension"])
def test_non_json_candidate_control_or_container_uses_full_commit(
    checkpoint_store, case
):
    manual = _ManualRun(checkpoint_store)
    manual.save()
    manual.messages.append({"role": "assistant", "content": "delta"})
    if case == "metadata":
        manual.messages.metadata["opaque"] = b"legacy bytes metadata"
    elif case == "runtime_extension":
        manual.run.set_extension("opaque", b"legacy bytes extension")
    else:
        original = manual.agent._build_checkpoint_state

        def patched(*args, **kwargs):
            state = original(*args, **kwargs)
            if case == "scope":
                state["scope"]["opaque"] = b"legacy bytes scope"
            else:
                state["opaque"] = b"legacy bytes control"
            return state

        manual.agent._build_checkpoint_state = patched

    before_revision = manual.revision()
    before_appends = len(manual.appends)
    manual.save()
    assert len(manual.appends) == before_appends
    assert manual.revision() == before_revision + 1
    assert manual.run._checkpoint_tracker is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_late_full_commit_seed_does_not_overwrite_newer_tracker(
    checkpoint_store, monkeypatch, asynchronous
):
    import msgflux.nn.modules.agent.conversation as conversation_module

    manual = _ManualRun(checkpoint_store, asynchronous=asynchronous)
    original = conversation_module.tracker_from_commit
    installed = []

    def delayed(committed, store):
        replacement = original(committed, store)
        assert replacement is not None
        newer = replace(
            replacement,
            base=replace(replacement.base, revision=replacement.base.revision + 1),
        )
        # Simulate a receipt publishing a newer tracker while the seed hashes.
        commit_tracker(manual.run, None, newer, newer.base.revision)
        installed.append(newer)
        return replacement

    monkeypatch.setattr(conversation_module, "tracker_from_commit", delayed)
    manual.save()
    assert installed
    assert manual.run._checkpoint_tracker is installed[0]
    assert manual.run.revision == installed[0].base.revision


@pytest.mark.parametrize("asynchronous", [False, True])
def test_late_append_ack_does_not_overwrite_newer_tracker(
    checkpoint_store, monkeypatch, asynchronous
):
    import msgflux.nn.modules.agent.conversation as conversation_module

    manual = _ManualRun(checkpoint_store, asynchronous=asynchronous)
    manual.save()
    seed_tracker_value = manual.run._checkpoint_tracker
    assert seed_tracker_value is not None
    manual.messages.append({"role": "assistant", "content": "delta"})
    original = conversation_module.accept_append
    installed = []

    def delayed(plan, tail):
        replacement = original(plan, tail)
        assert replacement is not None
        newer = replace(
            replacement,
            base=replace(replacement.base, revision=replacement.base.revision + 1),
        )
        # Simulate a receipt publishing a newer tracker before the ack is stored.
        commit_tracker(manual.run, seed_tracker_value, newer, newer.base.revision)
        installed.append(newer)
        return replacement

    monkeypatch.setattr(conversation_module, "accept_append", delayed)
    manual.save()
    assert installed
    assert manual.run._checkpoint_tracker is installed[0]
    assert manual.run.revision == installed[0].base.revision


@pytest.mark.parametrize("asynchronous", [False, True])
def test_append_failure_propagates_without_full_retry(checkpoint_store, asynchronous):
    manual = _ManualRun(checkpoint_store, asynchronous=asynchronous)
    manual.save()
    manual.messages.append({"role": "assistant", "content": "delta"})
    revision = manual.revision()
    failure = OSError("append rejected")
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(args)
        raise failure

    checkpoint_store.append_state = fail
    with pytest.raises(OSError) as raised:
        manual.save()
    assert raised.value is failure
    assert len(attempts) == 1
    assert manual.revision() == revision
    assert manual.run._checkpoint_tracker is None

    # The existing failure path can still full-save a repair without retrying
    # the failed append in the same call.
    manual.messages.append({"role": "user", "content": "repair"})
    checkpoint_store.append_state = manual.original_append
    restored_appends = _record_appends(checkpoint_store)
    manual.save(status="failed")
    assert restored_appends == []
    assert manual.revision() == revision + 1
    assert manual.run._checkpoint_tracker is not None


def test_commit_tracker_cas_and_revision_monotonic():
    run = AgentRun(namespace=NS, thread_id=THREAD, run_id=RUN, revision=1)
    old = SimpleNamespace(base=SimpleNamespace(revision=1))
    newer = SimpleNamespace(base=SimpleNamespace(revision=2))
    late = SimpleNamespace(base=SimpleNamespace(revision=1))
    run._checkpoint_tracker = old

    assert commit_tracker(run, old, newer, 2) is True
    assert run._checkpoint_tracker is newer
    assert run.revision == 2

    # A late ack from the old tracker cannot overwrite or regress the newer one.
    assert commit_tracker(run, old, late, 1) is False
    assert run._checkpoint_tracker is newer
    assert run.revision == 2
    assert invalidate_tracker(run, old) is False
    assert run._checkpoint_tracker is newer
    assert invalidate_tracker(run, newer) is True
    assert run._checkpoint_tracker is None
    assert run.revision == 2

    # A tracker whose base disagrees with the acknowledged revision is not
    # installed, even when the identity still matches.
    fresh = AgentRun(namespace=NS, thread_id=THREAD, run_id=RUN)
    assert commit_tracker(fresh, None, old, 2) is True
    assert fresh._checkpoint_tracker is None
    assert fresh.revision == 2


def test_stale_append_base_propagates_conflict_without_full_retry(tmp_path):
    path = str(tmp_path / "append-cas.db")
    writer = SQLiteCheckpointStore(path)
    other = SQLiteCheckpointStore(path)
    try:
        manual = _ManualRun(writer)
        manual.save()
        manual.messages.append({"role": "assistant", "content": "delta"})
        before_revision = manual.revision()

        replacement = other.load_state(NS, THREAD, RUN)
        replacement["_checkpoint"]["stream_id"] = "other-stream"
        other.save_state(NS, THREAD, RUN, replacement)

        with pytest.raises(CheckpointConflictError):
            manual.save()
        state = writer.load_state(NS, THREAD, RUN)
        assert state["_checkpoint"]["revision"] == before_revision
        assert state["_checkpoint"]["stream_id"] == "other-stream"
    finally:
        writer.close()
        other.close()


def test_late_commit_wrapper_is_honored_and_can_be_removed(
    checkpoint_store, monkeypatch
):
    manual = _ManualRun(checkpoint_store)
    manual.save()
    manual.messages.append({"role": "assistant", "content": "one"})
    manual.save()
    assert len(manual.appends) == 1

    original = checkpoint_store.commit_state
    wrapped = []

    def wrapper(*args, **kwargs):
        wrapped.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(checkpoint_store, "commit_state", wrapper)
    assert checkpoint_store.supports_agent_incremental is False
    manual.messages.append({"role": "user", "content": "wrapped"})
    manual.save()
    assert len(wrapped) == 1
    assert len(manual.appends) == 1
    assert manual.run._checkpoint_tracker is None

    monkeypatch.setattr(checkpoint_store, "commit_state", original)
    assert checkpoint_store.supports_agent_incremental is True
    manual.messages.append({"role": "assistant", "content": "reseeding"})
    manual.save()  # Full commit because the custom commit cleared the tracker.
    assert len(manual.appends) == 1
    manual.messages.append({"role": "user", "content": "incremental again"})
    manual.save()
    assert len(manual.appends) == 2


def test_subclass_commit_override_never_bypasses(checkpoint_store, tmp_path):
    class CustomStore(type(checkpoint_store)):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.commits = 0

        def commit_state(self, *args, **kwargs):
            self.commits += 1
            return super().commit_state(*args, **kwargs)

    store = (
        CustomStore(str(tmp_path / "custom.db"))
        if isinstance(checkpoint_store, SQLiteCheckpointStore)
        else CustomStore()
    )
    try:
        assert store.supports_agent_incremental is False
        manual = _ManualRun(store)
        manual.save()
        manual.messages.append({"role": "assistant", "content": "one"})
        manual.save()
        manual.messages.append({"role": "user", "content": "two"})
        manual.save()
        assert manual.appends == []
        assert store.commits == 3
        assert (
            manual.state()["messages"]["items"] == manual.messages._to_state()["items"]
        )
    finally:
        if hasattr(store, "close"):
            store.close()


def test_tracker_is_ephemeral_and_run_local():
    run = AgentRun(namespace=NS, thread_id=THREAD, run_id=RUN)
    run._checkpoint_tracker = object()
    durable = run.durable_state()
    assert "_checkpoint_tracker" not in durable
    assert "_checkpoint_tracker" not in repr(run)
    assert run == AgentRun(namespace=NS, thread_id=THREAD, run_id=RUN)
    assert AgentRun.from_durable_state(durable)._checkpoint_tracker is None


def test_interleaved_runs_keep_independent_trackers(checkpoint_store):
    first = _ManualRun(checkpoint_store)
    second = AgentRun(namespace=NS, thread_id=THREAD, run_id="second")
    second_history = ChatMessages(namespace=NS, thread_id=THREAD)
    second_history.begin_turn(turn_id="second")
    second_history.append({"role": "user", "content": "second seed"})

    first.save()
    with execution_context(scope=_scope("second")), agent_run_context(second):
        first.agent._checkpoint_save(second_history, {}, status="running")
    assert first.run._checkpoint_tracker is not None
    assert second._checkpoint_tracker is not None
    assert first.run._checkpoint_tracker.base.run_id == RUN
    assert second._checkpoint_tracker.base.run_id == "second"

    first.messages.append({"role": "assistant", "content": "first delta"})
    first.save()
    with execution_context(scope=_scope("second")), agent_run_context(second):
        second_history.append({"role": "assistant", "content": "second delta"})
        first.agent._checkpoint_save(second_history, {}, status="running")

    assert [call["run_id"] for call in first.appends] == [RUN, "second"]
    assert first.state()["_checkpoint"]["revision"] == 2
    second_state = checkpoint_store.load_state(NS, THREAD, "second")
    assert second_state["_checkpoint"]["revision"] == 2
    assert second_state["messages"]["items"][-1]["content"] == "second delta"
