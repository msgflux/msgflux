"""Crash/restart gates using spawn: no inherited stores or runtime contexts."""

import multiprocessing
import os
import sqlite3
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from msgflux.data.stores import CheckpointConflictError, SQLiteCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import AgentApprovals, ExecutionScope, SQLiteApprovalStore


@pytest.fixture(autouse=True)
def offline_workers(monkeypatch):
    # spawn inherits environment changes from earlier telemetry tests. Disable
    # exporters before child imports, so shutdown never waits on external I/O.
    monkeypatch.setenv("MSGTRACE_TELEMETRY_ENABLED", "false")
    monkeypatch.setenv("MSGTRACE_EXPORTER", "console")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")


@contextmanager
def workers(*jobs):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=target, args=args) for target, args in jobs]
    try:
        for process in processes:
            process.start()
        yield processes
    finally:
        for process in processes:
            if process.pid is None:
                continue
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            assert not process.is_alive()
            process.close()


def run_worker(target, *args, expected=0):
    with workers((target, args)) as processes:
        processes[0].join(timeout=20)
        assert processes[0].exitcode == expected


def _competing_commit(path, barrier):
    store = SQLiteCheckpointStore(path)
    try:
        state = store.load_state("a", "t", "r")
        barrier.wait(timeout=10)
        try:
            store.commit_state(
                "a",
                "t",
                "r",
                {"winner": os.getpid()},
                expected_revision=state["_checkpoint"]["revision"],
            )
        except CheckpointConflictError:
            return
    finally:
        store.close()


def test_process_checkpoint_cas_has_exactly_one_winner(tmp_path):
    path = str(tmp_path / "cp.db")
    store = SQLiteCheckpointStore(path)
    store.commit_state("a", "t", "r", {})
    cursor = store.read_commits("a", "t", "r").cursor
    barrier = multiprocessing.get_context("spawn").Barrier(2)
    try:
        with workers(
            *[(_competing_commit, (path, barrier)) for _ in range(2)]
        ) as processes:
            for process in processes:
                process.join(timeout=20)
                assert process.exitcode == 0
            pids = [process.pid for process in processes]
        assert store.load_state("a", "t", "r")["winner"] in pids
        assert len(store.read_commits("a", "t", "r", after=cursor).events) == 1
    finally:
        store.close()


def _die_inside_transaction(path):
    store = SQLiteCheckpointStore(path)
    original = store._serialize

    def crash(value):
        if value.get("event_type") == "crash":
            # commit_state has already UPSERTed the snapshot, but not committed.
            assert store._conn.in_transaction
            os._exit(73)
        return original(value)

    store._serialize = crash
    store.commit_state(
        "a",
        "t",
        "r",
        {"partial": True},
        expected_revision=1,
        event={"event_type": "crash"},
    )


def test_process_death_rolls_back_snapshot_and_both_event_logs(tmp_path):
    path = str(tmp_path / "cp.db")
    store = SQLiteCheckpointStore(path)
    first = store.commit_state("a", "t", "r", {"before": True})
    cursor = store.read_commits("a", "t", "r").cursor
    store.close()
    run_worker(_die_inside_transaction, path, expected=73)
    reopened = SQLiteCheckpointStore(path)
    try:
        assert reopened.load_state("a", "t", "r") == first.state
        assert reopened.load_events("a", "t", "r") == []
        assert not reopened.read_commits("a", "t", "r", after=cursor).events
        reopened.commit_state("a", "t", "r", {"recovered": True}, expected_revision=1)
    finally:
        reopened.close()


def _response(call=False):
    response = ModelResponse()
    if call:
        calls = ToolCallAggregator()
        calls.process(0, "write:1", "publish", '{"value":"entry"}')
        response.set_response_type("tool_call")
        response.add(calls)
    else:
        response.set_response_type("text_generation")
        response.add("done")
    return response


def _agent(paths, *, crash=False, entered=None, release=None):
    checkpoint, journal = SQLiteCheckpointStore(paths[0]), SQLiteApprovalStore(paths[1])

    def publish(value: str) -> str:
        """Record an observable effect in the test's external database."""
        with sqlite3.connect(paths[2]) as effects:
            effects.execute("INSERT INTO effects(value) VALUES (?)", (value,))
        if crash:
            os._exit(74)
        if entered is not None:
            entered.set()
            assert release.wait(timeout=15)
        return "published"

    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="publisher",
        model=model,
        tools=[publish],
        checkpoint_store=checkpoint,
        approvals=AgentApprovals(journal, {"publish": "v1"}, "p1"),
    )
    return agent, checkpoint, journal


def _scope():
    return ExecutionScope(
        namespace="publisher", thread_id="t", run_id="r", principal="user"
    )


def _pause(paths):
    agent, checkpoint, journal = _agent(paths)
    agent.generator.forward = Mock(return_value=_response(call=True))
    try:
        with pytest.raises(TaskPauseRequestedError):
            agent("publish", scope=_scope())
        assert agent.generator.forward.call_count == 1
    finally:
        checkpoint.close()
        journal.close()


def _approve(paths):
    agent, checkpoint, journal = _agent(paths)
    try:
        record = journal.pending("publisher", "t", "r")[0]
        agent.decide_approval(record.request_id, approved=True, decided_by="operator")
    finally:
        checkpoint.close()
        journal.close()


def _execute_and_die(paths):
    agent, _, _ = _agent(paths, crash=True)
    agent.generator.forward = Mock(side_effect=AssertionError("unexpected model call"))
    agent("", scope=_scope())


def _execute_held(paths, entered, release):
    agent, checkpoint, journal = _agent(paths, entered=entered, release=release)
    agent.generator.forward = Mock(return_value=_response())
    try:
        assert agent("", scope=_scope()) == "done"
        assert agent.generator.forward.call_count == 1
    finally:
        checkpoint.close()
        journal.close()


def _resume_uncertain(paths):
    agent, checkpoint, journal = _agent(paths)
    agent.generator.forward = Mock(side_effect=AssertionError("unexpected model call"))
    try:
        before = checkpoint.load_state("publisher", "t", "r")
        with pytest.raises(TaskPauseRequestedError, match="reconciliation"):
            agent("", scope=_scope())
        assert checkpoint.load_state("publisher", "t", "r") == before
        agent.generator.forward.assert_not_called()
    finally:
        checkpoint.close()
        journal.close()


def test_independent_agent_resume_does_not_invalidate_live_worker(tmp_path):
    paths = tuple(str(tmp_path / name) for name in ("cp.db", "ap.db", "effects.db"))
    with sqlite3.connect(paths[2]) as effects:
        effects.execute("CREATE TABLE effects(value TEXT)")
    run_worker(_pause, paths)
    run_worker(_approve, paths)
    context = multiprocessing.get_context("spawn")
    entered, release = context.Event(), context.Event()
    with workers((_execute_held, (paths, entered, release))) as processes:
        try:
            assert entered.wait(timeout=10)
            run_worker(_resume_uncertain, paths)
        finally:
            release.set()
        processes[0].join(timeout=10)
        assert processes[0].exitcode == 0
    store = SQLiteCheckpointStore(paths[0])
    try:
        assert store.load_state("publisher", "t", "r")["status"] == "completed"
        with sqlite3.connect(paths[2]) as effects:
            assert effects.execute("SELECT value FROM effects").fetchall() == [
                ("entry",)
            ]
    finally:
        store.close()


def _reconcile_and_finish(paths):
    agent, checkpoint, journal = _agent(paths)
    agent.generator.forward = Mock(return_value=_response())
    try:
        state = agent.inspect_approval_batch("t", "r")
        agent.reconcile_approval_batch(
            "t",
            "r",
            expected_revision=state["_checkpoint"]["revision"],
            decision_id="incident",
            decided_by="operator",
            worker_stopped=True,
            reason="verified effects database",
            results={"write:1": "published entry"},
        )
        assert agent("", scope=_scope()) == "done"
        assert agent.generator.forward.call_count == 1
    finally:
        checkpoint.close()
        journal.close()


def test_agent_process_recovery_reconciliation_and_cursor_reconnect(tmp_path):
    paths = tuple(str(tmp_path / name) for name in ("cp.db", "ap.db", "effects.db"))
    with sqlite3.connect(paths[2]) as effects:
        effects.execute("CREATE TABLE effects(value TEXT)")
    run_worker(_pause, paths)
    observer = SQLiteCheckpointStore(paths[0])
    initial = observer.read_commits("publisher", "t", "r")
    assert initial.snapshot["status"] == "paused"
    request_id = initial.snapshot["runtime"]["extensions"]["pending_approvals"][
        "requests"
    ]["write:1"]
    observer.close()
    run_worker(_approve, paths)
    run_worker(_execute_and_die, paths, expected=74)
    run_worker(_resume_uncertain, paths)
    with sqlite3.connect(paths[2]) as effects:
        assert effects.execute("SELECT value FROM effects").fetchall() == [("entry",)]
    run_worker(_reconcile_and_finish, paths)
    reopened, journal = SQLiteCheckpointStore(paths[0]), SQLiteApprovalStore(paths[1])
    try:
        assert journal.get("publisher", request_id).status == "consumed"
        page = reopened.read_commits("publisher", "t", "r", after=initial.cursor)
        assert (
            sum(e.data.get("event_type") == "approval.reconciled" for e in page.events)
            == 1
        )
        final = reopened.load_state("publisher", "t", "r")
        assert final["status"] == "completed"
        assert [e.cursor.revision for e in page.events] == list(
            range(initial.cursor.revision + 1, final["_checkpoint"]["revision"] + 1)
        )
        assert "pending_approvals" not in final["runtime"]["extensions"]
        outputs = [
            item
            for item in final["messages"]["items"]
            if item.get("type") == "function_call_output"
        ]
        assert len(outputs) == 1
        assert outputs[0]["output"] == "published entry"
        with sqlite3.connect(paths[2]) as effects:
            assert effects.execute("SELECT count(*) FROM effects").fetchone()[0] == 1
    finally:
        reopened.close()
        journal.close()
