"""Shared SQLite Agent backend layout, leases, and task-view scoping."""

from __future__ import annotations

import json
import stat
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from msgflux.exceptions import TaskIdCollisionError
from msgflux.runtime.agent_backend import SQLiteAgentBackend
from msgflux.runtime.agent_resources import AgentResources
from msgflux.tasks import TaskStoreProtocol


def _checkpoint_state() -> dict:
    return {
        "status": "completed",
        "messages": {
            "items": [
                {"type": "message", "role": "user", "content": "remember"},
                {"type": "message", "role": "assistant", "content": "ok"},
            ]
        },
    }


def test_construction_is_lazy_and_validates_cache_budget(tmp_path):
    root = tmp_path / "agent"
    backend = SQLiteAgentBackend(root)

    assert isinstance(backend, AgentResources)
    assert not root.exists()
    assert backend.agent_dir == root.absolute()
    assert backend.connection_count == 0
    assert backend.active_bindings == 0
    assert backend.has_thread("thread-a") is False
    assert not root.exists()

    for invalid in (0, -1, 4095, 1.5, True, "1048576"):
        with pytest.raises(ValueError, match="cache_bytes"):
            SQLiteAgentBackend(root, cache_bytes=invalid)
    assert not root.exists()


def test_bind_creates_one_shared_layout_and_reuses_four_connections(tmp_path):
    root = tmp_path / "agent"
    backend = SQLiteAgentBackend(root, cache_bytes=64 * 1024)
    first = backend.bind("thread-a", namespace="assistant")
    second = backend.bind("thread-b", namespace="assistant")

    assert backend.connection_count == 4
    assert backend.active_bindings == 2
    assert first.backend is backend
    assert second.backend is backend
    assert first.checkpoint_store is second.checkpoint_store
    assert first.inbox_store is second.inbox_store
    assert first.approval_store is second.approval_store
    assert first.task_store is not second.task_store
    assert first.task_store._conn is second.task_store._conn
    assert first.task_store._lock is second.task_store._lock
    assert first.thread_dir == root / "threads" / "thread-a"
    assert second.thread_dir == root / "threads" / "thread-b"
    for binding in (first, second):
        assert binding.thread_dir.is_dir()
        assert stat.S_IMODE(binding.thread_dir.stat().st_mode) == 0o700
        assert not list(binding.thread_dir.glob("*.sqlite3"))

    backend_dir = root / "runtime" / "backend"
    assert stat.S_IMODE(backend_dir.stat().st_mode) == 0o700
    marker = backend_dir / "layout.json"
    assert marker.is_file()
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert json.loads(marker.read_text()) == {
        "format": "msgflux-agent-backend",
        "layout": "shared-sqlite",
        "version": 1,
    }
    for name in (
        "checkpoints.sqlite3",
        "tasks.sqlite3",
        "inbox.sqlite3",
        "approvals.sqlite3",
    ):
        path = backend_dir / name
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    expected_cache_kib = 64 * 1024 // 4 // 1024
    for resource in backend._pool:
        assert (
            resource._conn.execute("PRAGMA cache_size").fetchone()[0]
            == -expected_cache_kib
        )

    first.close()
    first.close()
    assert backend.active_bindings == 1
    assert backend.connection_count == 4
    task = second.task_store.create("worker", task_id="still-open")
    assert task.metadata["thread_id"] == "thread-b"

    second.close()
    assert backend.active_bindings == 0
    backend.close()
    assert backend.connection_count == 0
    with pytest.raises(RuntimeError, match="closed"):
        backend.bind("thread-c", namespace="assistant")


def test_close_with_leases_refuses_without_poisoning_backend(tmp_path):
    backend = SQLiteAgentBackend(tmp_path / "agent")
    first = backend.bind("thread-a", namespace="assistant")

    with pytest.raises(RuntimeError, match="binding leases are active"):
        backend.close()
    assert backend.active_bindings == 1
    assert backend.connection_count == 4

    second = backend.bind("thread-b", namespace="assistant")
    assert backend.active_bindings == 2
    assert backend.connection_count == 4

    first.close()
    second.close()
    backend.close()
    assert backend.connection_count == 0


def test_concurrent_bindings_share_pool_and_release_independently(tmp_path):
    backend = SQLiteAgentBackend(tmp_path / "agent")

    def bind_one(index):
        binding = backend.bind(f"thread-{index}", namespace="assistant")
        binding.task_store.create("worker", task_id=f"task-{index}")
        return binding

    with ThreadPoolExecutor(max_workers=8) as executor:
        bindings = list(executor.map(bind_one, range(8)))

    assert backend.active_bindings == 8
    assert backend.connection_count == 4
    for index, binding in enumerate(bindings):
        assert [
            task.task_id for task in binding.task_store.list(include_result=False)
        ] == [f"task-{index}"]
    for binding in bindings:
        binding.close()
    assert backend.active_bindings == 0
    assert backend.connection_count == 4
    backend.close()
    assert backend.connection_count == 0


def test_task_views_filter_before_result_materialization_and_never_close_pool(
    tmp_path,
):
    backend = SQLiteAgentBackend(tmp_path / "agent")
    first = backend.bind("thread-a", namespace="assistant")
    second = backend.bind("thread-b", namespace="assistant")
    first_tasks = first.task_store
    second_tasks = second.task_store

    assert isinstance(first_tasks, TaskStoreProtocol)
    assert isinstance(second_tasks, TaskStoreProtocol)

    created = first_tasks.create(
        "worker",
        task_id="task-a",
        metadata={"root_run_id": "run-a"},
    )
    assert created.metadata["thread_id"] == "thread-a"
    second_tasks.create("worker", task_id="task-b")
    with pytest.raises(ValueError, match="thread_id"):
        first_tasks.create(
            "worker", task_id="wrong", metadata={"thread_id": "thread-b"}
        )
    with pytest.raises(ValueError, match="thread_id"):
        second_tasks.create(
            "worker", task_id="wrong-again", metadata={"thread_id": "thread-a"}
        )
    with pytest.raises(TaskIdCollisionError):
        second_tasks.create("worker", task_id="task-a")

    first_tasks._conn.execute(
        "UPDATE tasks SET result = 'not-json' WHERE task_id = 'task-a'"
    )
    first_tasks._conn.commit()
    projected = first_tasks.list(include_result=False)
    assert [task.task_id for task in projected] == ["task-a"]
    assert projected[0].result is None
    assert [task.task_id for task in second_tasks.list(include_result=False)] == [
        "task-b"
    ]
    assert first_tasks.get("task-b") is None
    assert second_tasks.get("task-a") is None
    assert [
        summary.task_id
        for summary in first_tasks.list_summaries(thread_id="thread-a", run_id="run-a")
    ] == ["task-a"]
    assert first_tasks.has_unfinished_for_thread(thread_id="thread-a") is True
    assert second_tasks.has_unfinished_for_thread(thread_id="thread-b") is True
    with pytest.raises(ValueError, match="another Agent thread"):
        first_tasks.has_unfinished_for_thread(thread_id="thread-b")
    with pytest.raises(ValueError, match="another Agent thread"):
        first_tasks.update_metadata("task-a", {"thread_id": "thread-b"})

    first_tasks.close()
    assert backend.connection_count == 4
    assert backend.active_bindings == 2
    assert second_tasks.get("task-b").status == "queued"

    first.close()
    second.close()
    backend.close()


def test_task_view_summaries_stay_scoped_to_bound_thread(tmp_path):
    backend = SQLiteAgentBackend(tmp_path / "agent")
    first = backend.bind("thread-a", namespace="assistant")
    second = backend.bind("thread-b", namespace="assistant")

    first.task_store.create(
        "child",
        task_id="root-child",
        metadata={"root_run_id": "run-a"},
    )
    first.task_store.create(
        "child",
        task_id="direct-child",
        metadata={"parent_run_id": "run-a"},
    )
    first.task_store.complete("root-child", "large result")
    second.task_store.create(
        "child",
        task_id="other-child",
        metadata={"root_run_id": "run-a"},
    )

    summaries = first.task_store.list_summaries(thread_id="thread-a", run_id="run-a")
    assert {summary.task_id for summary in summaries} == {
        "root-child",
        "direct-child",
    }
    with pytest.raises(ValueError, match="another Agent thread"):
        first.task_store.list_summaries(thread_id="thread-b", run_id="run-a")

    first.close()
    second.close()
    backend.close()


def test_has_thread_is_passive_and_reopens_shared_history(tmp_path):
    root = tmp_path / "agent"
    backend = SQLiteAgentBackend(root)
    assert backend.has_thread("thread-a") is False
    assert not root.exists()

    bound = backend.bind("thread-a", namespace="assistant")
    assert backend.has_thread("thread-a") is True
    assert bound.thread_dir.is_dir()
    bound.checkpoint_store.save_state(
        "assistant", "thread-a", "run-1", _checkpoint_state()
    )
    assert backend.has_thread("thread-a") is True
    assert backend.has_thread("thread-b") is False
    bound.close()
    backend.close()

    reopened = SQLiteAgentBackend(root)
    assert reopened.has_thread("thread-a") is True
    assert reopened.has_thread("thread-b") is False
    new_binding = reopened.bind("thread-a", namespace="assistant")
    state = new_binding.checkpoint_store.load_state("assistant", "thread-a", "run-1")
    assert state["status"] == "completed"
    new_binding.close()
    reopened.close()


def test_legacy_and_shared_layouts_refuse_mixed_opening(tmp_path):
    legacy_root = tmp_path / "legacy"
    legacy = AgentResources(legacy_root).bind("thread-a", namespace="assistant")
    legacy.close()

    backend = SQLiteAgentBackend(legacy_root)
    with pytest.raises(ValueError, match="legacy per-thread Agent stores"):
        backend.bind("thread-a", namespace="assistant")
    assert not (legacy_root / "runtime" / "backend").exists()
    backend.close()

    shared_root = tmp_path / "shared"
    shared = SQLiteAgentBackend(shared_root)
    binding = shared.bind("thread-a", namespace="assistant")
    binding.close()
    shared.close()
    with pytest.raises(ValueError, match="shared SQLite Agent backend layout"):
        AgentResources(shared_root).bind("thread-a", namespace="assistant")


def test_malformed_or_symlinked_layout_markers_fail_closed(tmp_path):
    root = tmp_path / "agent"
    backend_dir = root / "runtime" / "backend"
    backend_dir.mkdir(mode=0o700, parents=True)
    (backend_dir / "layout.json").write_text("{}")

    backend = SQLiteAgentBackend(root)
    with pytest.raises(ValueError, match="malformed"):
        backend.bind("thread-a", namespace="assistant")
    with pytest.raises(ValueError, match="malformed"):
        backend.has_thread("thread-a")
    assert not (backend_dir / "checkpoints.sqlite3").exists()

    symlink_root = tmp_path / "symlink-root"
    symlink_backend_dir = symlink_root / "runtime" / "backend"
    symlink_backend_dir.mkdir(mode=0o700, parents=True)
    outside = tmp_path / "outside-layout.json"
    outside.write_text("{}")
    (symlink_backend_dir / "layout.json").symlink_to(outside)
    symlink_backend = SQLiteAgentBackend(symlink_root)
    with pytest.raises(ValueError, match="regular file"):
        symlink_backend.has_thread("thread-a")
    assert outside.read_text() == "{}"


def test_new_and_legacy_first_initialization_are_serialized(tmp_path):
    root = tmp_path / "agent"
    barrier = threading.Barrier(2)
    outcomes = {}

    def open_backend():
        barrier.wait()
        try:
            binding = SQLiteAgentBackend(root).bind("thread-a", namespace="assistant")
        except Exception as error:
            outcomes["backend"] = error
        else:
            outcomes["backend"] = binding

    def open_legacy():
        barrier.wait()
        try:
            binding = AgentResources(root).bind("thread-a", namespace="assistant")
        except Exception as error:
            outcomes["legacy"] = error
        else:
            outcomes["legacy"] = binding

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(open_backend), executor.submit(open_legacy)]
        for future in futures:
            future.result(timeout=10)

    failures = [value for value in outcomes.values() if isinstance(value, Exception)]
    winners = [value for value in outcomes.values() if not isinstance(value, Exception)]
    assert len(failures) == 1
    assert len(winners) == 1
    assert isinstance(failures[0], ValueError)
    if "backend" in outcomes and not isinstance(outcomes["backend"], Exception):
        assert (root / "runtime" / "backend" / "layout.json").is_file()
        assert not (root / "threads" / "thread-a" / "checkpoints.sqlite3").exists()
        outcomes["backend"].close()
    else:
        assert (root / "threads" / "thread-a" / "checkpoints.sqlite3").is_file()
        assert not (root / "runtime" / "backend").exists()
        outcomes["legacy"].close()


def test_partial_pool_initialization_closes_handles_and_retries(tmp_path, monkeypatch):
    from msgflux.runtime import agent_backend

    root = tmp_path / "agent"
    backend = SQLiteAgentBackend(root)
    real_approval_store = agent_backend.SQLiteApprovalStore
    opened = []
    real_checkpoint_store = agent_backend.SQLiteCheckpointStore

    class Tracked:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            opened.append(self)

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def close(self):
            self.closed = True
            self.wrapped.close()

    def checkpoint(*, path):
        return Tracked(real_checkpoint_store(path=path))

    def fail_approval(*args, **kwargs):
        raise RuntimeError("approval adapter failed")

    monkeypatch.setattr(agent_backend, "SQLiteCheckpointStore", checkpoint)
    monkeypatch.setattr(agent_backend, "SQLiteApprovalStore", fail_approval)
    with pytest.raises(RuntimeError, match="approval adapter failed"):
        backend.bind("thread-a", namespace="assistant")
    assert backend.connection_count == 0
    assert opened and opened[0].closed
    assert (root / "runtime" / "backend" / "layout.json").is_file()

    monkeypatch.setattr(agent_backend, "SQLiteCheckpointStore", real_checkpoint_store)
    monkeypatch.setattr(agent_backend, "SQLiteApprovalStore", real_approval_store)
    binding = backend.bind("thread-a", namespace="assistant")
    assert backend.connection_count == 4
    binding.task_store.create("worker", task_id="retry-task")
    assert [task.task_id for task in binding.task_store.list()] == ["retry-task"]
    binding.close()
    backend.close()


def test_close_retry_skips_closed_handles_and_quarantines_new_binds(
    tmp_path, monkeypatch
):
    backend = SQLiteAgentBackend(tmp_path / "agent")
    binding = backend.bind("thread-a", namespace="assistant")
    checkpoints = binding.checkpoint_store
    tasks = backend._pool[1]
    binding.close()

    checkpoint_closes = 0
    task_closes = 0
    real_checkpoint_close = checkpoints.close
    real_task_close = tasks.close

    def fail_first_checkpoint_close():
        nonlocal checkpoint_closes
        checkpoint_closes += 1
        if checkpoint_closes == 1:
            raise RuntimeError("checkpoint close failed")
        return real_checkpoint_close()

    def track_task_close():
        nonlocal task_closes
        task_closes += 1
        return real_task_close()

    monkeypatch.setattr(checkpoints, "close", fail_first_checkpoint_close)
    monkeypatch.setattr(tasks, "close", track_task_close)

    with pytest.raises(ExceptionGroup, match="backend stores"):
        backend.close()
    assert checkpoint_closes == 1
    assert task_closes == 1
    assert backend.connection_count == 1
    assert backend._close_failed is True
    with pytest.raises(RuntimeError, match="close failed"):
        backend.bind("thread-b", namespace="assistant")

    backend.close()
    assert checkpoint_closes == 2
    assert task_closes == 1
    assert backend.connection_count == 0
    with pytest.raises(RuntimeError, match="closed"):
        backend.bind("thread-b", namespace="assistant")


@pytest.mark.asyncio
async def test_aclose_releases_host_owned_pool(tmp_path):
    backend = SQLiteAgentBackend(tmp_path / "agent")
    binding = backend.bind("thread-a", namespace="assistant")
    binding.close()
    await backend.aclose()
    assert backend.connection_count == 0
    assert backend.active_bindings == 0
