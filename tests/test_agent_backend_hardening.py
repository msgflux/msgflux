"""Cross-thread and cold-discovery boundaries of shared Agent persistence."""

import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.nn import Agent
from msgflux.runtime import SQLiteAgentBackend


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("lease", None),
        ("renew", False),
        ("release", False),
        ("enqueue", False),
        ("pending", []),
        ("ack", None),
    ],
)
def test_foreign_thread_cannot_read_or_change_messages_and_leases(
    tmp_path, operation, expected
):
    backend = SQLiteAgentBackend(tmp_path)
    first = backend.bind("first", namespace="agent")
    second = backend.bind("second", namespace="agent")
    try:
        owner = first.task_store
        foreign = second.task_store
        owner.create("worker", task_id="owned")
        assert owner.enqueue_message("owned", "message", "keep this")
        lease = owner.claim_worker("owned", "owner", lease_seconds=60)
        assert lease is not None
        actions = {
            "lease": lambda: foreign.get_worker_lease("owned"),
            "renew": lambda: foreign.renew_worker("owned", "owner", lease_seconds=120),
            "release": lambda: foreign.release_worker("owned", "owner"),
            "enqueue": lambda: foreign.enqueue_message("owned", "other", "wrong"),
            "pending": lambda: foreign.pending_messages("owned"),
            "ack": lambda: foreign.ack_messages("owned", ["message"]),
        }
        assert actions[operation]() == expected
        assert owner.pending_messages("owned") == [("message", "keep this")]
        assert owner.get_worker_lease("owned") == lease
        assert owner.get("owned").status == "running"
        assert foreign.get("owned") is None
    finally:
        first.close()
        second.close()
        backend.close()


def test_task_only_binding_survives_restart_without_checkpoint(tmp_path):
    root = tmp_path / "backend"
    backend = SQLiteAgentBackend(root)
    binding = backend.bind("known", namespace="worker")
    binding.task_store.create("background", task_id="queued")
    assert binding.checkpoint_store.list_runs("worker", "known") == []
    binding.close()
    backend.close()

    reopened = SQLiteAgentBackend(root)
    before = sorted(path.relative_to(root) for path in root.rglob("*"))
    assert reopened.has_thread("known")
    assert not reopened.has_thread("unknown")
    assert reopened.connection_count == 0
    assert sorted(path.relative_to(root) for path in root.rglob("*")) == before
    recovered = reopened.bind("known", namespace="worker")
    try:
        assert recovered.task_store.get("queued").status == "queued"
        assert recovered.checkpoint_store.list_runs("worker", "known") == []
    finally:
        recovered.close()
        reopened.close()


def test_offload_parents_are_private_and_observation_is_lazy(tmp_path):
    backend = SQLiteAgentBackend(tmp_path / "backend")
    binding = backend.bind("thread", namespace="worker")
    try:
        assert binding.tool_result_store(create=False) is None
        assert not (binding.thread_dir / "tool-results").exists()
        store = binding.tool_result_store()
        assert store.root == binding.thread_dir / "tool-results"
        for directory in (
            binding.thread_dir.parent,
            binding.thread_dir,
            store.root,
        ):
            assert directory.stat().st_mode & 0o777 == 0o700
    finally:
        binding.close()
        backend.close()


@pytest.mark.asyncio
async def test_public_watch_finds_backend_outside_bound_execution(tmp_path):
    backend = SQLiteAgentBackend(tmp_path / "backend")
    binding = backend.bind("known", namespace="reader")
    messages = ChatMessages(
        [{"role": "user", "content": "saved history"}],
        namespace="reader",
        thread_id="known",
    )
    binding.checkpoint_store.commit_state(
        "reader",
        "known",
        "run",
        {"status": "completed", "messages": messages._to_state()},
        expected_revision=0,
    )
    binding.close()
    agent = Agent(
        name="reader", model=Mock(model_type="chat_completion"), backend=backend
    )
    try:
        async with agent.watch("known") as observer:
            assert any(
                item.get("content") == "saved history"
                for item in observer.snapshot.messages
            )
        assert backend.active_bindings == 1
        assert agent._get_effective_checkpoint_store("unknown") is None
        assert backend.active_bindings == 1
    finally:
        await agent.aclose()
        backend.close()


@pytest.mark.parametrize("first", ["tasks", "nn", "runtime"])
def test_public_backend_import_does_not_depend_on_package_order(first):
    code = (
        f"import msgflux.{first}\n"
        "from msgflux.runtime import SQLiteAgentBackend\n"
        "from msgflux.nn import Agent\n"
        "assert SQLiteAgentBackend.__name__ == 'SQLiteAgentBackend'\n"
        "assert Agent.__name__ == 'Agent'\n"
    )
    subprocess.run(  # noqa: S603 - fixed Python executable and generated import test
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
        env=dict(os.environ),
    )


def test_explicit_resource_helper_cannot_create_hidden_legacy_checkpoint(tmp_path):
    from msgflux.runtime import RuntimeResources

    root = tmp_path / "backend"
    backend = SQLiteAgentBackend(root)
    binding = backend.bind("known", namespace="worker")
    binding.close()
    try:
        with pytest.raises(ValueError, match="shared SQLite"):
            RuntimeResources(root).checkpoint_store("hidden")
        assert not (root / "threads" / "hidden").exists()
        assert not backend.has_thread("hidden")
    finally:
        backend.close()


def test_shared_backend_refuses_existing_explicit_helper_layout(tmp_path):
    from msgflux.runtime import RuntimeResources

    root = tmp_path / "backend"
    legacy = RuntimeResources(root).checkpoint_store("known")
    legacy.save_state("worker", "known", "run", {"status": "completed"})
    legacy.close()
    backend = SQLiteAgentBackend(root)
    try:
        with pytest.raises(ValueError, match="legacy per-thread"):
            backend.bind("known", namespace="worker")
        assert not (root / "runtime" / "backend").exists()
    finally:
        backend.close()


def test_passive_discovery_waits_for_first_layout_publication(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    from msgflux.runtime import agent_resources

    root = tmp_path / "backend"
    first = SQLiteAgentBackend(root)
    second = SQLiteAgentBackend(root)
    publishing = threading.Event()
    publish = threading.Event()
    original = agent_resources._write_backend_layout_marker

    def delayed_marker(directory, marker):
        publishing.set()
        assert publish.wait(5)
        return original(directory, marker)

    monkeypatch.setattr(agent_resources, "_write_backend_layout_marker", delayed_marker)
    binding = None
    with ThreadPoolExecutor(max_workers=2) as workers:
        writer = workers.submit(first.bind, "known", namespace="worker")
        try:
            assert publishing.wait(5)
            reader = workers.submit(second.has_thread, "known")
            with pytest.raises(TimeoutError):
                reader.result(timeout=0.1)
            publish.set()
            binding = writer.result(timeout=5)
            assert reader.result(timeout=5)
            assert second.connection_count == 0
        finally:
            publish.set()
            if binding is None:
                binding = writer.result(timeout=5)
            binding.close()
            first.close()
            second.close()


def test_noncreating_layout_lock_does_not_swallow_body_errors(tmp_path):
    from msgflux.runtime.agent_resources import _agent_layout_lock

    root = tmp_path / "absent"
    with pytest.raises(FileNotFoundError, match="body failure"):
        with _agent_layout_lock(root, create=False):
            raise FileNotFoundError("body failure")
    assert not root.exists()


def test_initialization_cleanup_failure_retains_handles_for_host_retry(
    tmp_path, monkeypatch
):
    from msgflux.runtime import agent_backend

    native_checkpoint = agent_backend.SQLiteCheckpointStore
    opened = []

    class FailingClose:
        def __init__(self, **kwargs):
            self.store = native_checkpoint(**kwargs)
            self.attempts = 0
            opened.append(self)

        def close(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("checkpoint cleanup failed")
            self.store.close()

    def fail_approval(**kwargs):
        raise RuntimeError("approval initialization failed")

    monkeypatch.setattr(agent_backend, "SQLiteCheckpointStore", FailingClose)
    monkeypatch.setattr(agent_backend, "SQLiteApprovalStore", fail_approval)
    backend = SQLiteAgentBackend(tmp_path)
    with pytest.raises(RuntimeError, match="approval initialization failed") as failure:
        backend.bind("first", namespace="worker")
    assert any("checkpoint cleanup failed" in note for note in failure.value.__notes__)
    assert backend.active_bindings == 0
    assert backend.connection_count == 1
    with pytest.raises(RuntimeError, match="close failed"):
        backend.bind("second", namespace="worker")
    backend.close()
    assert opened[0].attempts == 2
    assert backend.connection_count == 0
    backend.close()
    assert opened[0].attempts == 2


def test_shared_task_history_is_selected_by_index_before_materialization(tmp_path):
    backend = SQLiteAgentBackend(tmp_path)
    first = backend.bind("first", namespace="worker")
    second = backend.bind("second", namespace="worker")
    try:
        for i in range(20):
            second.task_store.create("worker", task_id=f"foreign-{i}")
        first.task_store.create("worker", task_id="own")
        query = (
            "SELECT task_id FROM tasks "
            "WHERE json_extract(metadata, '$.thread_id')=? "
            "ORDER BY updated_at DESC"
        )
        plan = first.task_store._conn.execute(
            "EXPLAIN QUERY PLAN " + query, ("first",)
        ).fetchall()
        assert any("SEARCH tasks USING INDEX" in row[-1] for row in plan)
        assert not any("SCAN tasks" in row[-1] for row in plan)
        assert [task.task_id for task in first.task_store.list()] == ["own"]
    finally:
        first.close()
        second.close()
        backend.close()


@pytest.mark.parametrize("kind", ["legacy", "shared"])
@pytest.mark.parametrize(
    "thread_id", ["custom:thread", "../escape", "/absolute", "", None]
)
def test_passive_unrepresentable_storage_ids_are_missing_without_side_effects(
    tmp_path, kind, thread_id
):
    from msgflux.runtime.agent_resources import AgentResources

    root = tmp_path / "absent"
    resources = AgentResources(root) if kind == "legacy" else SQLiteAgentBackend(root)
    assert not resources.has_thread(thread_id)
    assert not root.exists()
    with pytest.raises(ValueError, match="safe single directory"):
        resources.bind(thread_id, namespace="worker")
    assert not root.exists()
    if kind == "shared":
        resources.close()
