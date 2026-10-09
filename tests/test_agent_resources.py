"""Per-thread durable Agent store bindings and lifecycle ownership."""

import stat

import pytest

from msgflux.runtime.agent_resources import AgentResources


def test_binding_opens_private_per_thread_stores_and_reopens_stably(tmp_path):
    root = tmp_path / "agents" / "main"
    resources = AgentResources(root)
    assert not root.exists()

    first = resources.bind("thread_a", namespace="assistant")
    first_checkpoint_id = first.checkpoint_store.routing_id
    first_inbox_id = first.inbox_store.routing_id
    first.agent_inbox.fork(run_id="run_stable").user_message("queued")
    assert first.agent_inbox.namespace == "assistant"
    assert first.agent_inbox.thread_id == "thread_a"
    assert first.task_store is not None
    assert first.approval_store is not None
    assert {
        "checkpoints.sqlite3",
        "tasks.sqlite3",
        "inbox.sqlite3",
        "approvals.sqlite3",
    } <= {path.name for path in first.thread_dir.iterdir()}
    assert stat.S_IMODE(first.thread_dir.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE((first.thread_dir / name).stat().st_mode) == 0o600
        for name in (
            "checkpoints.sqlite3",
            "tasks.sqlite3",
            "inbox.sqlite3",
            "approvals.sqlite3",
        )
    )
    first.close()
    first.close()

    reopened = resources.bind("thread_a", namespace="assistant")
    assert reopened.checkpoint_store.routing_id == first_checkpoint_id
    assert reopened.inbox_store.routing_id == first_inbox_id
    assert reopened.agent_inbox.fork(run_id="run_stable").peek()
    other = resources.bind("thread_b", namespace="assistant")
    assert other.thread_dir != reopened.thread_dir
    assert not other.agent_inbox.peek()
    reopened.close()
    other.close()


@pytest.mark.parametrize("thread_id", ["", "../escape", "/absolute", "a/b", "."])
def test_invalid_thread_id_does_not_create_resource_directories(tmp_path, thread_id):
    root = tmp_path / "agents" / "main"
    resources = AgentResources(root)
    with pytest.raises(ValueError, match="safe single directory"):
        resources.bind(thread_id, namespace="assistant")
    assert not root.exists()


def test_binding_rejects_invalid_namespace_and_verbose_type(tmp_path):
    resources = AgentResources(tmp_path / "agent")
    with pytest.raises(ValueError, match="namespace"):
        resources.bind("thread_a", namespace=" ")
    with pytest.raises(TypeError, match="verbose"):
        resources.bind("thread_a", namespace="assistant", verbose=1)
    assert not resources.agent_dir.exists()


def test_binding_rejects_database_symlink_before_opening_it(tmp_path):
    root = tmp_path / "agent"
    thread_dir = root / "threads" / "thread_a"
    thread_dir.mkdir(mode=0o700, parents=True)
    external = tmp_path / "outside.sqlite3"
    external.write_bytes(b"keep this file unchanged")
    (thread_dir / "checkpoints.sqlite3").symlink_to(external)

    with pytest.raises(ValueError, match="regular file"):
        AgentResources(root).bind("thread_a", namespace="assistant")

    assert external.read_bytes() == b"keep this file unchanged"


def test_partial_initialization_closes_open_connections_without_deleting_files(
    tmp_path, monkeypatch
):
    from msgflux.runtime import agent_resources

    closed = []
    real_checkpoint = agent_resources.SQLiteCheckpointStore
    real_tasks = agent_resources.SQLiteTaskStore
    real_inbox = agent_resources.SQLiteAgentInboxStore

    class Tracked:
        def __init__(self, wrapped, label):
            self.wrapped = wrapped
            self.label = label

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

        def close(self):
            closed.append(self.label)
            self.wrapped.close()

    def checkpoint(*, path):
        return Tracked(real_checkpoint(path=path), "checkpoint")

    def task(*, path):
        return Tracked(real_tasks(path=path), "task")

    def inbox(*, path):
        return Tracked(real_inbox(path=path), "inbox")

    def fail_approval(*args, **kwargs):
        raise RuntimeError("approval database could not open")

    monkeypatch.setattr(agent_resources, "SQLiteCheckpointStore", checkpoint)
    monkeypatch.setattr(agent_resources, "SQLiteTaskStore", task)
    monkeypatch.setattr(agent_resources, "SQLiteAgentInboxStore", inbox)
    monkeypatch.setattr(agent_resources, "SQLiteApprovalStore", fail_approval)

    with pytest.raises(RuntimeError, match="approval database"):
        AgentResources(tmp_path / "agent").bind("thread_a", namespace="assistant")

    assert closed == ["inbox", "task", "checkpoint"]
    thread_dir = tmp_path / "agent" / "threads" / "thread_a"
    assert (thread_dir / "checkpoints.sqlite3").exists()
    assert (thread_dir / "tasks.sqlite3").exists()
    assert (thread_dir / "inbox.sqlite3").exists()


def test_bound_resource_close_retries_only_handles_that_failed(tmp_path):
    from pathlib import Path

    from msgflux.runtime.agent_resources import BoundAgentResources

    class Handle:
        def __init__(self, *, failures=0):
            self.failures = failures
            self.calls = 0

        def close(self):
            self.calls += 1
            if self.failures:
                self.failures -= 1
                raise RuntimeError("close failed")

    approval = Handle()
    inbox = Handle(failures=1)
    task = Handle()
    checkpoint = Handle()
    resources = BoundAgentResources(
        thread_id="thread_a",
        namespace="assistant",
        thread_dir=Path(tmp_path),
        checkpoint_store=checkpoint,
        task_store=task,
        inbox_store=inbox,
        agent_inbox=object(),
        approval_store=approval,
    )

    with pytest.raises(ExceptionGroup, match="resource stores"):
        resources.close()
    assert not resources._closed
    assert [approval.calls, inbox.calls, task.calls, checkpoint.calls] == [1, 1, 1, 1]

    resources.close()
    assert resources._closed
    assert [approval.calls, inbox.calls, task.calls, checkpoint.calls] == [1, 2, 1, 1]
    resources.close()
    assert [approval.calls, inbox.calls, task.calls, checkpoint.calls] == [1, 2, 1, 1]
