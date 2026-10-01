"""Read-only background Agent recovery inspection tests."""

import msgspec

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.nn import Agent
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.runtime import AgentTaskRecovery, TaskRecoveryReport
from msgflux.runtime.agent_inbox import AgentInbox, InMemoryAgentInboxStore
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.tasks import InMemoryTaskStore


class _UnusedModel:
    model_type = "chat_completion"

    def __init__(self):
        self.calls = 0

    def __call__(self, **_kwargs):
        self.calls += 1
        raise AssertionError("inspection must not invoke the Agent model")


def test_inspection_reports_active_owner_without_dispatching(tmp_path):
    store = InMemoryTaskStore()
    library = ToolLibrary(name="recovery", tools=[], task_store=store)
    task = store.create(
        "worker",
        task_id="agent-task",
        metadata={"task_kind": "agent"},
    )
    lease = store.claim_worker(task.task_id, "existing-owner", lease_seconds=60)
    assert lease is not None

    report = AgentTaskRecovery(library).inspect(task.task_id)

    assert isinstance(report, TaskRecoveryReport)
    assert report.version == 1
    assert report.classification == "active"
    assert report.lease_owner_id == "existing-owner"
    assert report.task_status == "running"
    assert msgspec.to_builtins(report)["task_id"] == "agent-task"
    assert library.get_background_dispatcher().get_task_future(task.task_id) is None


def test_inspection_reports_missing_task_as_blocked():
    library = ToolLibrary(name="recovery", tools=[], task_store=InMemoryTaskStore())

    report = AgentTaskRecovery(library).inspect("missing")

    assert report.classification == "blocked"
    assert report.task_status == "missing"
    assert report.reasons == ("task record was not found",)


def test_inspection_blocks_missing_recovery_dependencies():
    def worker():
        return None

    store = InMemoryTaskStore()
    store._clock = lambda: 10.0
    library = ToolLibrary(name="recovery", tools=[worker], task_store=store)
    task = store.create(
        "worker",
        task_id="agent-task",
        metadata={"task_kind": "agent", "initial_call_params": {}},
    )
    assert (
        store.claim_worker(task.task_id, "expired-owner", lease_seconds=1) is not None
    )
    store._clock = lambda: 12.0

    report = AgentTaskRecovery(library).inspect(task.task_id)

    assert report.classification == "blocked"
    assert "checkpoint store or route is unavailable" in report.reasons
    assert "Agent inbox binding is unavailable" in report.reasons


def test_queued_agent_with_durable_input_is_recoverable_before_first_claim():
    task_store = InMemoryTaskStore()
    checkpoint_store = InMemoryCheckpointStore()
    inbox_store = InMemoryAgentInboxStore()
    worker = Agent(
        name="worker",
        model=_UnusedModel(),
        checkpoint_store=checkpoint_store,
    )
    worker.tool_config = {"background": True}
    library = ToolLibrary(name="recovery", tools=[worker], task_store=task_store)
    library.set_agent_inbox(AgentInbox(owner="root", store=inbox_store))
    task = task_store.create(
        "worker",
        task_id="queued-agent",
        metadata={
            "task_kind": "agent",
            "checkpoint_namespace": "worker",
            "checkpoint_thread_id": "thread",
            "checkpoint_run_id": "queued-agent",
            "checkpoint_store_id": checkpoint_store.routing_id,
            "inbox_store_id": inbox_store.routing_id,
            "initial_call_params": {"task": "Start"},
            "task_resume_params": {},
        },
    )

    report = AgentTaskRecovery(library).inspect(task.task_id)

    assert report.task_status == "queued"
    assert report.checkpoint_status is None
    assert report.lease_owner_id is None
    assert report.classification == "recoverable"
    assert task_store.get(task.task_id).status == "queued"
    assert worker.model.calls == 0


def test_terminal_reconciliation_uses_host_selected_workspace(tmp_path, monkeypatch):
    workspace = AgentWorkspace.local(tmp_path)
    library = ToolLibrary(name="recovery", tools=[])
    recovery = AgentTaskRecovery(library, workspace=workspace)
    terminal = TaskRecoveryReport(
        version=1,
        task_id="agent-task",
        classification="terminal_result",
        task_status="running",
        task_updated_at="now",
    )
    monkeypatch.setattr(recovery, "inspect", lambda _task_id: terminal)
    observed = {}

    def reconcile(_task_id):
        observed["workspace"] = get_execution_scope().workspace
        return "reconciled"

    monkeypatch.setattr(library, "reconcile_agent_task", reconcile)

    assert recovery.recover("agent-task", "message") == "reconciled"
    assert observed["workspace"] is workspace


def test_recovery_requires_literal_true_worker_stopped(monkeypatch):
    library = ToolLibrary(name="recovery", tools=[])
    recovery = AgentTaskRecovery(library)
    report = TaskRecoveryReport(
        version=1,
        task_id="agent-task",
        classification="recoverable",
        task_status="running",
        task_updated_at="now",
    )
    monkeypatch.setattr(recovery, "inspect", lambda _task_id: report)

    try:
        recovery.recover("agent-task", "message", worker_stopped=1)
    except RuntimeError as exc:
        assert "Confirm that the previous worker has stopped" in str(exc)
    else:
        raise AssertionError("truthy non-boolean stop confirmation was accepted")
