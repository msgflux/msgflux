"""Durable workspace binding and checkpoint fencing for Agent task recovery."""

from types import SimpleNamespace

import pytest

from msgflux.runtime.background import BackgroundTaskDispatcher
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.runtime.workspace.references import encode_workspace_reference


class _Library:
    pass


def test_recovery_validates_stored_workspace_reference_against_agent_default(
    tmp_path,
):
    workspace = AgentWorkspace.local(tmp_path)
    dispatcher = BackgroundTaskDispatcher(_Library())
    task = SimpleNamespace(
        task_id="task",
        metadata={"workspace_reference": encode_workspace_reference(workspace)},
    )
    tool = SimpleNamespace(impl=SimpleNamespace(workspace=workspace))

    dispatcher.validate_task_workspace(task, tool=tool, resume_params={})

    other_root = tmp_path / "other"
    other_root.mkdir()
    other = AgentWorkspace.local(other_root)
    tool.impl.workspace = other
    with pytest.raises(ValueError, match="does not match live workspace"):
        dispatcher.validate_task_workspace(task, tool=tool, resume_params={})


def test_claimed_recovery_requires_unchanged_checkpoint_revision():
    task = SimpleNamespace(task_id="task")
    before = {"status": "running", "_checkpoint": {"revision": 4}}
    after = {"status": "running", "_checkpoint": {"revision": 5}}

    with pytest.raises(RuntimeError, match="checkpoint changed"):
        BackgroundTaskDispatcher._verify_claimed_checkpoint(task, before, after)


def test_queued_recovery_accepts_only_durable_initial_input():
    task = SimpleNamespace(
        task_id="task", status="queued", metadata={"initial_call_params": {"task": "x"}}
    )
    assert (
        BackgroundTaskDispatcher._validate_recovery_checkpoint(
            task, None, "worker", "thread", "run"
        )
        is None
    )

    task.metadata = {}
    with pytest.raises(RuntimeError, match="no checkpoint or durable initial input"):
        BackgroundTaskDispatcher._validate_recovery_checkpoint(
            task, None, "worker", "thread", "run"
        )


def test_claimed_recovery_refuses_unresolved_approval_batch():
    task = SimpleNamespace(task_id="task")
    checkpoint = {
        "status": "running",
        "_checkpoint": {"revision": 4},
        "runtime": {"extensions": {"pending_approvals": {"phase": "executing"}}},
    }

    with pytest.raises(RuntimeError, match="approval batch"):
        BackgroundTaskDispatcher._verify_claimed_checkpoint(
            task, checkpoint, checkpoint
        )


@pytest.mark.parametrize("phase", ["consumed", "unexpected"])
def test_claimed_recovery_requires_reconciliation_for_consumed_or_unknown_phase(
    phase,
):
    task = SimpleNamespace(task_id="task")
    checkpoint = {
        "status": "running",
        "_checkpoint": {"revision": 4},
        "runtime": {
            "extensions": {"pending_approvals": {"schema_version": 1, "phase": phase}}
        },
    }

    with pytest.raises(RuntimeError, match="approval batch"):
        BackgroundTaskDispatcher._verify_claimed_checkpoint(
            task, checkpoint, checkpoint
        )


def test_claimed_recovery_allows_waiting_or_approved_batch_to_resume():
    task = SimpleNamespace(task_id="task")
    for phase in (None, "awaiting_decision", "awaiting-decision", "approved"):
        pending = {"schema_version": 1}
        if phase is not None:
            pending["phase"] = phase
        checkpoint = {
            "status": "running",
            "_checkpoint": {"revision": 4},
            "runtime": {"extensions": {"pending_approvals": pending}},
        }
        BackgroundTaskDispatcher._verify_claimed_checkpoint(
            task, checkpoint, checkpoint
        )
