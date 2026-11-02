"""Approval recovery combines durable task results with command evidence."""

from types import SimpleNamespace

import msgspec
import pytest

from msgflux.runtime.service.records import ServiceRecoveryRequiredError
from msgflux.runtime.service.recovery import _validate_background_work
from msgflux.runtime.workspace.receipts import new_command_receipt


def _session(task_store, state, *, result_recorded):
    task = task_store.create(
        "background-tool",
        task_id="task",
        metadata={"thread_id": "thread", "root_run_id": "run"},
    )
    receipt = new_command_receipt(
        workspace_reference=None, backend="local", task_id=task.task_id
    )
    receipt = msgspec.structs.replace(
        receipt,
        state=state,
        returncode=0
        if state == "completed"
        else -15
        if state == "terminated"
        else None,
    )
    task_store.add_activity(
        task.task_id,
        kind="command_receipt",
        summary="command evidence",
        metadata={"receipt": receipt.to_dict()},
    )
    if result_recorded:
        task_store.complete(task.task_id, {"result": "durably recorded"})
    agent = SimpleNamespace(
        tool_library=None,
        _active_background_future_reason=lambda _libraries: None,
        _unfinished_task_reason=lambda _thread, _store: None,
    )
    return SimpleNamespace(agent=agent, task_store=task_store)


@pytest.mark.parametrize("state", ["completed", "failed", "terminated"])
def test_durable_background_result_resolves_known_terminal_command(task_store, state):
    session = _session(task_store, state, result_recorded=True)
    _validate_background_work(session, "thread", "run")


@pytest.mark.parametrize("state", ["intent", "launched", "unknown"])
def test_completed_task_never_resolves_uncertain_command(task_store, state):
    session = _session(task_store, state, result_recorded=True)
    with pytest.raises(ServiceRecoveryRequiredError, match="host reconciliation"):
        _validate_background_work(session, "thread", "run")


@pytest.mark.parametrize("state", ["completed", "failed", "terminated"])
def test_terminal_command_without_durable_task_result_remains_blocked(
    task_store, state
):
    session = _session(task_store, state, result_recorded=False)
    with pytest.raises(ServiceRecoveryRequiredError, match="host reconciliation"):
        _validate_background_work(session, "thread", "run")
