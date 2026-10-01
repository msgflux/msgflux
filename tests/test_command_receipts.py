import asyncio

import pytest

from msgflux.exceptions import TaskPauseRequestedError
from msgflux.runtime.agent_run import AgentRun, agent_run_context
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.workspace.environment import _reject_unknown_prior_command
from msgflux.runtime.workspace.receipts import (
    CommandExecution,
    bind_command_execution,
    decode_command_receipt,
    get_command_execution,
    mark_tool_outputs_recorded,
    new_command_receipt,
    retain_command_receipts,
    resolved_command_execution_ids,
    unresolved_command_receipts,
)


def _receipt(**kwargs):
    return new_command_receipt(
        workspace_reference={"version": 1},
        backend="local",
        tool_call_id="call-1",
        message_offset=2,
        **kwargs,
    )


def test_receipt_session_persists_monotonic_updates_and_context_binding():
    saved = []
    command = CommandExecution(_receipt(), saved.append)
    assert get_command_execution() is None
    with bind_command_execution(command):
        assert get_command_execution() is command
        asyncio.run(command.update("launched", resource={"pid": 10}))
        asyncio.run(command.update("completed", returncode=0, stdout=b"ok"))
    assert get_command_execution() is None
    assert [item.state for item in saved] == ["launched", "completed"]
    assert saved[-1].stdout == "ok"
    with pytest.raises(ValueError, match="Invalid command receipt transition"):
        asyncio.run(command.update("launched"))


def test_output_pair_must_follow_the_receipt_message_offset():
    receipt = asyncio.run(_complete(_receipt()))
    saved = retain_command_receipts([], receipt)
    older = [{"role": "tool", "tool_call_id": "call-1", "content": "old"}]
    assert unresolved_command_receipts(
        [
            decode_command_receipt(item)
            for item in mark_tool_outputs_recorded(saved, older)
        ]
    )
    later = [
        {"role": "assistant", "content": "intent"},
        {"role": "assistant", "content": "more"},
        {"role": "tool", "tool_call_id": "call-1", "content": "new"},
    ]
    resolved = mark_tool_outputs_recorded(saved, later)
    assert not unresolved_command_receipts(
        [decode_command_receipt(item) for item in resolved]
    )


async def _complete(receipt):
    command = CommandExecution(receipt, lambda _receipt: None)
    await command.update("completed", returncode=0, stdout="z" * 20_000)
    assert len(command.receipt.stdout.encode()) <= 8192
    return command.receipt


def test_retain_keeps_unresolved_and_only_bounded_terminals():
    unresolved = _receipt()
    terminal = asyncio.run(_complete(_receipt()))
    result = retain_command_receipts([], unresolved, terminal_limit=1)
    result = retain_command_receipts(result, terminal, terminal_limit=1)
    assert {item["execution_id"] for item in result} == {
        unresolved.execution_id,
        terminal.execution_id,
    }


def test_reconciliation_requires_explicit_id_and_matching_workspace():
    receipt = _receipt()
    extensions = {
        "command_reconciliations": {
            "schema_version": 1,
            "decisions": {
                "decision-1": {
                    "expected_revision": 4,
                    "decided_by": "host",
                    "reason": "Inspected process exited",
                    "workspace_reference": receipt.workspace_reference,
                    "execution_ids": [receipt.execution_id],
                    "results": {"call-1": "No process remains"},
                }
            },
        }
    }
    assert resolved_command_execution_ids(extensions, receipts=[receipt]) == {
        receipt.execution_id
    }
    extensions["command_reconciliations"]["decisions"]["decision-1"][
        "workspace_reference"
    ] = {"version": 1, "workspace_id": "other"}
    with pytest.raises(ValueError, match="workspace does not match"):
        resolved_command_execution_ids(extensions, receipts=[receipt])


def test_live_agent_cannot_start_another_command_after_unknown_result():
    receipt = asyncio.run(_unknown(_receipt()))
    run = AgentRun(namespace="agent", thread_id="thread", run_id="run")
    run.set_extension("command_receipts", [receipt.to_dict()])
    with (
        agent_run_context(run),
        pytest.raises(
            TaskPauseRequestedError,
            match="previous workspace command outcome is unknown",
        ),
    ):
        _reject_unknown_prior_command(
            {"checkpoint_store": None},
            ExecutionScope(namespace="agent", thread_id="thread", run_id="run"),
            recorder=None,
        )


async def _unknown(receipt):
    command = CommandExecution(receipt, lambda _receipt: None)
    await command.update("unknown")
    return command.receipt
