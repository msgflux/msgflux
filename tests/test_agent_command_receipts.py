import asyncio

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.context import ExecutionScope
from msgflux.runtime.workspace.receipts import (
    CommandExecution,
    decode_command_receipt,
    new_command_receipt,
)
from msgflux.runtime.approvals.reconciliation import (
    _resolve_command_receipts_for_batch,
)


class _Model:
    model_type = "chat_completion"


def test_agent_resume_blocks_unresolved_command_receipt_before_provider_call():
    store = InMemoryCheckpointStore()
    agent = Agent(name="agent", model=_Model(), checkpoint_store=store)
    receipt = new_command_receipt(
        workspace_reference=None,
        backend="local",
        run_id="run-1",
        tool_call_id="call-1",
        message_offset=1,
    )
    state = {
        "schema_version": 1,
        "status": "running",
        "messages": {"items": [], "metadata": {}, "thread_id": "thread-1"},
        "runtime": {
            "schema_version": 1,
            "extensions": {"command_receipts": [receipt.to_dict()]},
        },
    }
    store.save_state("agent", "thread-1", "run-1", state)

    with pytest.raises(TaskPauseRequestedError, match="command outcome"):
        agent._try_resume_from_checkpoint(
            None,
            scope=ExecutionScope(
                namespace="agent", thread_id="thread-1", run_id="run-1"
            ),
        )


def test_approval_reconciliation_resolves_only_a_paired_current_command_receipt():
    receipt = new_command_receipt(
        workspace_reference={"version": 1, "workspace_id": "workspace"},
        backend="local",
        run_id="run-1",
        tool_call_id="call-1",
        message_offset=1,
    )
    command = CommandExecution(receipt, lambda _receipt: None)
    asyncio.run(command.update("unknown"))
    extensions = {"command_receipts": [command.receipt.to_dict()]}

    class _Messages:
        def _to_state(self):
            return {
                "items": [
                    {"role": "assistant", "content": "tool call"},
                    {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
                ]
            }

    _resolve_command_receipts_for_batch(extensions, _Messages(), {"call-1"}, "run-1")
    resolved = decode_command_receipt(extensions["command_receipts"][0])
    assert resolved.state == "reconciled"
    assert resolved.tool_output_recorded is True
