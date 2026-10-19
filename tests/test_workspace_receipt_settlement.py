from __future__ import annotations

import asyncio

import pytest

from msgflux.runtime import (
    AgentWorkspace,
    ExecutionEnvironment,
    ExecutionScope,
    LocalWorkspace,
    PermissionSet,
    ProcessRequest,
    execution_context,
)
from msgflux.runtime.isolation import SandboxCapabilities
from msgflux.runtime.workspace.environment import ProcessExecutor
from msgflux.runtime.workspace.receipts import bind_command_receipt_persist


class _FailingExecutor(ProcessExecutor):
    capabilities = SandboxCapabilities(
        {"filesystem", "network", "process", "resource_limits"}
    )

    def __init__(self, filesystem, error):
        self.filesystem = filesystem
        self.error = error

    def supports_workspace(self, filesystem):
        return filesystem is self.filesystem

    async def execute_stream(
        self,
        request,
        *,
        filesystem,
        permissions,
        requirements,
        abort_signal,
        on_output,
    ):
        raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("primary_error_kind", ["executor-error", "cancellation"])
@pytest.mark.parametrize(
    ("secondary_message", "expected_note"),
    [
        ("unknown receipt persistence failure", "unknown receipt persistence failure"),
        ("   ", "OSError"),
    ],
)
async def test_receipt_settlement_failure_preserves_primary_arun_error(
    tmp_path, streaming, primary_error_kind, secondary_message, expected_note
):
    primary_error = (
        OSError("primary executor failure")
        if primary_error_kind == "executor-error"
        else asyncio.CancelledError("primary cancel")
    )
    filesystem = LocalWorkspace("receipt-settlement", tmp_path)
    environment = ExecutionEnvironment(
        filesystem, _FailingExecutor(filesystem, primary_error)
    )
    workspace = AgentWorkspace.from_environment(
        environment,
        permissions=PermissionSet(["process.execute"]),
    )
    scope = ExecutionScope(workspace=workspace, thread_id="thread", run_id="run")
    persisted_states = []
    secondary_error = OSError(secondary_message)

    def persist(receipt):
        if receipt.state == "unknown":
            raise secondary_error
        persisted_states.append(receipt.state)

    async def on_output(_channel, _data):
        return None

    with (
        execution_context(scope=scope),
        bind_command_receipt_persist(persist),
    ):
        with pytest.raises(type(primary_error)) as raised:
            await environment.arun(
                ProcessRequest(("fake-command",)),
                on_output=on_output if streaming else None,
            )

    assert raised.value is primary_error
    assert any(
        f"Command receipt update failed: {expected_note}" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert persisted_states == ["intent"]
