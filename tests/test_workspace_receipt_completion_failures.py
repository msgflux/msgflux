from __future__ import annotations

import pytest

from msgflux.exceptions import TaskPauseRequestedError
from msgflux.runtime import (
    AgentWorkspace,
    ExecutionEnvironment,
    ExecutionScope,
    LocalWorkspace,
    PermissionSet,
    ProcessRequest,
    execution_context,
)
from msgflux.runtime.agent_run import AgentRun, agent_run_context
from msgflux.runtime.isolation import SandboxCapabilities
from msgflux.runtime.workspace.environment import ProcessExecutor, ProcessResult
from msgflux.runtime.workspace.receipts import (
    bind_command_receipt_persist,
    get_command_execution,
    retain_command_receipts,
)


class _EffectExecutor(ProcessExecutor):
    capabilities = SandboxCapabilities(
        {"filesystem", "network", "process", "resource_limits"}
    )

    def __init__(self, filesystem, *, result=None):
        self.filesystem = filesystem
        self.result = ProcessResult(0) if result is None else result
        self.effects = 0

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
        self.effects += 1
        await get_command_execution().update(
            "launched", resource={"effect_id": str(self.effects)}
        )
        return self.result


def _setup(tmp_path, *, result=None):
    filesystem = LocalWorkspace("receipt-completion", tmp_path)
    executor = _EffectExecutor(filesystem, result=result)
    environment = ExecutionEnvironment(filesystem, executor)
    workspace = AgentWorkspace.from_environment(
        environment,
        permissions=PermissionSet(["process.execute"]),
    )
    scope = ExecutionScope(workspace=workspace, thread_id="thread", run_id="run")
    run = AgentRun(namespace="agent", thread_id="thread", run_id="run")
    return environment, executor, scope, run


def _persist_to_run(run, *, completed_error, unknown_error=None):
    def persist(receipt):
        if receipt.state == "completed":
            raise completed_error
        if receipt.state == "unknown" and unknown_error is not None:
            raise unknown_error
        run.set_extension(
            "command_receipts",
            retain_command_receipts(run.get_extension("command_receipts", []), receipt),
        )

    return persist


@pytest.mark.asyncio
async def test_failed_completion_settlement_blocks_next_live_run_command(tmp_path):
    environment, executor, scope, run = _setup(tmp_path)
    completion_error = OSError("checkpoint unavailable")
    persist = _persist_to_run(run, completed_error=completion_error)

    with (
        execution_context(scope=scope),
        agent_run_context(run),
        bind_command_receipt_persist(persist),
    ):
        with pytest.raises(OSError) as raised:
            await environment.arun(ProcessRequest(("fake-command", "first")))

        with pytest.raises(
            TaskPauseRequestedError,
            match="previous workspace command outcome is unknown",
        ):
            await environment.arun(ProcessRequest(("fake-command", "second")))

    assert raised.value is completion_error
    assert executor.effects == 1
    receipts = run.get_extension("command_receipts")
    assert [receipt["state"] for receipt in receipts] == ["unknown"]
    assert receipts[0]["resource"] == {"effect_id": "1"}


@pytest.mark.asyncio
async def test_completion_and_unknown_settlement_failures_preserve_primary(tmp_path):
    environment, executor, scope, run = _setup(tmp_path)
    completion_error = OSError("completion persistence failed")
    unknown_error = OSError("unknown persistence failed")
    persist = _persist_to_run(
        run,
        completed_error=completion_error,
        unknown_error=unknown_error,
    )

    with (
        execution_context(scope=scope),
        agent_run_context(run),
        bind_command_receipt_persist(persist),
        pytest.raises(OSError) as raised,
    ):
        await environment.arun(ProcessRequest(("fake-command",)))

    assert raised.value is completion_error
    assert any(
        "Command receipt update failed: unknown persistence failed" in note
        for note in getattr(raised.value, "__notes__", ())
    )
    assert executor.effects == 1
    receipts = run.get_extension("command_receipts")
    assert [receipt["state"] for receipt in receipts] == ["launched"]


@pytest.mark.asyncio
async def test_invalid_executor_result_is_settled_unknown(tmp_path):
    environment, executor, scope, run = _setup(tmp_path, result=object())
    persist = _persist_to_run(run, completed_error=OSError("unused"))

    with (
        execution_context(scope=scope),
        agent_run_context(run),
        bind_command_receipt_persist(persist),
        pytest.raises(TypeError, match="must return ProcessResult"),
    ):
        await environment.arun(ProcessRequest(("fake-command",)))

    assert executor.effects == 1
    receipts = run.get_extension("command_receipts")
    assert [receipt["state"] for receipt in receipts] == ["unknown"]
