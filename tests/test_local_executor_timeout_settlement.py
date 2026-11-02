"""Deterministic timeout settlement for the unsandboxed local executor.

A local execution timeout is a *confirmed* termination: the drain helper
signals the owned process group, reaps the leader and the executor still knows
the leader's return code.  The receipt must record that evidence, otherwise
every timed-out command leaves an ``unknown`` receipt that pauses the Agent
before the next command even though nothing is genuinely uncertain.

The guards stay: an adapter that cannot prove termination (custom executor, a
failed completion persistence, an unconfirmed descendant) must still settle
``unknown`` and block the next live command.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from contextlib import contextmanager

import pytest

from msgflux.exceptions import AbortRequestedError, TaskPauseRequestedError
from msgflux.runtime import (
    AbortSignal,
    AgentWorkspace,
    ExecutionEnvironment,
    ExecutionScope,
    LocalProcessExecutor,
    LocalWorkspace,
    PermissionSet,
    ProcessRequest,
    SandboxRequirements,
    execution_context,
)
from msgflux.runtime.agent_run import AgentRun, agent_run_context
from msgflux.runtime.isolation import SandboxCapabilities
from msgflux.runtime.workspace.environment import ProcessExecutor
from msgflux.runtime.workspace.process_capture import ProcessOutputLimitError
from msgflux.runtime.workspace.receipts import (
    bind_command_receipt_persist,
    mark_tool_outputs_recorded,
    retain_command_receipts,
    unresolved_command_receipts,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="LocalWorkspace requires POSIX"
)

TOOL_CALL_ID = "call-timeout"
OFFSET = 0


class _UnprovableExecutor(ProcessExecutor):
    """Adapter without local process evidence: it must stay unknown."""

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


def _local_setup(tmp_path, *, abort_signal=None):
    filesystem = LocalWorkspace("timeout-settlement", tmp_path)
    executor = LocalProcessExecutor(filesystem)
    environment = ExecutionEnvironment(
        filesystem=filesystem,
        process_executor=executor,
        requirements=SandboxRequirements(),
        write_guarantee="cooperative_compare",
    )
    workspace = AgentWorkspace.from_environment(
        environment, permissions=PermissionSet(["process.execute"])
    )
    scope = ExecutionScope(
        namespace="agent",
        thread_id="thread",
        run_id="run",
        workspace=workspace,
        abort_signal=abort_signal,
    )
    run = AgentRun(namespace="agent", thread_id="thread", run_id="run")
    return environment, executor, scope, run


def _persist(run, *, terminated_error=None, completed_error=None):
    def persist(receipt):
        if receipt.state == "terminated" and terminated_error is not None:
            raise terminated_error
        if receipt.state == "completed" and completed_error is not None:
            raise completed_error
        run.set_extension(
            "command_receipts",
            retain_command_receipts(run.get_extension("command_receipts", []), receipt),
        )

    return persist


@contextmanager
def _scope_context(scope, run, persist):
    with (
        execution_context(
            scope=scope,
            run_id="run",
            namespace="agent",
            thread_id="thread",
            tool_call_id=TOOL_CALL_ID,
            tool_call_message_offset=OFFSET,
        ),
        agent_run_context(run),
        bind_command_receipt_persist(persist),
    ):
        yield


def _child_program(marker_path):
    """Emit one deterministic line, then outlive any short deadline."""
    return (
        "import pathlib,sys,time;"
        f"pathlib.Path({str(marker_path)!r}).write_text(str(__import__('os').getpid()));"
        "print('started', flush=True);"
        "time.sleep(30)"
    )


GROUP_DESCENDANT_PROGRAM = """
import os
import signal
import sys
import time

marker = sys.argv[1]
child = os.fork()
if child == 0:
    # A same-group descendant that ignores SIGTERM and holds no captured pipe.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    time.sleep(60)
with open(marker, "w", encoding="ascii") as stream:
    stream.write(str(child))
print("started", flush=True)
time.sleep(60)
"""


def _assert_process_gone(pid, *, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    pytest.fail(f"descendant {pid} survived the process group cleanup")


async def _wait_for_marker(marker, *, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.exists() and marker.read_text():
            return int(marker.read_text())
        await asyncio.sleep(0.01)
    pytest.fail(f"marker {marker} was not written")


@pytest.mark.asyncio
async def test_local_timeout_settles_confirmed_termination_and_allows_next_call(
    tmp_path,
):
    environment, executor, scope, run = _local_setup(tmp_path)
    marker = tmp_path / "child.pid"
    request = ProcessRequest(
        (sys.executable, "-c", _child_program(marker)),
        timeout_seconds=0.4,
    )

    with _scope_context(scope, run, _persist(run)):
        with pytest.raises(asyncio.TimeoutError):
            await environment.arun(request)
        # The next command must not be refused by the unknown guard.
        follow_up = await environment.arun(
            ProcessRequest((sys.executable, "-c", "print('next')"))
        )

    assert follow_up.stdout == b"next\n"
    receipts = run.get_extension("command_receipts")
    assert len(receipts) == 2, receipts
    timed_out = receipts[0]
    assert timed_out["state"] == "terminated", timed_out
    assert timed_out["returncode"] is not None, timed_out
    assert timed_out["resource"]["pid"] > 0, timed_out
    # Partial streamed output is retained as bounded evidence.
    assert "started" in (timed_out["stdout"] or ""), timed_out
    inspected = await executor.inspect_command(timed_out)
    assert inspected.classification == "completed"
    assert inspected.returncode == timed_out["returncode"]

    # Recorded tool output resolves the terminal receipt for checkpoint resume.
    recorded = mark_tool_outputs_recorded(
        receipts,
        [
            {
                "type": "function_call_output",
                "call_id": TOOL_CALL_ID,
                "output": "Command timed out.",
            }
        ],
    )
    settled = [
        item for item in recorded if item["execution_id"] == timed_out["execution_id"]
    ]
    assert [item["state"] for item in settled] == ["terminated"], settled
    assert settled[0]["tool_output_recorded"] is True, settled
    assert unresolved_command_receipts(recorded) == ()

    _assert_process_gone(int(marker.read_text()))


@pytest.mark.asyncio
async def test_local_abort_settles_confirmed_termination(tmp_path):
    signal = AbortSignal()
    environment, _, scope, run = _local_setup(tmp_path, abort_signal=signal)
    marker = tmp_path / "abort.pid"
    request = ProcessRequest((sys.executable, "-c", _child_program(marker)))

    with _scope_context(scope, run, _persist(run)):
        task = asyncio.ensure_future(environment.arun(request))
        pid = await _wait_for_marker(marker)
        signal.abort("operator stop")
        with pytest.raises(AbortRequestedError):
            await task

    receipts = run.get_extension("command_receipts")
    assert [receipt["state"] for receipt in receipts] == ["terminated"], receipts
    assert receipts[0]["returncode"] is not None, receipts
    _assert_process_gone(pid)


@pytest.mark.asyncio
async def test_output_limit_failure_settles_confirmed_termination(tmp_path):
    environment, _, scope, run = _local_setup(tmp_path)
    marker = tmp_path / "limit.pid"
    request = ProcessRequest(
        (sys.executable, "-c", _child_program(marker)),
        max_output_bytes=2,
    )

    with _scope_context(scope, run, _persist(run)):
        with pytest.raises(ProcessOutputLimitError):
            await environment.arun(request)

    receipts = run.get_extension("command_receipts")
    assert [receipt["state"] for receipt in receipts] == ["terminated"], receipts
    _assert_process_gone(int(marker.read_text()))


@pytest.mark.asyncio
async def test_timeout_cleanup_reaps_same_group_descendant_ignoring_sigterm(
    tmp_path,
):
    """A descendant that closes stdio and ignores SIGTERM must not survive."""
    environment, _, scope, run = _local_setup(tmp_path)
    marker = tmp_path / "grandchild.pid"
    request = ProcessRequest(
        (sys.executable, "-c", GROUP_DESCENDANT_PROGRAM, str(marker)),
        timeout_seconds=0.5,
    )

    with _scope_context(scope, run, _persist(run)):
        with pytest.raises(asyncio.TimeoutError):
            await environment.arun(request)

    descendant = int(marker.read_text())
    _assert_process_gone(descendant)
    receipts = run.get_extension("command_receipts")
    assert [receipt["state"] for receipt in receipts] == ["terminated"], receipts


@pytest.mark.asyncio
async def test_unprovable_executor_timeout_remains_unknown_and_blocks(tmp_path):
    filesystem = LocalWorkspace("timeout-settlement", tmp_path)
    executor = _UnprovableExecutor(filesystem, asyncio.TimeoutError("adapter timeout"))
    environment = ExecutionEnvironment(
        filesystem=filesystem,
        process_executor=executor,
        requirements=SandboxRequirements(),
        write_guarantee="cooperative_compare",
    )
    workspace = AgentWorkspace.from_environment(
        environment, permissions=PermissionSet(["process.execute"])
    )
    scope = ExecutionScope(
        namespace="agent", thread_id="thread", run_id="run", workspace=workspace
    )
    run = AgentRun(namespace="agent", thread_id="thread", run_id="run")

    with (
        execution_context(
            scope=scope, run_id="run", namespace="agent", thread_id="thread"
        ),
        agent_run_context(run),
        bind_command_receipt_persist(_persist(run)),
    ):
        with pytest.raises(asyncio.TimeoutError):
            await environment.arun(ProcessRequest(("fake-command", "first")))
        with pytest.raises(
            TaskPauseRequestedError,
            match="previous workspace command outcome is unknown",
        ):
            await environment.arun(ProcessRequest(("fake-command", "second")))

    assert [receipt["state"] for receipt in run.get_extension("command_receipts")] == [
        "unknown"
    ]


@pytest.mark.asyncio
async def test_local_completion_persistence_failure_is_not_terminated(tmp_path):
    """A settled command whose completion cannot persist is not a termination."""
    environment, _, scope, run = _local_setup(tmp_path)
    completion_error = OSError("checkpoint unavailable")
    persist = _persist(run, completed_error=completion_error)

    with _scope_context(scope, run, persist):
        with pytest.raises(OSError) as raised:
            await environment.arun(
                ProcessRequest((sys.executable, "-c", "print('done')"))
            )
        assert raised.value is completion_error
        receipts = run.get_extension("command_receipts")
        assert [receipt["state"] for receipt in receipts] == ["unknown"], receipts
        with pytest.raises(
            TaskPauseRequestedError,
            match="previous workspace command outcome is unknown",
        ):
            await environment.arun(ProcessRequest(("fake-command", "next")))


@pytest.mark.asyncio
async def test_local_timeout_with_failed_settlement_keeps_unknown(tmp_path):
    environment, _, scope, run = _local_setup(tmp_path)
    marker = tmp_path / "persist.pid"
    request = ProcessRequest(
        (sys.executable, "-c", _child_program(marker)),
        timeout_seconds=0.4,
    )

    with _scope_context(scope, run, _persist(run, terminated_error=OSError("persist"))):
        with pytest.raises(asyncio.TimeoutError):
            await environment.arun(request)

    # A confirmed termination whose settlement cannot persist must not be
    # silently dropped: the durable record stays unknown and blocks the run.
    with _scope_context(scope, run, _persist(run)):
        with pytest.raises(TaskPauseRequestedError):
            await environment.arun(ProcessRequest(("fake-command", "next")))

    assert [receipt["state"] for receipt in run.get_extension("command_receipts")] == [
        "unknown"
    ]
    _assert_process_gone(int(marker.read_text()))


@pytest.mark.asyncio
async def test_descendant_outside_owned_group_with_open_pipes_remains_unknown(tmp_path):
    environment, _, scope, run = _local_setup(tmp_path)
    marker, stop = tmp_path / "detached.pid", tmp_path / "stop-detached"
    program = """
import os, pathlib, sys, time
marker, stop = map(pathlib.Path, sys.argv[1:])
if os.fork() == 0:
    os.setsid()
    marker.write_text(str(os.getpid()))
    while not stop.exists():
        time.sleep(0.01)
    os._exit(0)
while not marker.exists():
    time.sleep(0.01)
time.sleep(60)
"""
    request = ProcessRequest(
        (sys.executable, "-c", program, str(marker), str(stop)), timeout_seconds=0.4
    )
    try:
        with _scope_context(scope, run, _persist(run)):
            with pytest.raises(asyncio.TimeoutError):
                await environment.arun(request)
            receipts = run.get_extension("command_receipts")
            assert [receipt["state"] for receipt in receipts] == ["unknown"]
            with pytest.raises(TaskPauseRequestedError):
                await environment.arun(ProcessRequest(("must-not-execute",)))
    finally:
        # This deliberately detached test process belongs to the test host,
        # outside the executor's group. Release it even if an assertion fails.
        stop.touch()
        if marker.exists():
            pid = int(marker.read_text())
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                await asyncio.sleep(0.02)
            else:
                os.kill(pid, signal.SIGKILL)
                pytest.fail("Detached test process did not observe its stop marker")
