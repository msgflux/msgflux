import asyncio
import os
import sys
import time

import pytest

from msgflux.exceptions import AbortRequestedError
from msgflux.runtime import (
    AbortSignal,
    ExecutionEnvironment,
    ExecutionScope,
    LocalProcessExecutor,
    LocalWorkspace,
    PermissionSet,
    ProcessOutputLimitError,
    ProcessRequest,
    SandboxRequirements,
    execution_context,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="LocalWorkspace requires POSIX"
)


def _environment(root, *, grants=("process.execute",), requirements=None):
    filesystem = LocalWorkspace("local-executor", root)
    executor = LocalProcessExecutor(filesystem)
    environment = ExecutionEnvironment(
        filesystem=filesystem,
        process_executor=executor,
        requirements=requirements or SandboxRequirements(),
        write_guarantee="cooperative_compare",
    )
    scope = ExecutionScope(
        namespace="test",
        thread_id="local-executor",
        run_id="local-executor",
        principal="test-user",
        environment=environment,
        permissions=PermissionSet(grants=grants),
    )
    return filesystem, executor, environment, scope


def _pid_program(pid_path):
    return (
        "import os,time; "
        f"open({str(pid_path)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )


def _assert_process_gone(pid):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    pytest.fail(f"child process {pid} remained alive after cleanup")


@pytest.mark.asyncio
async def test_local_executor_maps_workspace_cwd_and_runs_argv(tmp_path):
    (tmp_path / "nested").mkdir()
    filesystem, _, environment, scope = _environment(tmp_path)
    assert os.fspath(filesystem.host_root) == os.fspath(tmp_path)

    with execution_context(scope=scope):
        bash_result = await environment.arun(
            ProcessRequest(("/bin/bash", "-c", "pwd; exit 7"), cwd="/nested")
        )
        argv_result = await environment.arun(
            ProcessRequest(
                (
                    sys.executable,
                    "-c",
                    "import sys; print('|'.join(sys.argv[1:]))",
                    "first arg",
                    "second",
                ),
            )
        )

    assert bash_result.returncode == 7
    assert bash_result.stdout.strip() == os.fsencode(tmp_path / "nested")
    assert bash_result.stderr == b""
    assert argv_result.returncode == 0
    assert argv_result.stdout == b"first arg|second\n"


@pytest.mark.asyncio
async def test_local_executor_supports_streamed_and_buffered_results(tmp_path):
    _, _, environment, scope = _environment(tmp_path)
    request = ProcessRequest(
        (
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        )
    )
    chunks = []

    async def on_output(channel, data):
        chunks.append((channel, data))

    with execution_context(scope=scope):
        buffered = await environment.arun(request)
        streamed = await environment.arun(request, on_output=on_output)

    assert buffered.returncode == streamed.returncode == 0
    assert buffered.stdout == b"out\n"
    assert buffered.stderr == b"err\n"
    assert streamed.stdout == streamed.stderr == b""
    assert b"".join(data for channel, data in chunks if channel == "stdout") == b"out\n"
    assert b"".join(data for channel, data in chunks if channel == "stderr") == b"err\n"


@pytest.mark.asyncio
async def test_local_executor_rejects_foreign_workspace_and_isolation(tmp_path):
    filesystem, executor, _, scope = _environment(tmp_path)
    other = LocalWorkspace("other", tmp_path)
    foreign_environment = ExecutionEnvironment(
        filesystem=other,
        process_executor=executor,
        requirements=SandboxRequirements(),
        write_guarantee="cooperative_compare",
    )
    foreign_scope = ExecutionScope(
        namespace="test",
        thread_id="foreign",
        environment=foreign_environment,
        permissions=scope.permissions,
    )
    isolated_environment = ExecutionEnvironment(
        filesystem=filesystem,
        process_executor=executor,
        requirements=SandboxRequirements({"network"}),
        write_guarantee="cooperative_compare",
    )
    isolated_scope = ExecutionScope(
        namespace="test",
        thread_id="isolated",
        environment=isolated_environment,
        permissions=scope.permissions,
    )
    request = ProcessRequest((sys.executable, "-c", "print('should not run')"))

    with execution_context(scope=foreign_scope), pytest.raises(PermissionError):
        await foreign_environment.arun(request)
    with execution_context(scope=isolated_scope), pytest.raises(PermissionError):
        await isolated_environment.arun(request)


@pytest.mark.asyncio
async def test_local_executor_requires_process_permission(tmp_path):
    _, _, environment, scope = _environment(tmp_path, grants=())

    with execution_context(scope=scope), pytest.raises(PermissionError):
        await environment.arun(ProcessRequest((sys.executable, "-c", "pass")))


@pytest.mark.asyncio
async def test_local_executor_output_limit_terminates_child(tmp_path):
    pid_path = tmp_path / "pid"
    _, _, environment, scope = _environment(tmp_path)
    request = ProcessRequest(
        (
            sys.executable,
            "-c",
            f"import os,time; open({str(pid_path)!r}, 'w').write(str(os.getpid())); print('output', flush=True); time.sleep(30)",
        ),
        max_output_bytes=2,
    )

    with execution_context(scope=scope), pytest.raises(ProcessOutputLimitError):
        await environment.arun(request)

    assert pid_path.exists()
    _assert_process_gone(int(pid_path.read_text()))


@pytest.mark.asyncio
async def test_local_executor_timeout_and_abort_reap_children(tmp_path):
    _, _, environment, scope = _environment(tmp_path)
    timeout_pid = tmp_path / "timeout.pid"
    timeout_request = ProcessRequest(
        (sys.executable, "-c", _pid_program(timeout_pid)), timeout_seconds=1
    )
    with execution_context(scope=scope), pytest.raises(asyncio.TimeoutError):
        await environment.arun(timeout_request)
    assert timeout_pid.exists()
    _assert_process_gone(int(timeout_pid.read_text()))

    abort_pid = tmp_path / "abort.pid"
    abort_signal = AbortSignal()
    abort_scope = ExecutionScope(
        namespace="test",
        thread_id="abort",
        environment=environment,
        permissions=scope.permissions,
        abort_signal=abort_signal,
    )

    async def abort_soon():
        await asyncio.sleep(1)
        abort_signal.abort("test abort")

    with execution_context(scope=abort_scope):
        abort_task = asyncio.create_task(abort_soon())
        with pytest.raises(AbortRequestedError):
            await environment.arun(
                ProcessRequest((sys.executable, "-c", _pid_program(abort_pid)))
            )
        await abort_task
    assert abort_pid.exists()
    _assert_process_gone(int(abort_pid.read_text()))


@pytest.mark.asyncio
async def test_local_executor_task_cancellation_reaps_child(tmp_path):
    pid_path = tmp_path / "cancel.pid"
    _, _, environment, scope = _environment(tmp_path)

    with execution_context(scope=scope):
        task = asyncio.create_task(
            environment.arun(
                ProcessRequest((sys.executable, "-c", _pid_program(pid_path)))
            )
        )
        deadline = asyncio.get_running_loop().time() + 3
        while not pid_path.exists() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert pid_path.exists(), "child did not start"
        pid = int(pid_path.read_text())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    _assert_process_gone(pid)


@pytest.mark.asyncio
async def test_local_executor_cancellation_during_launch_reaps_child(
    tmp_path, monkeypatch
):
    pid_path = tmp_path / "launch.pid"
    _, _, environment, scope = _environment(tmp_path)
    original_create = asyncio.create_subprocess_exec
    process_created = asyncio.Event()
    allow_return = asyncio.Event()

    async def delayed_create(*args, **kwargs):
        process = await original_create(*args, **kwargs)
        deadline = asyncio.get_running_loop().time() + 3
        while not pid_path.exists() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert pid_path.exists(), "child did not start during delayed launch"
        process_created.set()
        await allow_return.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_create)
    with execution_context(scope=scope):
        task = asyncio.create_task(
            environment.arun(
                ProcessRequest((sys.executable, "-c", _pid_program(pid_path)))
            )
        )
        await asyncio.wait_for(process_created.wait(), timeout=3)
        pid = int(pid_path.read_text())
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done(), "launch cleanup should wait for the child handle"
        allow_return.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    _assert_process_gone(pid)
