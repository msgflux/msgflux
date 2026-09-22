import asyncio
import math
import os
import signal as signal_module
import sys
from unittest.mock import Mock

import pytest

from msgflux.exceptions import AbortRequestedError
from msgflux.runtime import (
    AbortSignal,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    PermissionSet,
    ProcessExecutor,
    ProcessRequest,
    ProcessResult,
    SandboxCapabilities,
    execution_context,
)
from msgflux.runtime.process_capture import (
    ProcessOutputLimitError,
    drain_subprocess,
)


class StreamingExecutor(ProcessExecutor):
    @property
    def capabilities(self):
        return SandboxCapabilities(
            {"filesystem", "network", "process", "resource_limits"}
        )

    def supports_workspace(self, filesystem):
        return True

    async def execute(self, request, **kwargs):
        return ProcessResult(0, b"buffered", b"")


class DuplicateStreamingExecutor(StreamingExecutor):
    async def execute_stream(self, request, **kwargs):
        await kwargs["on_output"]("stdout", b"streamed")
        return ProcessResult(0, b"duplicate", b"")


@pytest.mark.asyncio
async def test_environment_streaming_compatibility_fallback_delivers_chunks():
    environment = ExecutionEnvironment(InMemoryWorkspace("stream"), StreamingExecutor())
    seen = []
    scope = ExecutionScope(
        environment=environment,
        permissions=PermissionSet(["process.execute"]),
    )

    async def collect(channel, data):
        seen.append((channel, data))

    with execution_context(scope=scope):
        result = await environment.arun(
            ProcessRequest(("ignored",)),
            on_output=collect,
        )
    assert seen == [("stdout", b"buffered")]
    assert result == ProcessResult(0)


@pytest.mark.asyncio
async def test_environment_rejects_duplicate_streamed_buffers():
    environment = ExecutionEnvironment(
        InMemoryWorkspace("stream"), DuplicateStreamingExecutor()
    )
    scope = ExecutionScope(
        environment=environment,
        permissions=PermissionSet(["process.execute"]),
    )

    async def ignore(channel, data):
        return None

    with execution_context(scope=scope), pytest.raises(RuntimeError, match="duplicate"):
        await environment.arun(
            ProcessRequest(("ignored",)),
            on_output=ignore,
        )


@pytest.mark.asyncio
async def test_drain_subprocess_drains_both_pipes_incrementally():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('out'*1000); sys.stderr.write('err'*1000)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    chunks = []

    async def collect(channel, data):
        chunks.append((channel, data))

    code = await drain_subprocess(
        process,
        collect,
        max_output_bytes=10000,
        timeout_seconds=5,
        chunk_size=127,
    )
    assert code == 0
    assert (
        b"".join(data for channel, data in chunks if channel == "stdout")
        == b"out" * 1000
    )
    assert (
        b"".join(data for channel, data in chunks if channel == "stderr")
        == b"err" * 1000
    )
    assert len(chunks) > 2


@pytest.mark.asyncio
async def test_drain_subprocess_kills_and_reaps_on_limit():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys,time; sys.stdout.write('x'*1000000); sys.stdout.flush(); time.sleep(30)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def ignore(channel, data):
        return None

    with pytest.raises(ProcessOutputLimitError):
        await drain_subprocess(
            process,
            ignore,
            max_output_bytes=1024,
            timeout_seconds=5,
            chunk_size=128,
        )
    assert process.returncode is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["callback", "timeout", "cancel"])
async def test_drain_failure_stops_callbacks_and_reaps_child(failure, monkeypatch):
    import msgflux.runtime.process_capture as module

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import os,time; os.write(1,b'ready'); os.write(2,b'error'); time.sleep(30)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    callbacks = []
    received = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()
    cleanup = module._cleanup_process

    async def controlled_cleanup(*args, **kwargs):
        cleanup_started.set()
        if failure == "cancel":
            await cleanup_release.wait()
        await cleanup(*args, **kwargs)

    monkeypatch.setattr(module, "_cleanup_process", controlled_cleanup)

    async def consume(channel, data):
        callbacks.append((channel, data))
        received.set()
        if failure == "callback":
            raise OSError("sink failed")
        await asyncio.Event().wait()

    task = asyncio.create_task(
        drain_subprocess(
            process,
            consume,
            max_output_bytes=1024,
            timeout_seconds=0.2 if failure == "timeout" else 10,
        )
    )
    await asyncio.wait_for(received.wait(), 5)
    if failure == "cancel":
        task.cancel()
        await asyncio.wait_for(cleanup_started.wait(), 5)
        task.cancel()  # cleanup must survive repeated external cancellation
        cleanup_release.set()
    expected = {
        "callback": OSError,
        "timeout": asyncio.TimeoutError,
        "cancel": asyncio.CancelledError,
    }[failure]
    with pytest.raises(expected):
        await asyncio.wait_for(task, 5)
    assert process.returncode is not None
    count = len(callbacks)
    await asyncio.sleep(0)
    assert len(callbacks) == count


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [True, 0, -1, math.inf, math.nan])
async def test_drain_rejects_invalid_deadline_before_taking_ownership(timeout):
    async def ignore(channel, data):
        pass

    with pytest.raises(ValueError, match="timeout_seconds"):
        await drain_subprocess(
            Mock(),
            ignore,
            max_output_bytes=128,
            timeout_seconds=timeout,
        )


@pytest.mark.asyncio
async def test_drain_subprocess_honors_abort_and_reaps_child():
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(30)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    signal = AbortSignal()

    async def ignore(channel, data):
        return None

    task = asyncio.create_task(
        drain_subprocess(
            process,
            ignore,
            max_output_bytes=1024,
            timeout_seconds=10,
            abort_signal=signal,
        )
    )
    await asyncio.sleep(0.05)
    signal.abort("stop")
    with pytest.raises(AbortRequestedError):
        await task
    assert process.returncode is not None


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX process signals")
@pytest.mark.parametrize("owns_group", [False, True])
async def test_cleanup_escalates_when_child_ignores_termination(owns_group):
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "os.write(1,b'ready'); time.sleep(30)",
        env={},
        start_new_session=owns_group,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def reject(channel, data):
        raise OSError("sink rejected output")

    with pytest.raises(OSError, match="sink rejected"):
        await asyncio.wait_for(
            drain_subprocess(
                process,
                reject,
                max_output_bytes=1024,
                timeout_seconds=10,
                owns_process_group=owns_group,
            ),
            5,
        )
    assert process.returncode == -signal_module.SIGKILL
