import asyncio
import os
import shutil
import subprocess

import pytest

from msgflux.runtime import (
    AbortSignal,
    ExecutionEnvironment,
    ExecutionScope,
    LocalWorkspace,
    PermissionSet,
    ProcessRequest,
    execution_context,
)
from msgflux.runtime.docker_executor import DockerLimits, DockerProcessExecutor
from msgflux.runtime.process_capture import ProcessOutputLimitError


@pytest.fixture
def docker_scope(tmp_path):
    if os.environ.get("MSGFLUX_TEST_DOCKER") != "1":
        pytest.skip("Set MSGFLUX_TEST_DOCKER=1 for real isolated process tests")
    image = subprocess.check_output(  # noqa: S603 -- fixed host-owned image lookup
        [
            shutil.which("docker"),
            "image",
            "inspect",
            "python:3.12-slim",
            "--format",
            "{{.Id}}",
        ],
        text=True,
    ).strip()
    root = tmp_path / "workspace"
    root.mkdir()
    fs = LocalWorkspace("docker-test", root)
    executor = DockerProcessExecutor(fs, image=image)
    environment = ExecutionEnvironment(
        fs, executor, write_guarantee="cooperative_compare"
    )
    scope = ExecutionScope(
        environment=environment,
        permissions=PermissionSet(
            ["process.execute"], [fs.permission("/", "process.workspace")]
        ),
    )
    return scope, root


@pytest.mark.asyncio
async def test_real_container_writes_and_enforces_isolation(docker_scope, tmp_path):
    scope, root = docker_scope
    outside = tmp_path / "private"
    outside.write_text("not visible")
    code = (
        "import os,socket,pathlib; "
        "pathlib.Path('created').write_text('real container'); "
        f"assert not pathlib.Path({str(outside)!r}).exists(); "
        "assert os.getuid()!=0; "
        "assert pathlib.Path('/sys/fs/cgroup/memory.max').read_text().strip()=='268435456'; "
        "assert pathlib.Path('/sys/fs/cgroup/pids.max').read_text().strip()=='64'; "
        "s=socket.socket(); s.settimeout(.2); "
        "assert s.connect_ex(('192.0.2.1',443))!=0; "
        "print('isolated')"
    )
    with execution_context(scope=scope):
        result = await scope.environment.arun(ProcessRequest(("python", "-c", code)))
    assert result.returncode == 0, result.stderr
    assert result.stdout == b"isolated\n"
    assert (root / "created").read_text() == "real container"
    assert outside.read_text() == "not visible"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["quota", "timeout", "cancel", "sink", "early_timeout"]
)
async def test_real_container_failure_cleanup(docker_scope, mode):
    scope, root = docker_scope
    ready = asyncio.Event()
    code = (
        "import subprocess,time; "
        "subprocess.Popen(['python','-c',\"import time,pathlib; time.sleep(2); pathlib.Path('leaked').touch()\"]); "
        "print('ready',flush=True); "
        + ("print('x'*100000,flush=True); " if mode == "quota" else "")
        + "time.sleep(30)"
    )

    async def output(channel, data):
        if b"ready" in data:
            ready.set()
            if mode == "sink":
                raise OSError("sink failed")

    with execution_context(scope=scope):
        task = asyncio.create_task(
            scope.environment.arun(
                ProcessRequest(
                    ("python", "-c", code),
                    timeout_seconds=(
                        0.001
                        if mode == "early_timeout"
                        else 1
                        if mode == "timeout"
                        else 10
                    ),
                    max_output_bytes=128 if mode == "quota" else 65536,
                ),
                on_output=output,
            )
        )
        if mode == "cancel":
            await asyncio.wait_for(ready.wait(), 10)
            task.cancel()
        expected = {
            "quota": ProcessOutputLimitError,
            "timeout": TimeoutError,
            "cancel": asyncio.CancelledError,
            "sink": OSError,
            "early_timeout": TimeoutError,
        }[mode]
        with pytest.raises(expected):
            await task
    await asyncio.sleep(2.1)
    assert not (root / "leaked").exists()


def test_limits_and_exact_workspace_grant(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda value: "/usr/bin/docker")
    fs = LocalWorkspace("unit", tmp_path)
    executor = DockerProcessExecutor(fs, image="trusted")
    assert executor.supports_workspace(fs)
    with pytest.raises(ValueError):
        DockerLimits(memory_bytes=0)


@pytest.mark.asyncio
async def test_real_container_memory_limit_is_enforced(docker_scope):
    scope, _ = docker_scope
    with execution_context(scope=scope):
        result = await scope.environment.arun(
            ProcessRequest(
                ("python", "-c", "x=bytearray(512*1024*1024); print('unexpected')")
            )
        )
    assert result.returncode == 137
    assert b"unexpected" not in result.stdout


@pytest.mark.asyncio
async def test_real_backend_bash_output_can_be_offloaded(tmp_path):
    if os.environ.get("MSGFLUX_TEST_DOCKER") != "1":
        pytest.skip("Set MSGFLUX_TEST_DOCKER=1")
    from msgflux.runtime import DockerWorkspaceBackend, LocalToolResultStore
    from msgflux.runtime.shell_capture import ShellOutputCapture
    from msgflux.tools.builtin import BashTool
    import msgspec

    root = tmp_path / "workspace"
    root.mkdir()
    backend = DockerWorkspaceBackend(root, image="python:3.12-slim")
    store = LocalToolResultStore(tmp_path / "results", max_store_bytes=2_000_000)
    capture = ShellOutputCapture(
        store, max_inline_bytes=1024, preview_bytes=128, max_capture_bytes=2_000_000
    )
    async with await backend.open("bash") as binding:
        environment = ExecutionEnvironment.from_binding(
            binding, write_guarantee="cooperative_compare"
        )
        scope = ExecutionScope(
            environment=environment,
            permissions=PermissionSet(
                ["process.execute"],
                [binding.filesystem.permission("/", "process.workspace")],
            ),
        )
        with execution_context(scope=scope):
            result = await BashTool().acall(
                "printf 'real bash' > result.txt; python -c \"print('x'*1048576,end='')\"",
                environment=environment,
                shell_capture=capture,
            )
        assert result.output_reference is not None
        assert len(result.results[0].stdout) <= 128
        store.verify(result.output_reference)
        restored = msgspec.json.decode(
            store.read(result.output_reference, limit=2_000_000)
        )
        assert len(restored["results"][0]["stdout"]) == 1048576
        assert (root / "result.txt").read_text() == "real bash"


@pytest.mark.asyncio
async def test_individual_file_grants_do_not_authorize_mount(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda value: "/usr/bin/docker")
    fs = LocalWorkspace("unit", tmp_path)
    executor = DockerProcessExecutor(fs, image="trusted")
    environment = ExecutionEnvironment(fs, executor)
    with (
        execution_context(
            scope=ExecutionScope(
                environment=environment,
                permissions=PermissionSet(
                    ["process.execute"], [fs.permission("/", "filesystem.read")]
                ),
            )
        ),
        pytest.raises(PermissionError, match="whole-workspace"),
    ):
        await environment.arun(ProcessRequest(("true",)))


@pytest.mark.asyncio
async def test_real_agent_stream_checkpoints_only_offload_reference(
    docker_scope, tmp_path
):
    from unittest.mock import AsyncMock, Mock
    from dataclasses import replace
    import msgspec
    from msgflux.data.stores import SQLiteCheckpointStore
    from msgflux.models.response import ModelResponse
    from msgflux.models.tool_call_agg import ToolCallAggregator
    from msgflux.nn import Agent
    from msgflux.nn.extensions import ToolOutputOffloadExtension
    from msgflux.runtime import LocalToolResultStore, get_tool_result_reference
    from msgflux.tools.builtin import BashTool
    from msgflux.utils.msgspec import msgspec_dumps

    scope, _ = docker_scope
    scope = replace(scope, thread_id="thread", run_id="run")
    store = LocalToolResultStore(tmp_path / "results")
    checkpoints = SQLiteCheckpointStore(str(tmp_path / "checkpoint.sqlite"))
    calls = ToolCallAggregator()
    calls.process(
        0,
        "real-shell",
        "bash",
        msgspec_dumps({"command": "python -c \"print('x'*1048576,end='')\""}),
    )
    first, last = ModelResponse(), ModelResponse()
    first.set_response_type("tool_call")
    first.add(calls)
    last.set_response_type("text_generation")
    last.add("done")
    agent = Agent(
        name="container-agent",
        model=Mock(model_type="chat_completion"),
        tools=[BashTool()],
        checkpoint_store=checkpoints,
    )
    extension = ToolOutputOffloadExtension(
        store, max_inline_bytes=1024, preview_bytes=128, max_capture_bytes=2_000_000
    )
    agent.tool_library.register_extension(extension.name, extension)
    agent.generator.aforward = AsyncMock(side_effect=[first, last])
    try:
        events = [event async for event in agent.stream_events("run", scope=scope)]
        output = next(event for event in events if event.type == "tool.end")
        ref = get_tool_result_reference(output.data["result"])
        assert ref is not None
        store.verify(ref)
        assert len(msgspec.json.encode(events)) < 100_000
        state = checkpoints.load_state("container-agent", "thread", "run")
        assert ref.result_id.encode() in msgspec.json.encode(state)
        assert b"x" * 10000 not in msgspec.json.encode(state)
        assert events[-1].type == "run.end"
    finally:
        checkpoints.close()
