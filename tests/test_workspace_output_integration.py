"""Real fixed subprocesses through the workspace tool and offload runtime.

The executor below is a test fixture, not a sandbox: it ignores model commands
and runs only a constant Python program, with an empty inherited environment.
"""

import asyncio
import os
import sys
import tracemalloc
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent, ToolLibrary
from msgflux.nn.extensions import ToolOutputOffloadExtension
from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    PermissionSet,
    ProcessExecutor,
    ProcessResult,
    RuntimeResources,
    SandboxCapabilities,
    SandboxRequirements,
    drain_subprocess,
    execution_context,
    get_tool_result_reference,
)

from msgflux.tools.builtin import BashTool

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX local store")


class FixedSubprocessExecutor(ProcessExecutor):
    def __init__(self, blocks):
        self.blocks = blocks
        self.processes = []

    @property
    def capabilities(self):
        return SandboxCapabilities()

    def supports_workspace(self, filesystem):
        return isinstance(filesystem, InMemoryWorkspace)

    async def execute(self, request, **kwargs):
        raise AssertionError("Expected incremental execution")

    async def execute_stream(self, request, *, on_output, **kwargs):
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import os,sys\n"
            "for _ in range(int(sys.argv[1])):\n"
            " os.write(1,b'x'*32768)\n"
            " os.write(2,b'e'*256)\n"
            "sys.exit(7)\n",
            str(self.blocks),
            env={},
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.processes.append(process)
        code = await drain_subprocess(
            process,
            on_output,
            max_output_bytes=request.max_output_bytes,
            timeout_seconds=request.timeout_seconds,
            abort_signal=kwargs["abort_signal"],
            owns_process_group=True,
        )
        return ProcessResult(code)


def setup_capture(tmp_path, blocks):
    resources = RuntimeResources(tmp_path).initialize()
    results = resources.tool_result_store()
    executor = FixedSubprocessExecutor(blocks)
    environment = ExecutionEnvironment(
        InMemoryWorkspace("fixture"),
        executor,
        requirements=SandboxRequirements(),
    )
    extension = ToolOutputOffloadExtension(
        results,
        max_inline_bytes=8192,
        preview_bytes=128,
        max_capture_bytes=16 * 1024 * 1024,
    )
    scope = ExecutionScope(
        namespace="fixture",
        thread_id="fixture",
        run_id="run",
        environment=environment,
        permissions=PermissionSet(["process.execute"]),
    )
    return resources, results, executor, extension, scope


@pytest.mark.asyncio
async def test_actual_process_agent_events_checkpoint_and_next_model_turn(tmp_path):
    resources, results, executor, extension, scope = setup_capture(tmp_path, 64)
    checkpoints = resources.checkpoint_store("fixture")
    calls = ToolCallAggregator()
    calls.process(0, "call", "bash", '{"command":"fixture only"}')
    first = ModelResponse()
    first.set_response_type("tool_call")
    first.add(calls)
    final = ModelResponse()
    final.set_response_type("text_generation")
    final.add("done")
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="fixture", model=model, tools=[BashTool()], checkpoint_store=checkpoints
    )
    agent.tool_library.register_extension(extension.name, extension)
    agent.generator.aforward = AsyncMock(side_effect=[first, final])
    try:
        events = [event async for event in agent.stream_events("run", scope=scope)]
        event = next(event for event in events if event.type == "tool.end")
        ref = get_tool_result_reference(event.data["result"])
        assert ref is not None
        assert len(msgspec.json.encode(events)) < 100_000
        results.verify(ref)
        complete = msgspec.json.decode(results.read(ref, limit=3 * 1024 * 1024))
        assert complete["results"][0]["stdout"] == "x" * (64 * 32768)
        assert complete["results"][0]["stderr"] == "e" * (64 * 256)
        assert complete["results"][0]["returncode"] == 7
        state = checkpoints.load_state("fixture", "fixture", "run")
        assert state["status"] == "completed"
        encoded = msgspec.json.encode(state)
        assert ref.result_id.encode() in encoded
        assert b"x" * 1000 not in encoded
        messages = agent.generator.aforward.call_args.kwargs["messages"]
        assert ref.result_id in str(messages.to_chatml())
        assert "x" * 1000 not in str(messages.to_chatml())
        assert all(process.returncode == 7 for process in executor.processes)
    finally:
        checkpoints.close()


@pytest.mark.asyncio
async def test_capture_memory_does_not_scale_with_complete_output(
    tmp_path, record_property
):
    async def capture(blocks):
        _, results, executor, extension, scope = setup_capture(
            tmp_path / str(blocks), blocks
        )
        library = ToolLibrary("fixture", [BashTool()], extensions=[extension])
        with execution_context(scope=scope):
            tracemalloc.start()
            try:
                response = await library.acall(
                    [("call", "bash", {"command": "fixture"})]
                )
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
        ref = get_tool_result_reference(response.tool_calls[0].result)
        assert ref is not None
        results.verify(ref)  # hash incrementally, never join the large output
        assert all(process.returncode == 7 for process in executor.processes)
        return peak, ref.size_bytes

    await capture(1)  # warm up runtime/thread-pool paths
    small_peak, small_size = await capture(32)
    large_peak, large_size = await capture(256)
    record_property("small_capture_peak_bytes", small_peak)
    record_property("large_capture_peak_bytes", large_peak)
    assert large_size > small_size * 7
    # Generous allocator/scheduling allowance, still rejects an 8 MiB retained copy.
    assert large_peak < small_peak + 3 * 1024 * 1024


@pytest.mark.asyncio
async def test_permission_denial_never_launches_process(tmp_path):
    _, results, executor, extension, scope = setup_capture(tmp_path, 1)
    library = ToolLibrary("fixture", [BashTool()], extensions=[extension])
    with execution_context(scope=scope.with_overrides(permissions=PermissionSet())):
        response = await library.acall([("call", "bash", {"command": "fixture"})])
    assert response.tool_calls[0].error
    assert not executor.processes
    assert list(results.root.iterdir()) == []
