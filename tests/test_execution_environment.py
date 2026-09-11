import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.exceptions import AbortRequestedError, TaskPauseRequestedError
from msgflux.nn import Agent, ToolLibrary
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.runtime import (
    AbortSignal,
    AgentApprovals,
    InMemoryApprovalStore,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    PermissionSet,
    ProcessExecutor,
    ProcessRequest,
    ProcessResult,
    SandboxCapabilities,
    execution_context,
    get_execution_scope,
)
from msgflux.runtime.workspace import workspace_path
from msgflux.tools.config import tool_config


def authorized(filesystem, *requirements, executor=None, signal=None):
    environment = ExecutionEnvironment(filesystem, process_executor=executor)
    scope = ExecutionScope(
        principal="user",
        environment=environment,
        abort_signal=signal,
        permissions=PermissionSet(
            ["process.execute"],
            resources=[
                filesystem.permission(path, action) for path, action in requirements
            ],
        ),
    )
    return execution_context(scope=scope)


@pytest.mark.parametrize(
    "path", ["relative", "/a/../b", "/a\\b", "//host/file", "/bad\0path"]
)
def test_virtual_paths_never_resolve_on_host(path):
    with pytest.raises(ValueError):
        workspace_path(path)


def test_line_read_backend_hook_is_authorized_before_io():
    fs = InMemoryWorkspace("lines")
    fs._read_lines = Mock(return_value=b"selected\n")
    with authorized(fs):
        with pytest.raises(PermissionError):
            fs.read_lines("/input", offset=500, limit=1)
    fs._read_lines.assert_not_called()
    with authorized(fs, ("/input", "filesystem.read")):
        assert fs.read_lines("/input", offset=500, limit=1) == b"selected\n"
    fs._read_lines.assert_called_once_with("/input", 500, 1, 1_000_000)


@pytest.mark.asyncio
async def test_async_line_read_limits_and_empty_files():
    fs = InMemoryWorkspace("lines", {"/input": b"a\nb\nc", "/empty": b""})
    with authorized(fs, ("/input", "filesystem.read"), ("/empty", "filesystem.read")):
        assert await fs.aread_lines("/input", offset=2, limit=1, max_bytes=2) == b"b\n"
        assert await fs.aread_lines("/empty") == b""
        with pytest.raises(ValueError):
            await fs.aread_lines("/input", limit=True)
        with pytest.raises(ValueError):
            await fs.aread_lines("/input", max_bytes=1)


@pytest.mark.parametrize("data", [b"long", b"a\nb", "not bytes"])
def test_line_read_backend_cannot_exceed_limits(data):
    fs = InMemoryWorkspace("lines")
    fs._read_lines = Mock(return_value=data)
    with authorized(fs, ("/input", "filesystem.read")):
        with pytest.raises(ValueError, match="Backend"):
            fs.read_lines("/input", limit=1, max_bytes=3)


def test_memory_workspace_operations_and_exact_grants():
    fs = InMemoryWorkspace("w", {"/workspace/input": b"hello"})
    with authorized(
        fs,
        ("/workspace", "filesystem.list"),
        ("/workspace/input", "filesystem.read"),
        ("/workspace/output", "filesystem.write"),
        ("/workspace/new", "filesystem.mkdir"),
        ("/workspace/output", "filesystem.delete"),
    ):
        assert fs.read_text("/workspace/./input") == "hello"
        assert fs.listdir("/workspace") == ("input",)
        fs.write_text("/workspace/output", "result")
        fs.mkdir("/workspace/new")
        assert fs.listdir("/workspace") == ("input", "new", "output")
        with pytest.raises(PermissionError):
            fs.read_text("/workspace/output")
        with pytest.raises(PermissionError):
            fs.write_text("/workspace/new/child", "not inherited")
        fs.unlink("/workspace/output")
        assert fs.listdir("/workspace") == ("input", "new")
    with pytest.raises(PermissionError):
        fs.read_text("/workspace/input")


def test_workspace_identity_and_nested_authority():
    fs = InMemoryWorkspace("w", {"/a": b"secret"})
    other = InMemoryWorkspace("w", {"/a": b"different"})
    with authorized(fs, ("/a", "filesystem.read")) as scope:
        with pytest.raises(PermissionError, match="bound"):
            other.read_text("/a")
        with pytest.raises(ValueError, match="environment"):
            with execution_context(
                scope=replace(scope, environment=ExecutionEnvironment(other))
            ):
                pass
        with execution_context(scope=ExecutionScope(permissions=PermissionSet())):
            assert get_execution_scope().environment is scope.environment
            with pytest.raises(PermissionError):
                fs.read_text("/a")
        serialized = scope.to_dict()
        assert "environment" not in serialized
        assert "permissions" not in serialized
    with execution_context(scope=ExecutionScope(**serialized)):
        with pytest.raises(PermissionError):
            fs.read_text("/a")


@pytest.mark.asyncio
async def test_injected_filesystem_has_dynamic_authorization_sync_and_async():
    fs = InMemoryWorkspace("w", {"/allowed": b"yes", "/denied": b"no"})

    @tool_config(runtime_inputs=["filesystem"], retry=False)
    async def read_file(path: str, *, filesystem) -> str:
        """Read an authorized virtual file."""
        return await filesystem.aread_text(path)

    library = ToolLibrary("files", [read_file])
    assert (
        "filesystem"
        not in library.get_tool_definition("read_file").input_schema["properties"]
    )
    with authorized(fs, ("/allowed", "filesystem.read")):
        assert await library.arun("read_file", {"path": "/allowed"}) == "yes"
        assert library.run("read_file", {"path": "/allowed"}) == "yes"
        with pytest.raises(ValueError, match="both visible and runtime"):
            await library.arun(
                "read_file", {"path": "/allowed", "filesystem": "forged"}
            )
        with pytest.raises(PermissionError, match="resource permissions"):
            await library.arun("read_file", {"path": "/denied"})
    with pytest.raises(RuntimeError, match="unavailable"):
        await library.arun("read_file", {"path": "/allowed"}, vars={"filesystem": fs})


@pytest.mark.asyncio
async def test_async_vfs_and_abort_before_effect():
    fs = InMemoryWorkspace("w")
    signal = AbortSignal()
    with authorized(
        fs,
        ("/a", "filesystem.mkdir"),
        ("/a/file", "filesystem.write"),
        ("/a/file", "filesystem.read"),
        ("/a/file", "filesystem.delete"),
        ("/a", "filesystem.list"),
        signal=signal,
    ):
        await fs.amkdir("/a")
        await fs.awrite_bytes("/a/file", b"data")
        assert await fs.aread_bytes("/a/file") == b"data"
        assert await fs.alistdir("/a") == ("file",)
        signal.abort("stop")
        with pytest.raises(AbortRequestedError):
            await fs.awrite_text("/a/file", "changed")
    with authorized(
        fs, ("/a/file", "filesystem.read"), ("/a/file", "filesystem.delete")
    ):
        assert await fs.aread_text("/a/file") == "data"
        await fs.aunlink("/a/file")
        with pytest.raises(FileNotFoundError):
            await fs.aread_bytes("/a/file")


class FakeExecutor(ProcessExecutor):
    def __init__(self, *, supported=True, capabilities=None, block=False):
        self.supported = supported
        self._capabilities = capabilities or SandboxCapabilities(
            {"filesystem", "network", "process", "resource_limits"}
        )
        self.calls = []
        self.block = block
        self.started = asyncio.Event()
        self.cleaned = False

    @property
    def capabilities(self):
        return self._capabilities

    def supports_workspace(self, filesystem):
        return self.supported

    async def execute(self, request, **context):
        self.calls.append((request, context))
        self.started.set()
        try:
            if self.block:
                await asyncio.Future()
            return ProcessResult(0, stdout=b"done")
        finally:
            self.cleaned = True


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["absent", "incompatible", "incapable", "no_grants"])
async def test_process_preflight_never_falls_back_to_host(mode):
    fs = InMemoryWorkspace("w")
    executor = (
        None
        if mode == "absent"
        else FakeExecutor(
            supported=mode != "incompatible",
            capabilities=SandboxCapabilities() if mode == "incapable" else None,
        )
    )
    with authorized(fs, executor=executor) as scope:
        if mode == "no_grants":
            with execution_context(scope=replace(scope, permissions=PermissionSet())):
                with pytest.raises(PermissionError):
                    await scope.environment.arun(ProcessRequest(["bash"]))
        else:
            with pytest.raises(PermissionError):
                await scope.environment.arun(ProcessRequest(["bash"]))
    if executor is not None:
        assert not executor.calls


@pytest.mark.asyncio
async def test_process_gets_same_workspace_and_live_authority():
    fs, executor = InMemoryWorkspace("w"), FakeExecutor()
    with authorized(fs, ("/a", "filesystem.read"), executor=executor) as scope:
        result = await scope.environment.arun(ProcessRequest(["program"], cwd="/"))
        assert result.stdout == b"done"
        _, context = executor.calls[0]
        assert context["filesystem"] is fs
        assert context["permissions"] is scope.permissions
        assert context["requirements"] is scope.environment.requirements


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cancel", "abort", "timeout"])
async def test_process_cancellation_cleanup(mode):
    fs, executor, signal = (
        InMemoryWorkspace("w"),
        FakeExecutor(block=True),
        AbortSignal(),
    )
    with authorized(fs, executor=executor, signal=signal) as scope:
        task = asyncio.create_task(
            scope.environment.arun(
                ProcessRequest(
                    ["program"],
                    timeout_seconds=0.02 if mode == "timeout" else 5,
                )
            )
        )
        await asyncio.wait_for(executor.started.wait(), timeout=2)
        if mode == "cancel":
            task.cancel()
            expected = asyncio.CancelledError
        elif mode == "abort":
            signal.abort("stop")
            expected = AbortRequestedError
        else:
            expected = asyncio.TimeoutError
        with pytest.raises(expected):
            await task
        assert executor.cleaned


def test_request_validation_and_initial_file_conflicts():
    for argv in ("bash -c x", [], [None], ["program", "bad\0arg"]):
        with pytest.raises((TypeError, ValueError)):
            ProcessRequest(argv)
    with pytest.raises(ValueError):
        ProcessRequest(["program"], timeout_seconds=float("inf"))
    with pytest.raises(ValueError):
        InMemoryWorkspace("w", {"/a": b"file", "/a/child": b"child"})
    with pytest.raises(ValueError):
        InMemoryWorkspace("w", {"/a/child": b"child", "/a": b"file"})


@pytest.mark.asyncio
async def test_environment_injection_and_output_limit():
    fs, executor = InMemoryWorkspace("w"), FakeExecutor()

    @tool_config(runtime_inputs=["environment"], retry=False)
    async def run_program(*, environment) -> str:
        """Demonstrate process context injection using a test executor."""
        result = await environment.arun(ProcessRequest(["program"]))
        return result.stdout.decode()

    library = ToolLibrary("processes", [run_program])
    with authorized(fs, executor=executor) as scope:
        with pytest.raises(ValueError, match="both visible and runtime"):
            await library.arun("run_program", {"environment": "forged"})
        assert executor.calls == []
        assert await library.arun("run_program", {}) == "done"
        with pytest.raises(RuntimeError, match="output limit"):
            await scope.environment.arun(
                ProcessRequest(["program"], max_output_bytes=1)
            )


@pytest.mark.asyncio
async def test_agent_resumes_with_live_workspace_not_checkpoint_authority():
    fs = InMemoryWorkspace("w", {"/report": b"report"})
    observed = []

    @tool_config(runtime_inputs=["filesystem"], retry=False)
    async def read_file(path: str, *, filesystem) -> str:
        """Read a file after host approval."""
        observed.append(filesystem)
        return await filesystem.aread_text(path)

    checkpoint, journal = InMemoryCheckpointStore(), InMemoryApprovalStore()
    model = Mock(model_type="chat_completion")
    agent = Agent(
        name="reader",
        model=model,
        tools=[read_file],
        checkpoint_store=checkpoint,
        approvals=AgentApprovals(journal, {"read_file": "v1"}, "p1"),
    )
    calls = ToolCallAggregator()
    calls.process(0, "read:1", "read_file", '{"path":"/report"}')
    response, final = ModelResponse(), ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    final.set_response_type("text_generation")
    final.add("done")
    agent.generator.aforward = AsyncMock(side_effect=[response, final])
    scope = ExecutionScope(
        namespace="reader",
        thread_id="t",
        run_id="r",
        principal="user",
        environment=ExecutionEnvironment(fs),
        permissions=PermissionSet(
            resources=[fs.permission("/report", "filesystem.read")]
        ),
    )
    with pytest.raises(TaskPauseRequestedError):
        await agent.acall("read", scope=scope)
    assert observed == []
    stored_scope = checkpoint.load_state("reader", "t", "r")["scope"]
    assert "environment" not in stored_scope
    assert "permissions" not in stored_scope
    request = journal.pending("reader", "t", "r")[0]
    agent.decide_approval(request.request_id, approved=True, decided_by="host")
    assert await agent.acall("", scope=scope) == "done"
    assert observed == [fs]
    assert agent.generator.aforward.call_count == 2
