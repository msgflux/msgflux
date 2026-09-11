import base64
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.nn import Agent, ToolLibrary
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    PermissionSet,
    ProcessExecutor,
    ProcessResult,
    SandboxCapabilities,
    execution_context,
)
from msgflux.tools.builtin import BashTool, ReadFileTool
from msgflux.runtime.agent_inbox import AgentInbox, InMemoryAgentInboxStore


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6"
    "AAAAAElFTkSuQmCC"
)


@pytest.mark.asyncio
async def test_reader_image_agent_trajectory():
    fs = InMemoryWorkspace("images", {"/image.png": PNG})
    store = InMemoryCheckpointStore()
    model = Mock(model_type="chat_completion")
    agent = Agent(
        name="viewer",
        model=model,
        tools=[ReadFileTool(supports_vision=True)],
        checkpoint_store=store,
    )
    calls = ToolCallAggregator()
    calls.process(0, "read:1", "read", '{"path":"/image.png"}')
    first, final = ModelResponse(), ModelResponse()
    first.set_response_type("tool_call")
    first.add(calls)
    final.set_response_type("text_generation")
    final.add("done")
    agent.generator.aforward = AsyncMock(side_effect=[first, final])
    scope = ExecutionScope(
        namespace="viewer",
        thread_id="t",
        run_id="r",
        environment=ExecutionEnvironment(fs),
        permissions=PermissionSet(
            resources=[fs.permission("/image.png", "filesystem.read")]
        ),
    )
    assert await agent.acall("Inspect the image", scope=scope) == "done"
    history = store.load_state("viewer", "t", "r")["messages"]["items"]
    output_index = next(
        i
        for i, item in enumerate(history)
        if item.get("type") == "function_call_output"
    )
    image_index = next(
        i
        for i, item in enumerate(history)
        if item.get("metadata", {}).get("inbox_ref") == "read:1"
    )
    assert image_index > output_index
    assert history[image_index]["role"] == "user"
    assert "base64" not in str(history[output_index])
    image = history[image_index]["content"][1]
    assert base64.b64decode(image["image_url"]["url"].split(",", 1)[1]) == PNG


def test_reader_is_single_public_tool_and_concatenates_guidance():
    from msgflux.tools import builtin

    assert not hasattr(builtin, "read_file")

    class ConfiguredReader(ReadFileTool):
        tool_config = {
            **ReadFileTool.tool_config,
            "usage_guidance": "Existing instruction.",
        }

    plain = ConfiguredReader()
    visual = ConfiguredReader(supports_vision=True)
    assert visual.tool_config["usage_guidance"].startswith("Existing instruction.\n\n")
    assert "user-role message" in visual.tool_config["usage_guidance"]
    assert plain.tool_config["usage_guidance"] == "Existing instruction."
    assert ReadFileTool().tool_config["usage_guidance"] is None

    class CustomReader(ReadFileTool):
        tool_config = {
            **ReadFileTool.tool_config,
            "usage_guidance": "Inherited instruction.",
        }

    assert (
        CustomReader(supports_vision=True)
        .tool_config["usage_guidance"]
        .startswith("Inherited instruction.")
    )
    with pytest.raises(TypeError, match="boolean"):
        ReadFileTool(supports_vision="yes")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["enabled", "disabled", "no_inbox", "denied"])
async def test_reader_image_publication(asynchronous, mode):
    fs = InMemoryWorkspace("images", {"/image.png": PNG})
    inbox = AgentInbox(store=InMemoryAgentInboxStore())
    scope = ExecutionScope(
        environment=ExecutionEnvironment(fs),
        permissions=PermissionSet(
            resources=[]
            if mode == "denied"
            else [fs.permission("/image.png", "filesystem.read")]
        ),
    )
    reader = ReadFileTool(supports_vision=mode != "disabled")
    library = ToolLibrary("reader", [reader])
    if mode == "no_inbox":
        # ToolLibrary normally supplies a default inbox even without an Agent.
        library.get_agent_inbox = Mock(return_value=None)
    definition = library.get_tool_definition("read")
    assert set(definition.input_schema["properties"]) == {"path", "offset", "limit"}
    assert definition.usage_guidance == reader.tool_config["usage_guidance"]

    async def invoke():
        if asynchronous:
            return await library.arun("read", {"path": "/image.png"})
        return library.run("read", {"path": "/image.png"})

    with execution_context(
        scope=scope, agent_inbox=None if mode == "no_inbox" else inbox
    ):
        if mode == "enabled":
            assert "user-role message" in await invoke()
        else:
            error = {
                "disabled": ValueError,
                "no_inbox": RuntimeError,
                "denied": PermissionError,
            }[mode]
            with pytest.raises(error):
                await invoke()
    if mode != "enabled":
        assert inbox.peek() == []
        return
    notifications = inbox.peek()
    assert len(notifications) == 1
    assert notifications[0].ref
    message = inbox.render(notifications)
    assert message["role"] == "user"
    assert "incoming_user_message" not in message["content"][0]["text"]
    uri = message["content"][1]["image_url"]["url"]
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == PNG


@pytest.mark.asyncio
async def test_read_guidance_is_per_instance_and_does_not_enable_vision(
    workspace_tools,
):
    _, scope, _ = workspace_tools
    first, second = ReadFileTool(), ReadFileTool()
    first.tool_config["usage_guidance"] = "Delegate image interpretation explicitly."
    second.tool_config["usage_guidance"] = (
        "Inspect images only when attached to the conversation."
    )
    for reader in (first, second, ReadFileTool()):
        library = ToolLibrary("reader", [reader])
        definition = library.get_tool_definition("read")
        assert definition.usage_guidance == reader.tool_config["usage_guidance"]
        assert "filesystem" not in definition.input_schema["properties"]
        with execution_context(scope=scope):
            assert library.run("read", {"path": "/input"}) == "hello"
            assert await library.arun("read", {"path": "/input"}) == "hello"
            with pytest.raises(UnicodeDecodeError):
                await library.arun("read", {"path": "/binary"})
    assert first.tool_config["usage_guidance"] != second.tool_config["usage_guidance"]


@pytest.fixture
def workspace_tools():
    fs = InMemoryWorkspace(
        "tools", {"/input": b"hello", "/binary": b"\xff", "/large": b"x" * 1_000_001}
    )
    executor = Mock(spec=ProcessExecutor)
    executor.capabilities = SandboxCapabilities(
        {"filesystem", "network", "process", "resource_limits"}
    )
    executor.supports_workspace.return_value = True
    executor.execute.return_value = ProcessResult(7, b"output\xff", b"error")
    scope = ExecutionScope(
        environment=ExecutionEnvironment(fs, executor),
        permissions=PermissionSet(
            ["process.execute"],
            [
                fs.permission(path, "filesystem.read")
                for path in ("/input", "/binary", "/large")
            ],
        ),
    )
    return ToolLibrary("workspace", [ReadFileTool(), BashTool()]), scope, executor


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_workspace_tools_invocation(workspace_tools, asynchronous):
    library, scope, executor = workspace_tools

    async def invoke(name, arguments):
        if asynchronous:
            return await library.arun(name, arguments)
        return library.run(name, arguments)

    with execution_context(scope=scope):
        assert await invoke("read", {"path": "/input"}) == "hello"
        result = await invoke("bash", {"command": "printf output; exit 7"})
        assert result.results[0].returncode == 7
        assert result.results[0].stdout == "output\ufffd"
        assert result.results[0].stderr == "error"
    request = executor.execute.call_args.args[0]
    assert request.argv == (
        "bash",
        "--noprofile",
        "--norc",
        "-c",
        "printf output; exit 7",
    )
    assert request.cwd == "/"
    assert request.timeout_seconds == 30
    assert request.max_output_bytes == 1_000_000
    assert (
        executor.execute.call_args.kwargs["filesystem"] is scope.environment.filesystem
    )
    assert executor.execute.call_args.kwargs["permissions"] is scope.permissions
    executor.execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,binding,args",
    [
        ("read", "filesystem", {"path": "/input"}),
        ("bash", "environment", {"command": "true"}),
    ],
)
async def test_runtime_dependencies_not_model_arguments(
    workspace_tools, name, binding, args
):
    library, scope, executor = workspace_tools
    schema = library.get_tool_definition(name).input_schema
    assert binding not in schema["properties"]
    assert "timeout_seconds" not in schema["properties"]
    with execution_context(scope=scope):
        # Public annotations exclude bindings; forged collisions fail closed.
        with pytest.raises(ValueError, match="both visible and runtime-provided"):
            await library.arun(name, {**args, binding: "forged"})
        executor.execute.assert_not_called()
        result = await library.arun(name, args)
    if name == "read":
        assert result == "hello"
        executor.execute.assert_not_called()
    else:
        assert result.results[0].returncode == 7
        assert (
            executor.execute.call_args.kwargs["filesystem"]
            is scope.environment.filesystem
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,error",
    [
        ("/ungranted", PermissionError),
        ("/../input", ValueError),
        ("/binary", UnicodeDecodeError),
        ("/large", ValueError),
    ],
)
async def test_read_file_rejects_invalid_or_unauthorized_content(
    workspace_tools, path, error
):
    library, scope, _ = workspace_tools
    with execution_context(scope=scope):
        with pytest.raises(error):
            await library.arun("read", {"path": path})


@pytest.mark.asyncio
async def test_missing_live_context_cannot_be_supplied_by_vars(workspace_tools):
    library, scope, executor = workspace_tools
    with pytest.raises(RuntimeError, match="unavailable"):
        await library.arun(
            "read",
            {"path": "/input"},
            vars={"filesystem": scope.environment.filesystem},
        )
    with execution_context(scope=ExecutionScope(environment=scope.environment)):
        with pytest.raises(RuntimeError, match=r"process\.execute"):
            await library.arun("bash", {"command": "true"})
    executor.execute.assert_not_called()


@pytest.mark.asyncio
async def test_bash_requires_executor_and_never_retries(workspace_tools):
    library, scope, executor = workspace_tools
    with execution_context(
        scope=ExecutionScope(
            environment=ExecutionEnvironment(scope.environment.filesystem),
            permissions=scope.permissions,
        )
    ):
        with pytest.raises(PermissionError, match="No isolated"):
            await library.arun("bash", {"command": "true"})
    executor.execute.side_effect = RuntimeError("backend failed after an effect")
    with execution_context(scope=scope):
        with pytest.raises(RuntimeError, match="backend failed"):
            await library.arun("bash", {"command": "true"})
    executor.execute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        {"command": " "},
        {"command": "bad\0command"},
    ],
)
async def test_bash_invalid_requests_never_dispatch(workspace_tools, args):
    library, scope, executor = workspace_tools
    with execution_context(scope=scope):
        with pytest.raises(ValueError):
            await library.arun("bash", args)
    executor.execute.assert_not_called()


def test_workspace_public_annotations_and_ui_labels():
    library = ToolLibrary("workspace", [ReadFileTool(), BashTool()])
    for name, label, parameters in (
        ("read", "Read", {"path", "offset", "limit"}),
        ("bash", "Bash", {"command", "timeout_ms"}),
    ):
        definition = library.get_tool_definition(name)
        assert definition.display_name == label
        assert set(definition.annotations) == parameters | {"return"}
        assert set(definition.input_schema["properties"]) == parameters
        assert all(
            definition.input_schema["properties"][key].get("description")
            for key in parameters
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_reader_line_windows_and_configured_cwd(asynchronous):
    content = "first\r\n ação 🐍\r\nlast".encode()
    fs = InMemoryWorkspace("lines", {"/repo/file.txt": content})
    scope = ExecutionScope(
        environment=ExecutionEnvironment(fs),
        permissions=PermissionSet(
            resources=[fs.permission("/repo/file.txt", "filesystem.read")]
        ),
    )
    library = ToolLibrary("reader", [ReadFileTool(cwd="/repo")])
    with execution_context(scope=scope):
        for arguments, expected in (
            ({"path": "file.txt"}, content.decode()),
            ({"path": "file.txt", "offset": 2, "limit": 1}, " ação 🐍\r\n"),
            ({"path": "/repo/file.txt", "offset": 3, "limit": 10}, "last"),
        ):
            result = (
                await library.arun("read", arguments)
                if asynchronous
                else library.run("read", arguments)
            )
            assert result == expected
        with pytest.raises(ValueError, match="offset"):
            library.run("read", {"path": "file.txt", "offset": 4})


@pytest.mark.parametrize(
    "arguments",
    [
        {"offset": 0},
        {"offset": True},
        {"offset": -1},
        {"limit": 0},
        {"limit": False},
        {"limit": 1.5},
    ],
)
def test_read_invalid_paging_before_io(arguments):
    filesystem = Mock()
    with pytest.raises(ValueError):
        ReadFileTool()("/file", filesystem=filesystem, **arguments)
    filesystem.read_lines.assert_not_called()
    filesystem.read_bytes.assert_not_called()


def test_read_selected_window_not_whole_file_size_or_encoding():
    fs = InMemoryWorkspace(
        "large", {"/large": b"ok\n" + b"\xff" * 2_000_000, "/many": b"x\n" * 3000}
    )
    scope = ExecutionScope(
        environment=ExecutionEnvironment(fs),
        permissions=PermissionSet(
            resources=[
                fs.permission(path, "filesystem.read") for path in ("/large", "/many")
            ]
        ),
    )
    with execution_context(scope=scope):
        assert ReadFileTool()("/large", limit=1, filesystem=fs) == "ok\n"
        assert ReadFileTool()("/many", filesystem=fs) == "x\n" * 2000
        with pytest.raises(ValueError, match="byte limit"):
            ReadFileTool()("/large", offset=2, limit=1, filesystem=fs)


def test_image_paging_rejected_before_publication():
    fs, handle = Mock(), Mock()
    with pytest.raises(ValueError, match="text files"):
        ReadFileTool(supports_vision=True)(
            "/image.png", limit=1, filesystem=fs, handle=handle
        )
    fs.read_bytes.assert_not_called()
    handle.get_notification.assert_not_called()


@pytest.mark.parametrize("tool_class", [ReadFileTool, BashTool])
def test_cwd_is_virtual_and_constructor_owned(tool_class):
    for cwd in ("relative", "/../host", "//host", "/a\\b"):
        with pytest.raises(ValueError):
            tool_class(cwd=cwd)


@pytest.mark.asyncio
async def test_bash_configured_cwd_cannot_be_overridden_by_model(workspace_tools):
    _, scope, executor = workspace_tools
    library = ToolLibrary("bash", [BashTool(cwd="/project")])
    with execution_context(scope=scope):
        with pytest.raises(TypeError, match="cwd"):
            await library.arun("bash", {"command": "true", "cwd": "/elsewhere"})
        executor.execute.assert_not_called()
        await library.arun("bash", {"command": "true"})
    assert executor.execute.call_args.args[0].cwd == "/project"


@pytest.mark.asyncio
async def test_bash_output_budget_is_not_a_model_argument(workspace_tools):
    library, scope, executor = workspace_tools
    with execution_context(scope=scope):
        with pytest.raises(TypeError, match="max_output_bytes"):
            await library.arun("bash", {"command": "true", "max_output_bytes": 10})
    executor.execute.assert_not_called()
