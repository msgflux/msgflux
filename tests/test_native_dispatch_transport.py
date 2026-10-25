"""Local native protocols must preserve the result of the selected dispatch."""

import asyncio
import json
from functools import partial
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.nn import Agent, ToolLibrary
from msgflux.runtime import (
    AgentWorkspace,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    PermissionSet,
    ProcessExecutor,
    ProcessResult,
    SandboxCapabilities,
    execution_context,
)
from msgflux.tasks import InMemoryTaskStore
from msgflux.tools.builtin import ApplyPatchTool, BashTool
from msgflux.tools.definitions import ToolCatalog, ToolSpec
from tests.models._chat_transport import EndpointMockTransport
from tests.test_apply_patch import call as patch_call


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "in-process-test-key")
    client, async_client = Mock(), Mock()
    model = OpenAIChatCompletion(
        model_id="test-model",
        api_mode="responses",
        chat_transport=EndpointMockTransport(client, async_client),
        retry=False,
    )
    try:
        yield model, client, async_client
    finally:
        model.close()


def _tool(kind, mode):
    tool = BashTool() if kind == "shell" else ApplyPatchTool()
    if mode != "foreground":
        tool.tool_config[
            "allow_background" if mode == "optional_background" else mode
        ] = True
    return tool


@pytest.mark.parametrize("kind", ["shell", "apply_patch"])
@pytest.mark.parametrize(
    "mode", ["foreground", "background", "detached", "optional_background"]
)
@pytest.mark.parametrize("form", ["spec", "view"])
def test_dispatch_compatible_declaration_and_named_choice(provider, kind, mode, form):
    model, _, _ = provider
    tool = _tool(kind, mode)
    library = ToolLibrary("catalog", [tool])
    if form == "spec":
        selected = ToolCatalog(
            tools=[ToolSpec.from_definition(library.get_tool_definition(tool.name))],
            choice=tool.name,
        )
    else:
        selected = library.get_tool_catalog_view(thread_id="thread", choice=tool.name)
    declarations = model._tools_to_responses(selected)
    wire = next(
        item
        for item in declarations
        if item["type"] == kind or item.get("name") == tool.name
    )
    assert wire["type"] == (kind if mode == "foreground" else "function")
    params = model._build_responses_generation_params("run", None, None, selected)
    assert params["tool_choice"] == (
        {"type": kind}
        if mode == "foreground"
        else {"type": "function", "name": tool.name}
    )
    entries = model._catalog_tool_entries(selected)
    entry = next(item for item in entries if item.name == tool.name)
    assert entry.kind == kind
    # The frozen-view -> mutable-spec projection must retain the same behavior.
    if form == "view":
        converted = ToolCatalog.from_view(selected)
        assert model._tools_to_responses(converted) == declarations


def _reply(tool, kind, mode, requests, **params):
    requests.append(params)
    if len(requests) == 1:
        wire = next(
            item
            for item in params["tools"]
            if item["type"] == kind or item.get("name") == tool.name
        )
        if wire["type"] == "shell":
            item = {
                "type": "shell_call",
                "id": "shell1",
                "call_id": "call1",
                "status": "completed",
                "action": {"commands": ["simulated-command"]},
            }
        elif wire["type"] == "apply_patch":
            item = patch_call()
            item["call_id"] = "call1"
        else:
            arguments = (
                {"command": "simulated-command"}
                if kind == "shell"
                else {"operation": "update", "path": "a", "diff": "@@\n-old\n+new"}
            )
            if mode == "optional_background":
                arguments["run_in_background"] = True
            item = {
                "type": "function_call",
                "call_id": "call1",
                "name": tool.name,
                "arguments": json.dumps(arguments),
            }
        return {"id": "one", "status": "completed", "output": [item]}
    return {
        "id": "two",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["shell", "apply_patch"])
@pytest.mark.parametrize("mode", ["foreground", "background", "optional_background"])
async def test_agent_dispatch_completes_without_native_result_coercion(
    provider, kind, mode, asynchronous
):
    model, client, async_client = provider
    tool = _tool(kind, mode)
    filesystem = InMemoryWorkspace("native-dispatch", {"/a": b"old\n"})
    process = Mock(spec=ProcessExecutor)
    process.capabilities = SandboxCapabilities(
        {"filesystem", "network", "process", "resource_limits"}
    )
    process.supports_workspace.return_value = True
    process.execute.return_value = ProcessResult(0, b"simulated output")
    workspace = AgentWorkspace.from_environment(
        ExecutionEnvironment(filesystem, process),
        permissions=PermissionSet(
            ["process.execute"],
            resources=[
                filesystem.permission("/a", f"filesystem.{action}")
                for action in ("read", "write", "delete")
            ],
        ),
    )
    checkpoints, tasks = InMemoryCheckpointStore(), InMemoryTaskStore()
    agent = Agent(
        name="dispatch", model=model, tools=[tool], checkpoint_store=checkpoints
    )
    agent.tool_library.set_task_store(tasks)
    requests = []

    reply = partial(_reply, tool, kind, mode, requests)

    client.responses.create.side_effect = reply
    async_client.responses.create = AsyncMock(side_effect=reply)
    scope = ExecutionScope(
        thread_id="thread", run_id="run", principal="user", workspace=workspace
    )
    try:
        result = (
            await agent.acall("Run the simulated task", scope=scope)
            if asynchronous
            else agent("Run the simulated task", scope=scope)
        )
        assert result == "done"
        assert len(requests) >= 2
        dispatcher = agent.tool_library.get_background_dispatcher()
        for task in tasks.list():
            future = dispatcher.get_task_future(task.task_id)
            if future is not None:
                await asyncio.wait_for(asyncio.wrap_future(future), 2)
        assert len(tasks.list()) == (0 if mode == "foreground" else 1)
        assert all(task.status == "completed" for task in tasks.list())
        if kind == "shell":
            assert process.execute.call_count == 1
        else:
            with execution_context(scope=scope):
                assert filesystem.read_bytes("/a") == b"new\n"
            process.execute.assert_not_called()
        state = checkpoints.load_state("dispatch", "thread", "run")
        assert state["status"] == "completed"
        call_type = f"{kind}_call" if mode == "foreground" else "function_call"
        calls = [
            item for item in state["messages"]["items"] if item.get("type") == call_type
        ]
        outputs = [
            item
            for item in state["messages"]["items"]
            if item.get("type") == call_type + "_output"
        ]
        assert len(calls) == len(outputs) == 1
        assert calls[0]["call_id"] == outputs[0]["call_id"] == "call1"
        if mode != "foreground":
            assert "task_id=" in json.dumps(outputs[0]["output"])
    finally:
        dispatcher = agent.tool_library.get_background_dispatcher()
        for task in tasks.list():
            future = dispatcher.get_task_future(task.task_id)
            if future is not None:
                await asyncio.wait_for(asyncio.wrap_future(future), 2)
        await agent.aclose()
        await workspace.aclose()
