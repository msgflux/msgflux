import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.nn import Agent, ToolLibrary
from msgflux.runtime import (
    AgentApprovals,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryApprovalStore,
    InMemoryWorkspace,
    PermissionSet,
    ProcessExecutor,
    ProcessResult,
    SandboxCapabilities,
    execution_context,
)
from msgflux.tools.builtin import BashTool, ReadFileTool
from msgflux.tools.definitions import ToolCatalog, ToolSpec
from msgflux.tools.runtime import ToolOutcome
from msgflux.models.tool_adapters.openai_shell import shell_output
from msgflux.tools.shell import ShellResult, ShellCommandResult
from msgflux.models.tool_transport import render_native_output


CALL = {
    "type": "shell_call",
    "id": "sh_1",
    "call_id": "call_1",
    "status": "completed",
    "action": {
        "commands": ["echo first", "echo second"],
        "timeout_ms": 1000,
        "max_output_length": 100,
    },
}


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    return OpenAIChatCompletion(model_id="gpt-5", api_mode="responses")


def catalog(tool):
    definition = ToolLibrary("shell", [tool]).get_tool_definition("bash")
    return ToolCatalog(tools=[ToolSpec.from_definition(definition)])


def test_native_declaration_and_explicit_function_fallback(model):
    assert model._tools_to_responses(catalog(BashTool())) == [
        {"type": "shell", "environment": {"type": "local"}}
    ]
    model.native_tools = False
    assert model._tools_to_responses(catalog(BashTool()))[0]["type"] == "function"
    model.native_tools = True
    library = ToolLibrary("shell", [BashTool])
    assert library.get_tool_definition("bash").kind == "shell"
    assert not library.get_tool_definition("bash").native_bindings
    selected = catalog(BashTool())
    selected.choice = "bash"
    params = model._build_responses_generation_params("run", None, None, selected)
    assert params["tool_choice"] == {"type": "shell"}
    selected.tools[0].defer_loading = True
    with pytest.raises(ValueError, match="deferred"):
        model._tools_to_responses(selected)


def test_background_shell_preserves_selector_and_single_deadline(model):
    tool = BashTool()
    tool.tool_config["allow_background"] = True
    selected = catalog(tool)
    properties = selected.tools[0].parameters["properties"]
    assert "run_in_background" in properties
    assert "timeout_ms" in properties
    assert "timeout" not in properties
    assert "cwd" not in properties
    assert model._tools_to_responses(selected)[0]["type"] == "function"
    assert not model._native_tool_routes(selected)


@pytest.mark.parametrize(
    "action", [{}, {"commands": "echo x"}, {"commands": ["x"], "timeout_ms": -1}]
)
def test_invalid_native_actions_rejected(model, action):
    with pytest.raises(ValueError):
        model._process_responses_model_output(
            {"output": [{**CALL, "action": action}]},
            tool_routes=model._native_tool_routes(catalog(BashTool())),
        )


def test_native_results_validate_complete_batch():
    with pytest.raises(ValueError, match="every command"):
        shell_output(
            "c",
            ShellResult(results=(ShellCommandResult(status="exited", returncode=0),)),
            command_count=2,
        )
    blocked = shell_output("c", None, error="Permission denied", command_count=2)
    assert len(blocked["output"]) == 2
    assert all(part["outcome"]["exit_code"] != 0 for part in blocked["output"])


def test_native_interruption_and_portable_projection():
    history = ChatMessages([CALL])
    assert history.close_interrupted_tool_calls(reason="cancelled") == 1
    assert history.close_interrupted_tool_calls(reason="cancelled") == 0
    native = history.to_responses_input(provider="openai")
    assert native[-1]["type"] == "shell_call_output"
    assert len(native[-1]["output"]) == 2
    assert all(part["stderr"] == "cancelled" for part in native[-1]["output"])
    portable = history.to_chatml()
    call = portable[0]["tool_calls"][0]
    assert call["function"]["name"] == "bash_tool"
    assert (
        json.loads(call["function"]["arguments"])["command"]
        == CALL["action"]["commands"]
    )
    assert portable[1]["role"] == "tool"


def test_reader_constructor_config_is_compiled():
    class VisualReader(ReadFileTool):
        """Read configured files."""

        tool_config = {**ReadFileTool.tool_config, "usage_guidance": "Existing."}

        def __init__(self):
            super().__init__(supports_vision=True)

    definition = ToolLibrary("files", [VisualReader]).get_tool_definition("read")
    assert definition.usage_guidance.startswith("Existing.\n\n")
    assert VisualReader.tool_config["usage_guidance"] == "Existing."
    assert ReadFileTool.tool_config["usage_guidance"] is None


def test_native_parse_render_and_checkpoint_projection(model):
    response = model._process_responses_model_output(
        {"output": [CALL], "status": "completed"},
        tool_routes=model._native_tool_routes(catalog(BashTool())),
    )
    calls = response.data
    intent = calls.get_intents()[0]
    assert intent.name == "bash"
    assert "max_output_bytes" not in intent.arguments
    assert calls.native_calls[intent.id]["max_output_length"] == 100
    assert intent.arguments["command"] == CALL["action"]["commands"]
    result = ShellResult(
        results=(
            ShellCommandResult(status="exited", returncode=0, stdout="first"),
            ShellCommandResult(status="timed_out"),
        )
    )
    rendered = calls.render_outcomes([ToolOutcome.completed(intent, result)])
    assert rendered[0]["type"] == "shell_call_output"
    assert rendered[0]["max_output_length"] == 100
    history = ChatMessages([*response.history_items, *rendered])
    restored = ChatMessages()
    restored._hydrate_state(history._to_state())
    assert restored.to_responses_input(provider="openai") == [CALL, rendered[0]]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_stream_complete_shell_only_once(model, asynchronous):
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "shell_call", "call_id": "call_1"},
        },
        {"type": "response.output_item.done", "output_index": 0, "item": CALL},
    ]
    events.append(events[-1])
    stream = ModelStreamResponse()
    if asynchronous:

        async def source():
            for event in events:
                yield event

        model._aexecute_model = AsyncMock(return_value=source())
        await model._astream_responses_generate(
            stream_response=stream,
            _tool_routes=model._native_tool_routes(catalog(BashTool())),
        )
    else:
        model._execute_model = Mock(return_value=iter(events))
        model._stream_responses_generate(
            stream_response=stream,
            _tool_routes=model._native_tool_routes(catalog(BashTool())),
        )
    assert len(stream.data.get_intents()) == 1
    assert ChatMessages(stream.chat_accumulator.snapshot()).to_responses_input() == [
        CALL
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["bash", "terminal"])
@pytest.mark.parametrize("approval_mode", ["required", "none", "other_tools"])
async def test_native_approval_resume_uses_same_executor(model, name, approval_mode):
    executor = Mock(spec=ProcessExecutor)
    executor.capabilities = SandboxCapabilities(
        {"filesystem", "network", "process", "resource_limits"}
    )
    executor.supports_workspace.return_value = True
    executor.execute.return_value = ProcessResult(0, b"ok")
    workspace = InMemoryWorkspace("shell")
    scope = ExecutionScope(
        namespace="shell",
        principal="user",
        thread_id="t",
        run_id="r",
        environment=ExecutionEnvironment(workspace, executor),
        permissions=PermissionSet(["process.execute"]),
    )
    store, journal = InMemoryCheckpointStore(), InMemoryApprovalStore()
    tool = BashTool()
    tool.name = name
    agent = Agent(
        name="shell",
        model=model,
        tools=[tool, ReadFileTool()] if approval_mode == "other_tools" else [tool],
        checkpoint_store=store,
        approvals=None
        if approval_mode == "none"
        else AgentApprovals(
            journal, {name if approval_mode == "required" else "read": "v1"}, "p1"
        ),
    )
    first = model._process_responses_model_output(
        {"output": [CALL], "status": "completed"},
        tool_routes=model._native_tool_routes(renamed_catalog(name)),
    )
    final = ModelResponse()
    final.set_response_type("text_generation")
    final.add("done")
    agent.generator.aforward = AsyncMock(side_effect=[first, final])
    if approval_mode != "required":
        assert await agent.acall("run", scope=scope) == "done"
        assert executor.execute.await_count == 2
        assert not journal.pending("shell", "t", "r")
        return
    with pytest.raises(TaskPauseRequestedError):
        await agent.acall("run", scope=scope)
    executor.execute.assert_not_called()
    checkpoint = store.load_state("shell", "t", "r")
    transport = checkpoint["runtime"]["extensions"]["pending_approvals"][
        "native_calls"
    ]["call_1"]
    assert transport["name"] == name
    assert transport["version"] == 1
    # A later model configuration does not rewrite the pending continuation.
    model.native_tools = False
    pending = journal.pending("shell", "t", "r")[0]
    agent.decide_approval(pending.request_id, approved=True, decided_by="host")
    assert await agent.acall("", scope=scope) == "done"
    assert executor.execute.await_count == 2
    history = store.load_state("shell", "t", "r")["messages"]["items"]
    assert sum(item.get("type") == "shell_call_output" for item in history) == 1
    assert not any(item.get("type") == "function_call_output" for item in history)


def test_native_reconciliation_preserves_format():
    from msgflux.runtime.approvals.reconciliation import reconcile_batch

    store = InMemoryCheckpointStore()
    messages = ChatMessages()
    messages.begin_turn(turn_id="r")
    messages.append(CALL)
    state = {
        "status": "paused",
        "messages": messages._to_state(),
        "runtime": {
            "extensions": {
                "pending_approvals": {
                    "schema_version": 1,
                    "phase": "executing",
                    "api_mode": "responses",
                    "intents": [
                        {
                            "id": "call_1",
                            "name": "bash",
                            "arguments": {"command": ["echo first", "echo second"]},
                        }
                    ],
                    "native_calls": {
                        "call_1": {
                            "codec": "openai.responses.shell",
                            "version": 1,
                            "name": "bash",
                            "command_count": 2,
                            "max_output_length": 100,
                        }
                    },
                }
            }
        },
    }
    store.save_state("shell", "t", "r", state)
    saved = store.load_state("shell", "t", "r")
    result = {
        "results": [
            {
                "stdout": "confirmed",
                "stderr": "",
                "status": "exited",
                "returncode": 0,
            }
        ]
        * 2
    }
    reconcile_batch(
        store,
        "shell",
        "t",
        "r",
        expected_revision=saved.get("_checkpoint", {}).get("revision", 0),
        decision_id="repair",
        decided_by="host",
        reason="checked results",
        worker_stopped=True,
        results={"call_1": json.dumps(result)},
    )
    history = store.load_state("shell", "t", "r")["messages"]["items"]
    output = next(item for item in history if item.get("type") == "shell_call_output")
    assert len(output["output"]) == 2
    assert output["output"][0]["stdout"] == "confirmed"
    assert output["max_output_length"] == 100


@pytest.mark.asyncio
async def test_batch_limits_and_timeout_do_not_escape_environment():
    executor = Mock(spec=ProcessExecutor)
    executor.capabilities = SandboxCapabilities(
        {"filesystem", "network", "process", "resource_limits"}
    )
    executor.supports_workspace.return_value = True
    executor.execute.side_effect = [
        asyncio.TimeoutError(),
        ProcessResult(3, b"x", b"err"),
    ]
    environment = ExecutionEnvironment(InMemoryWorkspace("shell"), executor)
    with execution_context(
        scope=ExecutionScope(
            environment=environment, permissions=PermissionSet(["process.execute"])
        )
    ):
        result = await BashTool().acall(
            ["first", "second"],
            timeout_ms=999999,
            environment=environment,
        )
    assert result.results[0].status == "timed_out"
    assert result.results[1].returncode == 3
    for call in executor.execute.call_args_list:
        assert call.args[0].timeout_seconds == 30
        assert call.args[0].max_output_bytes <= 1_000_000


def renamed_catalog(name):
    selected = catalog(BashTool())
    selected.tools[0].name = name
    return selected


def test_renamed_shell_projection_and_no_internal_metadata_on_wire(model):
    selected = renamed_catalog("terminal")
    selected.choice = "terminal"
    params = model._build_responses_generation_params("run", None, None, selected)
    assert params["tool_choice"] == {"type": "shell"}
    parsed = model._process_responses_model_output(
        {"output": [CALL]}, tool_routes=model._native_tool_routes(selected)
    )
    intent = parsed.data.get_intents()[0]
    assert intent.name == "terminal"
    assert intent.arguments == {
        "command": CALL["action"]["commands"],
        "timeout_ms": 1000,
    }
    history = ChatMessages(parsed.history_items)
    assert history.to_chatml()[0]["tool_calls"][0]["function"]["name"] == "terminal"
    assert history.to_responses_input() == [CALL]
    portable = history.to_responses_input(provider="other")
    assert portable[0]["type"] == "function_call"
    assert portable[0]["name"] == "terminal"
    model.native_tools = False
    request = model._build_responses_generation_params(history, None, None, selected)
    assert request["input"][0]["type"] == "function_call"
    assert request["tools"][0]["type"] == "function"
    assert history.close_interrupted_tool_calls(reason="stopped") == 1
    assert history.to_responses_input()[-1]["type"] == "shell_call_output"


def test_native_call_without_binding_fails_closed(model):
    with pytest.raises(ValueError, match="Unbound"):
        model._process_responses_model_output({"output": [CALL]})
    model.native_tools = False
    model._execute_model = Mock(return_value={"output": [CALL]})
    with pytest.raises(ValueError, match="Unbound"):
        model("run", tool_catalog=catalog(BashTool()))


def test_multiple_native_shells_rejected(model):
    selected = renamed_catalog("first")
    selected.tools.extend(renamed_catalog("second").tools)
    with pytest.raises(ValueError, match="Only one"):
        model._tools_to_responses(selected)


@pytest.mark.asyncio
async def test_concurrent_native_routes_are_request_local(model):
    entered = asyncio.Event()
    count = 0

    async def execute(**kwargs):
        nonlocal count
        assert "tool_catalog" not in kwargs
        assert "_tool_routes" not in kwargs
        count += 1
        if count == 2:
            entered.set()
        await asyncio.wait_for(entered.wait(), timeout=2)
        return {"output": [deepcopy(CALL)]}

    model._aexecute_model = execute
    first, second = await asyncio.gather(
        model.acall("run", tool_catalog=renamed_catalog("first")),
        model.acall("run", tool_catalog=renamed_catalog("second")),
    )
    assert first.data.get_intents()[0].name == "first"
    assert second.data.get_intents()[0].name == "second"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_public_stream_binds_renamed_tool(model, asynchronous):
    event = {"type": "response.output_item.done", "output_index": 0, "item": CALL}

    async def source():
        yield event

    if asynchronous:
        model._aexecute_model = AsyncMock(return_value=source())
        stream = await model.acall(
            "run", tool_catalog=renamed_catalog("terminal"), stream=True
        )
        executor = model._aexecute_model
    else:
        model._execute_model = Mock(return_value=iter([event]))
        stream = model("run", tool_catalog=renamed_catalog("terminal"), stream=True)
        executor = model._execute_model
    async for _ in stream.consume():
        pass
    assert stream.data.get_intents()[0].name == "terminal"
    assert "_tool_routes" not in executor.call_args.kwargs


@pytest.mark.parametrize(
    "change", [{"version": 99}, {"codec": "untrusted.module"}, {"command_count": 0}]
)
def test_unknown_transport_metadata_rejected(model, change):
    parsed = model._process_responses_model_output(
        {"output": [CALL]}, tool_routes=model._native_tool_routes(catalog(BashTool()))
    )
    metadata = {**parsed.data.native_calls["call_1"], **change}
    with pytest.raises(ValueError):
        render_native_output("call_1", None, metadata, error="denied")
    item = deepcopy(parsed.history_items[0])
    item["metadata"]["tool_transport"] = metadata
    with pytest.raises(ValueError):
        ChatMessages([item]).to_responses_input()


def test_canonical_shell_result_json_round_trip():
    import msgspec

    result = ShellResult(
        results=(ShellCommandResult(status="not_executed", stderr="budget exhausted"),)
    )
    assert msgspec.json.decode(msgspec.json.encode(result), type=ShellResult) == result
    with pytest.raises(ValueError):
        ShellCommandResult(status="timed_out", returncode=0)


def test_native_cache_preserves_request_route(model):
    from msgflux.models.cache import ResponseCache

    model.enable_cache = True
    model._response_cache = ResponseCache()
    model._execute_model = Mock(return_value={"output": [CALL]})
    for name in ("first", "second", "first", "second"):
        response = model("run", tool_catalog=renamed_catalog(name))
        assert response.data.get_intents()[0].name == name
    assert model._execute_model.call_count == 2
