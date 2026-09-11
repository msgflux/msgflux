from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_transport import render_native_output, transport_adapter
from msgflux.nn import Agent, ToolLibrary
from msgflux.runtime import (
    AgentApprovals,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspace,
    PermissionSet,
    SQLiteApprovalStore,
    WorkspaceEditor,
    execution_context,
)
from msgflux.runtime.approvals.reconciliation import reconcile_batch
from msgflux.tools.builtin import ApplyPatchTool, BashTool
from msgflux.tools.definitions import ToolCatalog, ToolSpec
from msgflux.tools.patch import apply_diff
from msgflux.tools.runtime import ToolOutcome


@pytest.mark.parametrize(
    "before,diff,mode,after",
    [
        ("", "+first\n+second", "create", "first\nsecond"),
        ("", "", "create", ""),
        ("old\n", "@@\n-old\n+new", "default", "new\n"),
        ("old\r\n", "@@\n-old\n+new", "default", "new\r\n"),
        ("old", "@@\n-old\n+new", "default", "new"),
        (
            "first\nlast\n",
            "@@\n last\n+tail\n*** End of File",
            "default",
            "first\nlast\ntail\n",
        ),
        ("  old  \n", "@@\n-old\n+new", "default", "new\n"),
        (
            "class A:\n    def run():\n        old\n",
            "@@ class A:\n@@     def run():\n-        old\n+        new",
            "default",
            "class A:\n    def run():\n        new\n",
        ),
    ],
)
def test_v4a_parser(before, diff, mode, after):
    assert apply_diff(before, diff, mode=mode) == after


@pytest.mark.parametrize(
    "diff",
    [
        "@@\n-missing\n+new",
        "@@\n?bad",
        "*** Begin Patch\n*** End Patch",
        "@@\n-old\n+new\n*** End Patch\n+ignored",
        "@@\n-old\n+new\n*** Delete File: /other",
        "@@\n-old\n+new\n*** End of File\n+ignored",
    ],
)
def test_invalid_conflicting_and_multifile_diffs_rejected(diff):
    with pytest.raises(ValueError):
        apply_diff("old\n", diff)


def scope(fs):
    return ExecutionScope(
        namespace="patcher",
        thread_id="t",
        run_id="r",
        principal="user",
        environment=ExecutionEnvironment(fs),
        permissions=PermissionSet(
            resources=[
                fs.permission("/a", f"filesystem.{action}")
                for action in ("read", "write", "delete")
            ]
        ),
    )


def call(operation="update_file"):
    item = {
        "type": "apply_patch_call",
        "id": "apc_1",
        "call_id": "call_1",
        "status": "completed",
        "operation": {"type": operation, "path": "a"},
    }
    if operation != "delete_file":
        item["operation"]["diff"] = (
            "+new" if operation == "create_file" else "@@\n-old\n+new"
        )
    return item


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    result = OpenAIChatCompletion(model_id="gpt-5", api_mode="responses")
    yield result
    result.close()


def catalog(tool):
    definition = ToolLibrary("patch", [tool]).get_tool_definition(tool.name)
    return ToolCatalog(tools=[ToolSpec.from_definition(definition)])


def parse(model, tool, item):
    return model._process_responses_model_output(
        {"output": [deepcopy(item)], "status": "completed"},
        tool_routes=model._native_tool_routes(catalog(tool)),
    )


def test_native_selection_and_portable_schema(model):
    tool = ApplyPatchTool()
    tool.name = "modify"
    selected = catalog(tool)
    assert model._tools_to_responses(selected) == [{"type": "apply_patch"}]
    selected.choice = "modify"
    params = model._build_responses_generation_params("edit", None, None, selected)
    assert params["tool_choice"] == {"type": "apply_patch"}
    assert set(selected.tools[0].parameters["properties"]) == {
        "operation",
        "path",
        "diff",
    }
    model.native_tools = False
    assert model._tools_to_responses(selected)[0]["name"] == "modify"
    model.native_tools = True
    tool.tool_config["allow_background"] = True
    assert model._tools_to_responses(catalog(tool))[0]["type"] == "function"
    mixed = ToolCatalog(
        tools=[*catalog(ApplyPatchTool()).tools, *catalog(BashTool()).tools]
    )
    assert {item["type"] for item in model._tools_to_responses(mixed)} == {
        "apply_patch",
        "shell",
    }


@pytest.mark.parametrize("operation", ["create_file", "update_file", "delete_file"])
@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.asyncio
async def test_native_approval_preview_restart_and_execution(
    tmp_path, model, operation, approved
):
    fs = InMemoryWorkspace(
        "files", {} if operation == "create_file" else {"/a": b"old"}
    )
    cp_path, ap_path = str(tmp_path / "cp.db"), str(tmp_path / "ap.db")
    checkpoints, journal = SQLiteCheckpointStore(cp_path), SQLiteApprovalStore(ap_path)
    tool = ApplyPatchTool()
    tool.name = "modify"

    def build():
        return Agent(
            name="patcher",
            model=model,
            tools=[tool],
            checkpoint_store=checkpoints,
            approvals=AgentApprovals(journal, {"modify": "v1"}, "p1"),
        )

    agent = build()
    agent.generator.aforward = AsyncMock(
        return_value=parse(model, tool, call(operation))
    )
    with pytest.raises(TaskPauseRequestedError):
        await agent.acall("patch", scope=scope(fs))
    record = journal.pending("patcher", "t", "r")[0]
    preview = agent.inspect_approval_preview("t", "r", record.request_id)
    assert preview.operation == operation.removesuffix("_file")
    assert preview.path == "/a"
    assert (
        "/dev/null" in preview.diff
        if operation != "update_file"
        else "-old" in preview.diff
    )
    checkpoints.close()
    journal.close()
    checkpoints, journal = SQLiteCheckpointStore(cp_path), SQLiteApprovalStore(ap_path)
    agent = build()
    assert agent.inspect_approval_preview("t", "r", record.request_id) == preview
    agent.decide_approval(record.request_id, approved=approved, decided_by="host")
    final = ModelResponse()
    final.set_response_type("text_generation")
    final.add("done")
    agent.generator.aforward = AsyncMock(return_value=final)
    assert await agent.acall("resume", scope=scope(fs)) == "done"
    with execution_context(scope=scope(fs)):
        if (approved and operation == "delete_file") or (
            not approved and operation == "create_file"
        ):
            with pytest.raises(FileNotFoundError):
                fs.read_text("/a")
        else:
            assert fs.read_text("/a") == ("new" if approved else "old")
    items = checkpoints.load_state("patcher", "t", "r")["messages"]["items"]
    output = next(
        item for item in items if item.get("type") == "apply_patch_call_output"
    )
    assert output["status"] == ("completed" if approved else "failed")
    assert output["call_id"] == "call_1"
    assert journal.get("patcher", record.request_id).status == (
        "consumed" if approved else "denied"
    )
    checkpoints.close()
    journal.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_stream_only_dispatches_complete_patch_once(model, asynchronous):
    events = [
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "apply_patch_call", "call_id": "call_1"},
        },
        {"type": "response.apply_patch_call_operation_diff.delta", "delta": "@@\n-old"},
        {"type": "response.output_item.done", "output_index": 0, "item": call()},
    ]
    events.append(deepcopy(events[-1]))
    stream = ModelStreamResponse()
    routes = model._native_tool_routes(catalog(ApplyPatchTool()))
    if asynchronous:

        async def source():
            for event in events:
                yield event

        model._aexecute_model = AsyncMock(return_value=source())
        await model._astream_responses_generate(
            stream_response=stream, _tool_routes=routes
        )
    else:
        model._execute_model = Mock(return_value=iter(events))
        model._stream_responses_generate(stream_response=stream, _tool_routes=routes)
    assert len(stream.data.get_intents()) == 1
    assert stream.data.get_intents()[0].arguments["diff"] == "@@\n-old\n+new"


def test_history_interruption_projection_and_unknown_codec(model):
    tool = ApplyPatchTool()
    tool.name = "modify"
    response = parse(model, tool, call())
    intent = response.data.get_intents()[0]
    output = response.data.render_outcomes(
        [ToolOutcome.completed(intent, {"status": "completed"})]
    )[0]
    history = ChatMessages([*response.history_items, output])
    assert history.to_responses_input()[0] == call()
    assert history.to_responses_input(native_tools=False)[0]["name"] == "modify"
    assert history.to_chatml()[0]["tool_calls"][0]["function"]["name"] == "modify"
    interrupted = ChatMessages(response.history_items)
    assert interrupted.close_interrupted_tool_calls() == 1
    assert interrupted.to_responses_input()[-1]["status"] == "failed"
    metadata = response.data.native_calls["call_1"]
    with pytest.raises(ValueError):
        transport_adapter({**metadata, "version": 999})
    assert (
        render_native_output("call_1", None, metadata, error="denied")["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    "operation",
    [
        {"type": "move_file", "path": "a"},
        {"type": "delete_file", "path": "a", "diff": "bad"},
        {"type": "update_file", "path": "a"},
    ],
)
def test_malformed_native_operations_fail_closed(model, operation):
    with pytest.raises(ValueError):
        parse(model, ApplyPatchTool(), {**call(), "operation": operation})


def test_incomplete_or_unbound_native_call_is_not_dispatched(model):
    with pytest.raises(ValueError, match="complete"):
        parse(model, ApplyPatchTool(), {**call(), "status": "in_progress"})
    with pytest.raises(ValueError, match="Unbound"):
        model._process_responses_model_output({"output": [call()]})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_public_model_transport_and_renamed_patch(model, asynchronous, streaming):
    tool = ApplyPatchTool()
    tool.name = "modify"
    event = {"type": "response.output_item.done", "output_index": 0, "item": call()}

    async def source():
        yield event

    if asynchronous:
        model._aexecute_model = AsyncMock(
            return_value=source() if streaming else {"output": [call()]}
        )
        result = await model.acall("edit", tool_catalog=catalog(tool), stream=streaming)
        executor = model._aexecute_model
    else:
        model._execute_model = Mock(
            return_value=iter([event]) if streaming else {"output": [call()]}
        )
        result = model("edit", tool_catalog=catalog(tool), stream=streaming)
        executor = model._execute_model
    if streaming:
        async for _ in result.consume():
            pass
    assert result.data.get_intents()[0].name == "modify"
    assert executor.call_args.kwargs["tools"] == [{"type": "apply_patch"}]
    assert "_tool_routes" not in executor.call_args.kwargs
    assert "tool_catalog" not in executor.call_args.kwargs


@pytest.mark.asyncio
async def test_tool_conflicts_paths_and_shared_transform():
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    library = ToolLibrary("patch", [ApplyPatchTool])
    with execution_context(scope=scope(fs)):
        with pytest.raises(FileExistsError):
            await library.arun(
                "apply_patch", {"operation": "create", "path": "a", "diff": "+new"}
            )
        with pytest.raises(ValueError):
            await library.arun(
                "apply_patch",
                {"operation": "update", "path": "../a", "diff": "@@\n-old\n+new"},
            )
        assert fs.read_text("/a") == "old"
        editor = WorkspaceEditor(fs, require_approval=False)
        with pytest.raises(TypeError, match="return text"):
            editor.prepare_transform("/a", lambda _text: None)
        await editor.aapply(
            await editor.aprepare_transform("/a", lambda text: text.upper())
        )
        assert fs.read_text("/a") == "OLD"
        await editor.aapply(await editor.aprepare_delete("/a"))
        await editor.aapply(await editor.aprepare_create("/a", ""))
        assert fs.read_text("/a") == ""


def test_reconciliation_renders_patch_output(model):
    response = parse(model, ApplyPatchTool(), call())
    store = InMemoryCheckpointStore()
    messages = ChatMessages()
    messages.begin_turn(turn_id="r")
    messages.extend(response.history_items)
    intent = response.data.get_intents()[0]
    store.save_state(
        "patcher",
        "t",
        "r",
        {
            "status": "paused",
            "messages": messages._to_state(),
            "runtime": {
                "extensions": {
                    "pending_approvals": {
                        "schema_version": 1,
                        "phase": "executing",
                        "api_mode": "responses",
                        "native_calls": response.data.native_calls,
                        "intents": [
                            {
                                "id": intent.id,
                                "name": intent.name,
                                "arguments": dict(intent.arguments),
                            }
                        ],
                    }
                }
            },
        },
    )
    state = store.load_state("patcher", "t", "r")
    reconcile_batch(
        store,
        "patcher",
        "t",
        "r",
        expected_revision=state.get("_checkpoint", {}).get("revision", 0),
        decision_id="reconcile",
        decided_by="host",
        reason="verified file",
        worker_stopped=True,
        results={"call_1": '{"status":"completed"}'},
    )
    items = store.load_state("patcher", "t", "r")["messages"]["items"]
    assert any(
        item.get("type") == "apply_patch_call_output" and item["status"] == "completed"
        for item in items
    )
