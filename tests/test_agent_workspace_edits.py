from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent, ToolLibrary
from msgflux.nn.hooks import Hook
from msgflux.runtime import (
    AgentApprovals,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryApprovalStore,
    InMemoryWorkspace,
    PermissionSet,
    SQLiteApprovalStore,
    execution_context,
)
from msgflux.tools.builtin import DeleteTool, EditTool, WriteTool
from msgflux.utils.msgspec import msgspec_dumps


def scope(fs):
    return ExecutionScope(
        namespace="editor",
        thread_id="t",
        run_id="r",
        principal="user",
        environment=ExecutionEnvironment(fs),
        permissions=PermissionSet(
            resources=[
                fs.permission("/a", "filesystem.read"),
                fs.permission("/a", "filesystem.write"),
                fs.permission("/a", "filesystem.delete"),
            ]
        ),
    )


def response(name=None, arguments=None):
    result = ModelResponse()
    if name:
        calls = ToolCallAggregator()
        calls.process(0, "call", name, msgspec_dumps(arguments))
        result.set_response_type("tool_call")
        result.add(calls)
    else:
        result.set_response_type("text_generation")
        result.add("done")
    return result


def agent(checkpoints, journal):
    return Agent(
        name="editor",
        model=Mock(model_type="chat_completion"),
        tools=[WriteTool(), EditTool(), DeleteTool()],
        checkpoint_store=checkpoints,
        approvals=AgentApprovals(
            journal, {"write": "v1", "edit": "v1", "delete": "v1"}, "p1"
        ),
    )


@pytest.mark.parametrize(
    "name, args",
    [
        ("write", {"path": "a", "content": "new"}),
        ("edit", {"path": "a", "old": "old", "new": "new"}),
        ("delete", {"path": "a"}),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.asyncio
async def test_pause_preview_restart_and_single_consumption(
    tmp_path, name, args, asynchronous, persistent
):
    cp_path, ap_path = str(tmp_path / "cp.db"), str(tmp_path / "ap.db")
    checkpoints = (
        SQLiteCheckpointStore(cp_path) if persistent else InMemoryCheckpointStore()
    )
    journal = SQLiteApprovalStore(ap_path) if persistent else InMemoryApprovalStore()
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    current = agent(checkpoints, journal)
    current.generator.forward = Mock(return_value=response(name, args))
    current.generator.aforward = AsyncMock(return_value=response(name, args))

    async def invoke():
        if asynchronous:
            return await current.acall("change", scope=scope(fs))
        return current("change", scope=scope(fs))

    with pytest.raises(TaskPauseRequestedError):
        await invoke()
    with execution_context(scope=scope(fs)):
        assert fs.read_text("/a") == "old"
    record = journal.pending("editor", "t", "r")[0]
    preview = await current.ainspect_approval_preview("t", "r", record.request_id)
    assert preview.before == "old"
    assert preview.after == (None if name == "delete" else "new")
    assert "-old" in preview.diff
    assert ("+++ /dev/null" if name == "delete" else "+new") in preview.diff
    async with current.watch("t") as watcher:
        assert watcher.snapshot.approvals[0].request_id == record.request_id
    state = checkpoints.load_state("editor", "t", "r")
    assert "prepared_changes" in state["runtime"]["extensions"]["pending_approvals"]
    assert "prepared_change" not in str(state["messages"])
    if persistent:
        checkpoints.close()
        journal.close()
        checkpoints, journal = (
            SQLiteCheckpointStore(cp_path),
            SQLiteApprovalStore(ap_path),
        )
    current = agent(checkpoints, journal)
    assert current.inspect_approval_preview("t", "r", record.request_id) == preview
    current.decide_approval(record.request_id, approved=True, decided_by="host")
    current.generator.forward = Mock(return_value=response())
    current.generator.aforward = AsyncMock(return_value=response())
    assert await invoke() == "done"
    with execution_context(scope=scope(fs)):
        if name == "delete":
            with pytest.raises(FileNotFoundError):
                fs.read_text("/a")
        else:
            assert fs.read_text("/a") == "new"
    assert [event.status for event in journal.events("editor", record.request_id)] == [
        "pending",
        "approved",
        "consumed",
    ]
    history = checkpoints.load_state("editor", "t", "r")["messages"]["items"]
    output = next(
        item for item in history if item.get("type") == "function_call_output"
    )
    assert output["output"] == '{"status":"completed"}'
    if persistent:
        checkpoints.close()
        journal.close()


@pytest.mark.parametrize(
    "name, args",
    [
        ("write", {"path": "a", "content": "new"}),
        ("edit", {"path": "a", "old": "old", "new": "new"}),
    ],
)
def test_stale_file_never_reuses_approved_preview(name, args):
    checkpoints, journal = InMemoryCheckpointStore(), InMemoryApprovalStore()
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    current = agent(checkpoints, journal)
    current.generator.forward = Mock(return_value=response(name, args))
    with pytest.raises(TaskPauseRequestedError):
        current("change", scope=scope(fs))
    record = journal.pending("editor", "t", "r")[0]
    current.decide_approval(record.request_id, approved=True, decided_by="host")
    with execution_context(scope=scope(fs)):
        fs.write_text("/a", "old with concurrent work")
    with pytest.raises(TaskPauseRequestedError):
        current("resume", scope=scope(fs))
    with execution_context(scope=scope(fs)):
        assert fs.read_text("/a") == "old with concurrent work"
    assert current.inspect_approval_preview("t", "r", record.request_id).after == "new"
    assert journal.get("editor", record.request_id).status == "approved"


@pytest.mark.asyncio
async def test_full_access_public_schemas_and_live_denial():
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    library = ToolLibrary("files", [WriteTool(), EditTool()])
    for name, fields in (
        ("write", {"path", "content"}),
        ("edit", {"path", "old", "new"}),
    ):
        definition = library.get_tool_definition(name)
        assert set(definition.input_schema["properties"]) == fields
        assert definition.display_name == name.title()
        assert all(
            item.get("description")
            for item in definition.input_schema["properties"].values()
        )
    with execution_context(scope=scope(fs)):
        assert await library.arun("write", {"path": "a", "content": "new"}) == {
            "status": "completed"
        }
        assert library.run("edit", {"path": "a", "old": "new", "new": "last"}) == {
            "status": "completed"
        }
    with execution_context(scope=replace(scope(fs), permissions=PermissionSet())):
        with pytest.raises(PermissionError):
            await library.arun("write", {"path": "a", "content": "forbidden"})


@pytest.mark.parametrize("mode", ["deny", "arguments", "cwd", "permissions"])
def test_changed_execution_cannot_escape_review(mode):
    checkpoints, journal = InMemoryCheckpointStore(), InMemoryApprovalStore()
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    current = agent(checkpoints, journal)
    current.generator.forward = Mock(
        side_effect=[response("write", {"path": "a", "content": "new"}), response()]
    )
    with pytest.raises(TaskPauseRequestedError):
        current("change", scope=scope(fs))
    record = journal.pending("editor", "t", "r")[0]
    current.decide_approval(
        record.request_id, approved=mode != "deny", decided_by="host"
    )
    current_scope = scope(fs)
    if mode == "arguments":
        Hook(
            event="before_tool",
            handler=lambda event: replace(
                event, arguments={"path": "a", "content": "tampered"}
            ),
        ).register(current)
    elif mode == "cwd":
        current.tool_library.get_tool_definition("write").executor.impl.cwd = "/other"
    elif mode == "permissions":
        current_scope = replace(current_scope, permissions=PermissionSet())
    if mode in {"cwd", "permissions"}:
        with pytest.raises(TaskPauseRequestedError):
            current("resume", scope=current_scope)
    else:
        assert current("resume", scope=current_scope) == "done"
    with execution_context(scope=scope(fs)):
        assert fs.read_text("/a") == "old"
    assert journal.get("editor", record.request_id).status != "consumed"


def test_invalid_edit_returns_observation_without_approval_or_effect():
    checkpoints, journal = InMemoryCheckpointStore(), InMemoryApprovalStore()
    fs = InMemoryWorkspace("files", {"/a": b"old old"})
    current = agent(checkpoints, journal)
    current.generator.forward = Mock(
        side_effect=[
            response("edit", {"path": "a", "old": "old", "new": "new"}),
            response(),
        ]
    )
    assert current("edit", scope=scope(fs)) == "done"
    assert journal.pending("editor", "t", "r") == []
    history = checkpoints.load_state("editor", "t", "r")["messages"]["items"]
    assert "exactly once" in str(history)
    with execution_context(scope=scope(fs)):
        assert fs.read_text("/a") == "old old"


@pytest.mark.asyncio
async def test_stream_preview_is_host_only_and_runtime_none_is_full_access():
    checkpoints, journal = InMemoryCheckpointStore(), InMemoryApprovalStore()
    fs = InMemoryWorkspace("files", {"/a": b"secret original"})
    current = agent(checkpoints, journal)
    current.generator.aforward = AsyncMock(
        return_value=response("write", {"path": "a", "content": "new"})
    )
    events = []
    with pytest.raises(TaskPauseRequestedError):
        async for event in current.stream_events("change", scope=scope(fs)):
            events.append(event)
    required = [event for event in events if event.type == "tool.approval_required"]
    assert len(required) == 1
    assert "secret original" not in repr(required[0].data)
    preview = await current.ainspect_approval_preview(
        "t", "r", required[0].data["request_id"]
    )
    assert "secret original" in preview.diff
    # A separate run may explicitly disable prompts; pending runs cannot.
    current.generator.aforward = AsyncMock(
        side_effect=[response("write", {"path": "a", "content": "full"}), response()]
    )
    assert (
        await current.acall(
            "change",
            scope=replace(scope(fs), thread_id="full", run_id="full"),
            approvals=None,
        )
        == "done"
    )
    with execution_context(scope=scope(fs)):
        assert fs.read_text("/a") == "full"


def test_tampered_persisted_preview_is_not_presented():
    checkpoints, journal = InMemoryCheckpointStore(), InMemoryApprovalStore()
    fs = InMemoryWorkspace("files", {"/a": b"old"})
    current = agent(checkpoints, journal)
    current.generator.forward = Mock(
        return_value=response("write", {"path": "a", "content": "new"})
    )
    with pytest.raises(TaskPauseRequestedError):
        current("change", scope=scope(fs))
    record = journal.pending("editor", "t", "r")[0]
    state = checkpoints.load_state("editor", "t", "r")
    state["runtime"]["extensions"]["pending_approvals"]["prepared_changes"]["call"][
        "change"
    ]["after"] = "forged"
    checkpoints.save_state("editor", "t", "r", state)
    with pytest.raises(ValueError, match="binding"):
        current.inspect_approval_preview("t", "r", record.request_id)


@pytest.mark.asyncio
async def test_parallel_calls_share_tool_but_not_prepared_context():
    checkpoints, journal = InMemoryCheckpointStore(), InMemoryApprovalStore()
    fs = InMemoryWorkspace("files", {"/a": b"old a", "/b": b"old b"})
    current = agent(checkpoints, journal)
    first = response("write", {"path": "a", "content": "new a"})
    first.data.process(1, "second", "write", '{"path":"b","content":"new b"}')
    current.generator.aforward = AsyncMock(side_effect=[first, response()])
    current_scope = replace(
        scope(fs),
        permissions=PermissionSet(
            resources=[
                fs.permission(path, action)
                for path in ("/a", "/b")
                for action in ("filesystem.read", "filesystem.write")
            ]
        ),
    )
    with pytest.raises(TaskPauseRequestedError):
        await current.acall("change both", scope=current_scope)
    records = journal.pending("editor", "t", "r")
    assert len(records) == 2
    for record in records:
        current.decide_approval(record.request_id, approved=True, decided_by="host")
    assert await current.acall("resume", scope=current_scope) == "done"
    with execution_context(scope=current_scope):
        assert fs.read_text("/a") == "new a"
        assert fs.read_text("/b") == "new b"
