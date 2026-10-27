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
    AgentWorkspace,
    execution_context,
)
from msgflux.tools.builtin import DeleteTool, EditTool, TextEdit, WriteTool
from msgflux.utils.msgspec import msgspec_dumps


@pytest.mark.parametrize("mutate", [False, True])
def test_empty_directory_approval_survives_checkpoint_restart(tmp_path, mutate):
    cp_path, ap_path = (
        str(tmp_path / "directory-cp.db"),
        str(tmp_path / "directory-ap.db"),
    )
    checkpoints, journal = SQLiteCheckpointStore(cp_path), SQLiteApprovalStore(ap_path)
    fs = InMemoryWorkspace("directories")
    current_scope = scope(
        fs,
        permissions=PermissionSet(
            resources=[
                fs.permission(path, f"filesystem.{action}")
                for path in ("/", "/a", "/a/new")
                for action in ("list", "mkdir", "delete", "write", "read")
            ]
        ),
    )
    with execution_context(scope=current_scope):
        fs.mkdir("/a")
    current = agent(checkpoints, journal)
    current.generator.forward = Mock(return_value=response("delete", {"path": "/a"}))
    with pytest.raises(TaskPauseRequestedError):
        current("remove empty directory", scope=current_scope)
    request = journal.pending("editor", "t", "r")[0]
    preview = current.inspect_approval_preview("t", "r", request.request_id)
    assert preview.target_kind == "empty_directory"
    assert preview.directory_token and "/a/" in preview.diff
    checkpoints.close()
    journal.close()
    checkpoints, journal = SQLiteCheckpointStore(cp_path), SQLiteApprovalStore(ap_path)
    try:
        current = agent(checkpoints, journal)
        assert current.inspect_approval_preview("t", "r", request.request_id) == preview
        current.decide_approval(request.request_id, approved=True, decided_by="host")
        if mutate:
            with execution_context(scope=current_scope):
                fs.write_text("/a/new", "keep")
        current.generator.forward = Mock(return_value=response())
        if mutate:
            with pytest.raises(TaskPauseRequestedError):
                current("continue", scope=current_scope)
            with execution_context(scope=current_scope):
                assert fs.read_text("/a/new") == "keep"
        else:
            assert current("continue", scope=current_scope) == "done"
            with execution_context(scope=current_scope):
                assert fs.listdir("/") == ()
    finally:
        checkpoints.close()
        journal.close()


def scope(fs, *, permissions=None):
    return ExecutionScope(
        namespace="editor",
        thread_id="t",
        run_id="r",
        principal="user",
        workspace=AgentWorkspace.from_environment(
            ExecutionEnvironment(fs),
            permissions=permissions
            if permissions is not None
            else PermissionSet(
                resources=[
                    fs.permission("/a", "filesystem.read"),
                    fs.permission("/a", "filesystem.write"),
                    fs.permission("/a", "filesystem.delete"),
                ]
            ),
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
    "name, args, expected",
    [
        ("write", {"path": "a", "content": "new"}, "new"),
        ("edit", {"path": "a", "old": "old", "new": "new"}, "new"),
        (
            "edit",
            {
                "path": "a",
                "old": None,
                "new": None,
                "replace_all": None,
                "edits": [
                    {"old": "old", "new": "middle", "replace_all": True},
                    {"old": "middle", "new": "new"},
                ],
            },
            "new",
        ),
        ("delete", {"path": "a"}, None),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.asyncio
async def test_pause_preview_restart_and_single_consumption(
    tmp_path, name, args, expected, asynchronous, persistent
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
    records = journal.pending("editor", "t", "r")
    assert len(records) == 1
    record = records[0]
    preview = await current.ainspect_approval_preview("t", "r", record.request_id)
    assert preview.before == "old"
    assert preview.after == expected
    assert "-old" in preview.diff
    assert ("+++ /dev/null" if name == "delete" else f"+{expected}") in preview.diff
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
            assert fs.read_text("/a") == expected
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
        (
            "edit",
            {
                "path": "a",
                "edits": [
                    {"old": "old", "new": "middle"},
                    {"old": "middle", "new": "new"},
                ],
            },
        ),
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
        ("edit", {"path", "old", "new", "replace_all", "edits"}),
    ):
        definition = library.get_tool_definition(name)
        assert set(definition.input_schema["properties"]) == fields
        assert definition.display_name == name.title()
        assert all(
            item.get("description")
            for item in definition.input_schema["properties"].values()
        )
    edit_schema = library.get_tool_definition("edit").input_schema["properties"][
        "edits"
    ]["anyOf"][0]
    assert edit_schema["type"] == "array"
    assert edit_schema["items"]["required"] == ["new", "old", "replace_all"]
    assert set(edit_schema["items"]["properties"]) == {
        "new",
        "old",
        "replace_all",
    }
    assert edit_schema["items"]["properties"]["replace_all"] == {
        "type": "boolean",
        "default": False,
    }
    assert edit_schema["items"]["additionalProperties"] is False
    with execution_context(scope=scope(fs)):
        assert await library.arun("write", {"path": "a", "content": "new"}) == {
            "status": "completed"
        }
        assert library.run("edit", {"path": "a", "old": "new", "new": "middle"}) == {
            "status": "completed"
        }
        assert await library.arun(
            "edit",
            {
                "path": "a",
                "old": None,
                "new": None,
                "replace_all": None,
                "edits": [
                    {"old": "middle", "new": "last"},
                    {
                        "old": "last",
                        "new": "final",
                        "replace_all": True,
                    },
                ],
            },
        ) == {"status": "completed"}
        assert EditTool()(
            "a",
            edits=[TextEdit(old="final", new="done", replace_all=False)],
        ) == {"status": "completed"}
        assert fs.read_text("/a") == "done"
    with execution_context(scope=scope(fs, permissions=PermissionSet())):
        with pytest.raises(PermissionError):
            await library.arun("write", {"path": "a", "content": "forbidden"})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_edit_tool_mixed_replacements_chain_on_evolving_text(asynchronous):
    fs = InMemoryWorkspace("files", {"/a": b"one one\ntwo\nthree three\nremove-this\n"})
    library = ToolLibrary("files", [EditTool()])
    arguments = {
        "path": "a",
        "edits": [
            {"old": "one", "new": "1", "replace_all": True},
            {"old": "two", "new": "2"},
            {"old": "1 1\n2", "new": "first"},
            {"old": "three", "new": "3", "replace_all": True},
            {"old": "remove-this\n", "new": ""},
        ],
    }
    with execution_context(scope=scope(fs)):
        result = (
            await library.arun("edit", arguments)
            if asynchronous
            else library.run("edit", arguments)
        )
        assert result == {"status": "completed"}
        assert fs.read_text("/a") == "first\n3 3\n"


@pytest.mark.parametrize(
    "arguments, error",
    [
        (
            {
                "path": "a",
                "edits": [
                    {"old": "one", "new": "1"},
                    {"old": "missing", "new": "x", "replace_all": True},
                ],
            },
            "at least once",
        ),
        (
            {
                "path": "a",
                "edits": [
                    {"old": "one", "new": "1"},
                    {"old": "1", "new": "x x"},
                    {"old": "x", "new": "y"},
                ],
            },
            "exactly once",
        ),
    ],
)
def test_edit_batch_failure_leaves_the_snapshot_unchanged(arguments, error):
    fs = InMemoryWorkspace("files", {"/a": b"one two"})
    current_scope = scope(fs)
    with execution_context(scope=current_scope):
        with pytest.raises(ValueError, match=error):
            EditTool().prepare_workspace_change(arguments, current_scope.workspace)
        assert fs.read_text("/a") == "one two"


@pytest.mark.parametrize(
    "arguments",
    [
        {"path": "a"},
        {"path": "a", "old": "one"},
        {"path": "a", "new": "1"},
        {"path": "a", "old": None, "new": None},
        {
            "path": "a",
            "old": "one",
            "new": "1",
            "edits": [{"old": "one", "new": "1"}],
        },
        {"path": "a", "edits": []},
        {"path": "a", "edits": "one"},
        {"path": "a", "edits": [{"old": "", "new": "1"}]},
        {"path": "a", "edits": [{"old": "one"}]},
        {"path": "a", "edits": [{"old": "one", "new": "1", "extra": True}]},
        {"path": "a", "edits": [{"old": "one", "new": "1", "replace_all": 1}]},
        {"path": "a", "edits": [{"old": "one", "new": "1", "replace_all": "yes"}]},
        {"path": "a", "old": "one", "new": "1", "replace_all": "yes"},
        {"path": "a", "old": "one", "new": "1", "replace_all": 1},
        {
            "path": "a",
            "edits": [{"old": "one", "new": "1"}],
            "replace_all": True,
        },
    ],
)
def test_invalid_edit_arguments_fail_before_effects(arguments):
    fs = InMemoryWorkspace("files", {"/a": b"one two"})
    current_scope = scope(fs)
    with execution_context(scope=current_scope):
        with pytest.raises((TypeError, ValueError)):
            EditTool().prepare_workspace_change(arguments, current_scope.workspace)
        assert fs.read_text("/a") == "one two"


def test_text_edit_and_legacy_direct_calls_validate_flags_strictly():
    for arguments in (
        {"old": "", "new": "x"},
        {"old": 1, "new": "x"},
        {"old": "x", "new": 1},
        {"old": "x", "new": "y", "replace_all": 1},
        {"old": "x", "new": "y", "replace_all": "yes"},
    ):
        with pytest.raises((TypeError, ValueError)):
            TextEdit(**arguments)

    fs = InMemoryWorkspace("files", {"/a": b"one two"})
    current_scope = scope(fs)
    with execution_context(scope=current_scope):
        for keyword in ({"replace_all": 1}, {"replace_all": "yes"}):
            with pytest.raises(TypeError):
                EditTool()("a", "one", "1", **keyword)
        assert fs.read_text("/a") == "one two"


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
        current_scope = current_scope.with_overrides(
            workspace=current_scope.workspace.with_cwd("/other")
        )
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


@pytest.mark.parametrize(
    "text, arguments",
    [
        ("old old", {"path": "a", "old": "old", "new": "new"}),
        (
            "old only",
            {
                "path": "a",
                "edits": [
                    {"old": "old", "new": "middle"},
                    {"old": "missing", "new": "new", "replace_all": True},
                ],
            },
        ),
        (
            "old only",
            {
                "path": "a",
                "edits": [
                    {"old": "old", "new": "x x"},
                    {"old": "x", "new": "y"},
                ],
            },
        ),
    ],
)
def test_invalid_edit_returns_observation_without_approval_or_effect(text, arguments):
    checkpoints, journal = InMemoryCheckpointStore(), InMemoryApprovalStore()
    fs = InMemoryWorkspace("files", {"/a": text.encode()})
    current = agent(checkpoints, journal)
    current.generator.forward = Mock(
        side_effect=[
            response("edit", arguments),
            response(),
        ]
    )
    assert current("edit", scope=scope(fs)) == "done"
    assert journal.pending("editor", "t", "r") == []
    history = checkpoints.load_state("editor", "t", "r")["messages"]["items"]
    assert "exactly once" in str(history) or "at least once" in str(history)
    with execution_context(scope=scope(fs)):
        assert fs.read_text("/a") == text


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
    current_scope = scope(
        fs,
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
