import os
from threading import Event

import pytest

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.runtime import (
    AgentApprovals,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryApprovalStore,
    InMemoryWorkspace,
    PermissionSet,
    AgentWorkspace,
)
from msgflux.runtime.approvals.agent import ApprovalBatch
from msgflux.runtime.context import execution_context
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tools import Hidden
from msgflux.tools.config import tool_config


pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX local executor")


def _tool_response(name, arguments):
    response = ModelResponse()
    response.set_response_type("tool_call")
    calls = ToolCallAggregator()
    calls.process(0, "workspace-call", name, arguments)
    response.add(calls)
    response.reasoning = None
    response.metadata = {}
    return response


class _WorkspaceModel:
    model_type = "chat_completion"

    def __init__(self, tool_name, arguments):
        self.responses = [_tool_response(tool_name, arguments), self._text("complete")]
        self.requests = []

    @staticmethod
    def _text(value):
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add(value)
        response.reasoning = None
        response.metadata = {}
        return response

    async def acall(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)

    def __call__(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


class _WorkspaceToolFlowModel(_WorkspaceModel):
    """Script real workspace tools, then require their output in the answer."""

    def __init__(self):
        self.responses = [
            _tool_response("read", '{"path":"status.txt"}'),
            _tool_response(
                "edit",
                '{"path":"status.txt","old":"PENDING","new":"ACTIVE"}',
            ),
            _tool_response("read", '{"path":"status.txt"}'),
            _tool_response("bash", '{"command":"cat status.txt"}'),
        ]
        self.requests = []

    async def acall(self, **kwargs):
        self.requests.append(kwargs)
        if self.responses:
            return self.responses.pop(0)
        assert "status: ACTIVE" in str(kwargs)
        return self._text("Verified command output: status: ACTIVE")


def test_local_workspace_files_and_commands_share_project(tmp_path):
    (tmp_path / "sub").mkdir()
    workspace = AgentWorkspace.local(tmp_path)

    workspace.write_text("sub/note.txt", "hello")
    scoped = workspace.with_cwd("/sub")
    result = scoped.run(["cat", "note.txt"], timeout=5)

    assert scoped.cwd == "/sub"
    assert workspace.cwd == "/"
    assert scoped.read_text("note.txt") == "hello"
    assert result.returncode == 0
    assert result.stdout == b"hello"


def test_local_workspace_limits_and_path_validation(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    workspace.write_text("sample.txt", "abcdef")

    with pytest.raises(ValueError, match="max_bytes"):
        workspace.read_text("sample.txt", max_bytes=3)
    with pytest.raises(ValueError):
        workspace.read_bytes("sample.txt", max_bytes=0)
    with pytest.raises(ValueError):
        workspace.read_text("../outside")


def test_workspace_editor_prepares_exact_relative_path_proposal(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    (tmp_path / "src").mkdir()
    scoped = workspace.with_cwd("src")
    scoped.write_text("note.txt", "before marker after\n")

    proposal = scoped.editor.prepare_edit("note.txt", "marker", "reviewed")

    assert proposal.path == "/src/note.txt"
    assert proposal.before == "before marker after\n"
    assert proposal.after == "before reviewed after\n"
    assert "-before marker after" in proposal.diff
    assert "+before reviewed after" in proposal.diff
    assert scoped.read_text("note.txt") == "before marker after\n"
    with pytest.raises(ValueError, match="exactly once"):
        scoped.editor.prepare_edit("note.txt", "absent", "changed")


def test_editor_rejects_unreviewed_proposal_inside_approval_batch(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    proposal = workspace.editor.prepare_write("note.txt", "not approved")

    with execution_context(scope=ExecutionScope(workspace=workspace)):
        with ApprovalBatch(None, None, {}).activate():
            with pytest.raises(PermissionError, match="reviewed workspace proposal"):
                workspace.editor.apply(proposal)

    assert not (tmp_path / "note.txt").exists()


def test_read_only_workspace_rejects_mutation_and_commands(tmp_path):
    workspace = AgentWorkspace.local(tmp_path, read_only=True)
    assert workspace.read_only
    assert not workspace.supports_execution

    with pytest.raises(PermissionError):
        workspace.write_text("blocked.txt", "no")
    with pytest.raises(PermissionError):
        workspace.mkdir("blocked")
    with pytest.raises(PermissionError):
        workspace.run(["true"])
    assert not (tmp_path / "blocked.txt").exists()
    assert not (tmp_path / "blocked").exists()


def test_workspace_scope_does_not_gain_factory_grants(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    scope = ExecutionScope(workspace=workspace, permissions=PermissionSet())

    with execution_context(scope=scope):
        with pytest.raises(PermissionError):
            workspace.write_text("denied.txt", "no")
        with pytest.raises(PermissionError):
            workspace.run(["true"])
    assert not (tmp_path / "denied.txt").exists()


def test_nested_scope_restriction_takes_precedence_over_workspace_defaults(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    workspace.write_text("secret.txt", "secret")

    with execution_context(scope=ExecutionScope(workspace=workspace)):
        with execution_context(
            scope=ExecutionScope(workspace=workspace, permissions=PermissionSet())
        ):
            with pytest.raises(PermissionError):
                workspace.read_text("secret.txt")
        assert workspace.read_text("secret.txt") == "secret"


def test_custom_workspace_mutation_refuses_active_approval_batch(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    scope = ExecutionScope(workspace=workspace)

    with execution_context(scope=scope), ApprovalBatch(None, None, {}).activate():
        with pytest.raises(PermissionError, match="reviewed workspace proposal"):
            workspace.write_text("needs-review.txt", "not applied")

    assert not (tmp_path / "needs-review.txt").exists()


def test_workspace_cwd_views_are_independent(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    workspace = AgentWorkspace.local(tmp_path)
    one = workspace.with_cwd("one")
    two = workspace.with_cwd("/two")

    one.write_text("value", "one")
    two.write_text("value", "two")

    assert one.read_text("value") == "one"
    assert two.read_text("value") == "two"
    assert workspace.listdir("one") == ("value",)
    assert workspace.listdir("two") == ("value",)


@pytest.mark.asyncio
async def test_agent_injects_default_workspace_instance_without_exposing_schema(
    tmp_path,
):
    workspace = AgentWorkspace.local(tmp_path)
    workspace.write_text("default.txt", "from default")
    seen = []

    @tool_config(runtime_inputs=["workspace"])
    async def inspect(*, workspace: Hidden[AgentWorkspace]) -> str:
        """Read a file from the configured workspace."""
        seen.append(workspace)
        return await workspace.aread_text("default.txt")

    model = _WorkspaceModel("inspect", "{}")
    agent = Agent(
        name="workspace-agent", model=model, tools=[inspect], workspace=workspace
    )
    schema = agent.tool_library.get_tool_definition("inspect").input_schema

    assert "workspace" not in schema["properties"]
    assert await agent.acall("inspect") == "complete"
    assert seen == [workspace]


def test_sync_agent_uses_init_workspace_and_excludes_it_from_state_dict(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    workspace.write_text("default.txt", "from default")
    seen = []

    @tool_config(runtime_inputs=["workspace"])
    def inspect(*, workspace: Hidden[AgentWorkspace]) -> str:
        """Read a file from the configured workspace."""
        seen.append(workspace)
        return workspace.read_text("default.txt")

    model = _WorkspaceModel("inspect", "{}")
    agent = Agent(
        name="workspace-agent", model=model, tools=[inspect], workspace=workspace
    )

    assert agent("inspect") == "complete"
    assert seen == [workspace]
    assert all("workspace" not in key for key in agent.state_dict())


@pytest.mark.asyncio
async def test_agent_executes_workspace_read_edit_read_and_bash_tools(tmp_path):
    from msgflux.tools.builtin import BashTool, EditTool, ReadFileTool

    (tmp_path / "status.txt").write_text("status: PENDING\n")
    workspace = AgentWorkspace.local(tmp_path)
    model = _WorkspaceToolFlowModel()
    agent = Agent(
        name="workspace-tool-flow",
        model=model,
        tools=[ReadFileTool(), EditTool(), BashTool()],
        workspace=workspace,
    )

    answer = await agent.acall("Activate and verify the workspace status.")

    assert answer == "Verified command output: status: ACTIVE"
    assert len(model.requests) == 5
    assert (tmp_path / "status.txt").read_text() == "status: ACTIVE\n"


@pytest.mark.asyncio
async def test_scope_workspace_overrides_agent_default_without_duplicate_environment(
    tmp_path,
):
    (tmp_path / "default").mkdir()
    (tmp_path / "alternate").mkdir()
    default = AgentWorkspace.local(tmp_path / "default")
    alternate = AgentWorkspace.local(tmp_path / "alternate")
    default.write_text("value.txt", "default")
    alternate.write_text("value.txt", "alternate")
    seen = []

    @tool_config(runtime_inputs=["workspace"])
    async def inspect(*, workspace: Hidden[AgentWorkspace]) -> str:
        """Read the selected workspace."""
        seen.append(workspace)
        return await workspace.aread_text("value.txt")

    model = _WorkspaceModel("inspect", "{}")
    agent = Agent(
        name="workspace-agent", model=model, tools=[inspect], workspace=default
    )
    assert (
        await agent.acall("inspect", scope=ExecutionScope(workspace=alternate))
        == "complete"
    )
    assert seen == [alternate]
    assert not hasattr(ExecutionScope(workspace=default), "environment")


def test_background_tool_keeps_injected_workspace_and_hides_it_from_schema(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    workspace.write_text("background.txt", "available")
    completed = Event()
    observed = []

    @tool_config(runtime_inputs=["workspace"], background=True)
    def inspect_later(*, workspace: Hidden[AgentWorkspace]) -> str:
        """Read through the workspace in a background task."""
        try:
            observed.append((workspace, workspace.read_text("background.txt")))
            return observed[-1][1]
        finally:
            completed.set()

    library = ToolLibrary(name="workspace", tools=[inspect_later])
    schema = library.get_tool_definition("inspect_later").input_schema
    assert "workspace" not in schema["properties"]

    with execution_context(scope=ExecutionScope(workspace=workspace)):
        dispatch = library([("background-call", "inspect_later", {})])
    assert "task_id='" in dispatch.tool_calls[0].result
    assert completed.wait(timeout=5)
    assert observed == [(workspace, "available")]


def test_builtin_tools_share_workspace_cwd_and_allow_workspace_views(tmp_path):
    from msgflux.tools.builtin import BashTool, LsTool, ReadFileTool, WriteTool

    (tmp_path / "src").mkdir()
    workspace = AgentWorkspace.local(tmp_path).with_cwd("src")
    with execution_context(scope=ExecutionScope(workspace=workspace)):
        WriteTool()("note.txt", "shared", workspace=workspace)
        assert ReadFileTool()("note.txt", workspace=workspace) == "shared"
        assert LsTool()(workspace=workspace)["path"] == "/src"
        assert LsTool()(workspace=workspace.with_cwd("/"))["path"] == "/"
        result = BashTool()("cat note.txt", workspace=workspace)
        assert result.results[0].stdout == "shared"
    assert (tmp_path / "src" / "note.txt").read_text() == "shared"


@pytest.mark.asyncio
async def test_agent_scope_cannot_escalate_workspace_ceiling_or_catalog_grant(
    tmp_path,
):
    workspace = AgentWorkspace.local(
        tmp_path, permissions=PermissionSet(["filesystem.read"])
    )
    observed = []

    @tool_config(runtime_inputs=["workspace"], required_permissions=["catalog.write"])
    async def attempt_changes(*, workspace: Hidden[AgentWorkspace]) -> str:
        """Try file and process operations against the injected workspace."""
        for operation in (
            lambda: workspace.write_text("blocked.txt", "no"),
            lambda: workspace.run(["true"]),
        ):
            with pytest.raises(PermissionError):
                operation()
            observed.append("denied")
        return "workspace operations denied"

    model = _WorkspaceModel("attempt_changes", "{}")
    agent = Agent(
        name="workspace-ceiling-agent",
        model=model,
        tools=[attempt_changes],
        workspace=workspace,
    )
    generous_scope = ExecutionScope(
        permissions=PermissionSet(
            ["catalog.write", "filesystem.read", "filesystem.write", "process.execute"]
        )
    )

    assert (
        await agent.acall("Try the available changes", scope=generous_scope)
        == "complete"
    )
    assert observed == ["denied", "denied"]
    assert not (tmp_path / "blocked.txt").exists()

    # Direct host calls receive the same ceiling as tools dispatched by Agent.
    with execution_context(
        scope=ExecutionScope(
            workspace=workspace, permissions=generous_scope.permissions
        )
    ):
        with pytest.raises(PermissionError):
            workspace.write_text("direct.txt", "no")
        with pytest.raises(PermissionError):
            workspace.run(["true"])


def test_workspace_scope_can_narrow_broad_local_ceiling_to_exact_file(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    workspace.write_text("allowed.txt", "allowed")
    workspace.write_text("private.txt", "private")
    scope = ExecutionScope(
        workspace=workspace,
        permissions=PermissionSet(
            resources=[workspace.permission("allowed.txt", "filesystem.read")]
        ),
    )

    with execution_context(scope=scope):
        assert workspace.read_text("allowed.txt") == "allowed"
        with pytest.raises(PermissionError):
            workspace.read_text("private.txt")


def test_same_environment_handle_cannot_escape_its_own_narrow_ceiling(tmp_path):
    writable = AgentWorkspace.local(tmp_path)
    narrow = AgentWorkspace.from_environment(
        writable._environment, permissions=PermissionSet(["filesystem.read"])
    )
    scope = ExecutionScope(workspace=writable)

    with execution_context(scope=scope):
        with pytest.raises(PermissionError):
            narrow.write_text("blocked.txt", "no")

    assert not (tmp_path / "blocked.txt").exists()


def test_background_tool_cannot_expand_live_workspace_ceiling(tmp_path):
    workspace = AgentWorkspace.local(
        tmp_path, permissions=PermissionSet(["filesystem.read"])
    )
    completed = Event()
    outcomes = []

    @tool_config(runtime_inputs=["workspace"], background=True)
    def write_later(*, workspace: Hidden[AgentWorkspace]) -> str:
        """Try to write from a background worker."""
        try:
            workspace.write_text("background.txt", "no")
        except PermissionError:
            outcomes.append("denied")
        else:
            outcomes.append("written")
        finally:
            completed.set()
        return outcomes[-1]

    library = ToolLibrary(name="workspace", tools=[write_later])
    with execution_context(
        scope=ExecutionScope(
            workspace=workspace,
            permissions=PermissionSet(["filesystem.read", "filesystem.write"]),
        )
    ):
        library([("background-call", "write_later", {})])

    assert completed.wait(timeout=5)
    assert outcomes == ["denied"]
    assert not (tmp_path / "background.txt").exists()


@pytest.mark.asyncio
async def test_approved_write_resume_is_rejected_when_workspace_is_narrowed(tmp_path):
    from msgflux.tools.builtin import WriteTool

    filesystem = InMemoryWorkspace("approval-ceiling", {"/note.txt": b"before"})
    environment = ExecutionEnvironment(filesystem)
    writable = AgentWorkspace.from_environment(
        environment,
        permissions=PermissionSet(
            resources=[
                filesystem.permission("/note.txt", "filesystem.read"),
                filesystem.permission("/note.txt", "filesystem.write"),
            ]
        ),
    )
    read_only = AgentWorkspace.from_environment(
        environment,
        permissions=PermissionSet(
            resources=[filesystem.permission("/note.txt", "filesystem.read")]
        ),
    )
    journal = InMemoryApprovalStore()
    model = _WorkspaceModel("write", '{"path":"note.txt","content":"after"}')
    agent = Agent(
        name="approval-ceiling-agent",
        model=model,
        tools=[WriteTool()],
        checkpoint_store=InMemoryCheckpointStore(),
        approvals=AgentApprovals(journal, {"write": "v1"}, "p1"),
    )
    initial_scope = ExecutionScope(
        namespace="approval-ceiling-agent",
        thread_id="approval-thread",
        run_id="approval-run",
        principal="user:1",
        workspace=writable,
    )

    with pytest.raises(TaskPauseRequestedError):
        await agent.acall("Write after", scope=initial_scope)
    record = journal.pending(
        "approval-ceiling-agent", "approval-thread", "approval-run"
    )[0]
    agent.decide_approval(record.request_id, approved=True, decided_by="host")

    narrowed_scope = initial_scope.with_overrides(workspace=read_only)
    with pytest.raises(TaskPauseRequestedError):
        await agent.acall("Resume", scope=narrowed_scope)

    assert read_only.read_text("note.txt") == "before"
    assert journal.get("approval-ceiling-agent", record.request_id).status != "consumed"


def test_consumed_approval_does_not_override_read_only_workspace(tmp_path):
    from msgflux.runtime.approvals.agent import ApprovalBatch

    filesystem = InMemoryWorkspace("consumed-ceiling", {"/note.txt": b"before"})
    environment = ExecutionEnvironment(filesystem)
    writable = AgentWorkspace.from_environment(
        environment,
        permissions=PermissionSet(
            resources=[
                filesystem.permission("/note.txt", "filesystem.read"),
                filesystem.permission("/note.txt", "filesystem.write"),
            ]
        ),
    )
    read_only = AgentWorkspace.from_environment(
        environment,
        permissions=PermissionSet(
            resources=[filesystem.permission("/note.txt", "filesystem.read")]
        ),
    )

    with execution_context(scope=ExecutionScope(workspace=writable)):
        change = writable.editor.prepare_write("note.txt", "after")

    batch = ApprovalBatch(None, None, {}, changes={"approved-call": change})
    batch.consumed.add("approved-call")
    with execution_context(scope=ExecutionScope(workspace=read_only)):
        with batch.activate():
            with pytest.raises(PermissionError):
                read_only.editor.apply(change)

    assert read_only.read_text("note.txt") == "before"
