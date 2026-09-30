import os
from threading import Event

import pytest

from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import (
    ExecutionEnvironment,
    ExecutionScope,
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


def test_read_only_workspace_rejects_mutation_and_commands(tmp_path):
    workspace = AgentWorkspace.local(tmp_path, read_only=True)
    assert workspace.read_only
    assert not workspace.can_execute

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
        with pytest.raises(PermissionError, match="reviewed approval"):
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
async def test_scope_workspace_overrides_agent_default_and_conflicts_fail(tmp_path):
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

    with pytest.raises(ValueError, match="Conflicting workspace and environment"):
        ExecutionScope(
            workspace=alternate,
            environment=ExecutionEnvironment(default._environment.filesystem),
        )


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
