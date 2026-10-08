"""Automatic approvals derived from a thread's live workspace policy."""

from unittest.mock import AsyncMock, Mock

import msgflux as mf
import pytest

from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.modules.agent.context import _UNSET
from msgflux.runtime.approvals.workspace import (
    _definition_revision,
    get_workspace_approvals,
)
from msgflux.runtime.agent_run import AgentRun, agent_run_context
from msgflux.runtime.context import ExecutionScope, execution_context
from msgflux.runtime.permissions import PermissionSet, require_permissions
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.runtime.workspace.policy import WorkspacePolicy, WorkspacePolicyState
from msgflux.tools.builtin import WriteTool
from msgflux.utils.msgspec import msgspec_dumps


def _response(content="done"):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _tool_call(name, arguments):
    calls = ToolCallAggregator()
    calls.process(0, "call-1", name, msgspec_dumps(arguments))
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _policy(thread_id, workspace, *, mode="on-request", revision=0):
    permissions = workspace.permissions
    return WorkspacePolicy(
        thread_id=thread_id,
        permissions=tuple(permissions.grants),
        resources=tuple(permissions.resources),
        approval_policy=mode,
        revision=revision,
    )


def _agent(name, state_dir, workspace, responses, tool=None):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name=name,
        model=model,
        agent_dir=state_dir,
        workspace=workspace,
        tools=[] if tool is None else [tool],
    )
    agent.generator.aforward = AsyncMock(side_effect=responses)
    return agent


@pytest.mark.asyncio
async def test_managed_on_request_pauses_reviews_and_replays_write(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = AgentWorkspace.local(root)
    state = WorkspacePolicyState(workspace, _policy("thread-1", workspace))
    scope = ExecutionScope(
        thread_id="thread-1",
        namespace="editor",
        run_id="run-1",
        principal="user",
        workspace=workspace,
        permissions=workspace.permissions,
    )
    agent = _agent(
        "editor",
        tmp_path / "agent-state",
        workspace,
        [_tool_call("write", {"path": "note.txt", "content": "reviewed"}), _response()],
        WriteTool(),
    )
    try:
        with execution_context(scope=scope, workspace_policy=state):
            with pytest.raises(TaskPauseRequestedError):
                await agent.acall("Create a note", scope=scope)
            resources = agent._owned_threads[scope.thread_id].resources
            records = resources.approval_store.pending(
                scope.namespace, scope.thread_id, scope.run_id
            )
            assert len(records) == 1
            assert records[0].binding.tool_name == "write"
            assert records[0].binding.policy_version == "workspace-on-request-r0"
            assert not (root / "note.txt").exists()

            agent.decide_approval(
                records[0].request_id,
                approved=True,
                decided_by="user",
                expected_revision=records[0].revision,
            )
            state.current = _policy("thread-1", workspace, mode="never", revision=1)
            assert await agent.acall("Continue", scope=scope) == "done"
            assert (root / "note.txt").read_text() == "reviewed"
    finally:
        await agent.aclose()


def test_pending_approval_keeps_its_policy_version_after_override(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = AgentWorkspace.local(root)
    state = WorkspacePolicyState(
        workspace, _policy("thread-1", workspace, mode="never", revision=8)
    )
    scope = ExecutionScope(
        thread_id="thread-1",
        namespace="editor",
        run_id="run-1",
        workspace=workspace,
        permissions=workspace.permissions,
    )
    agent = _agent("editor", tmp_path / "agent-state", workspace, [], WriteTool())
    agent._bind_resources(scope.thread_id)
    run = AgentRun(
        namespace=scope.namespace,
        thread_id=scope.thread_id,
        run_id=scope.run_id,
        extension_state={
            "pending_approvals": {
                "policy_version": "workspace-on-request-r2",
                "intents": [],
                "requests": {},
                "phase": "review",
            }
        },
    )
    try:
        with execution_context(scope=scope, workspace_policy=state):
            with agent_run_context(run):
                with pytest.raises(TaskPauseRequestedError):
                    # The actual replay rejects the deliberately incomplete
                    # batch, but policy resolution must retain its binding.
                    agent._approval_replay()
                policy = get_workspace_approvals(agent)
                assert policy is not None
                assert policy.policy_version == "workspace-on-request-r2"
                forced = get_workspace_approvals(
                    agent, force=True, _policy_version="workspace-on-request-r2"
                )
                assert forced is not None
                assert forced.policy_version == "workspace-on-request-r2"
    finally:
        import asyncio

        asyncio.run(agent.aclose())


@pytest.mark.asyncio
async def test_never_mode_executes_future_write_without_creating_approval(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = AgentWorkspace.local(root)
    state = WorkspacePolicyState(
        workspace, _policy("thread-1", workspace, mode="never")
    )
    scope = ExecutionScope(
        thread_id="thread-1",
        namespace="editor",
        run_id="run-1",
        workspace=workspace,
        permissions=workspace.permissions,
    )
    agent = _agent(
        "editor",
        tmp_path / "agent-state",
        workspace,
        [
            _tool_call("write", {"path": "note.txt", "content": "unreviewed"}),
            _response(),
        ],
        WriteTool(),
    )
    try:
        with execution_context(scope=scope, workspace_policy=state):
            assert await agent.acall("Create a note", scope=scope) == "done"
        resources = agent._owned_threads[scope.thread_id].resources
        assert (root / "note.txt").read_text() == "unreviewed"
        assert (
            resources.approval_store.pending(
                scope.namespace, scope.thread_id, scope.run_id
            )
            == []
        )
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_read_only_policy_denies_workspace_write_even_with_approval_mode(
    tmp_path,
):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = AgentWorkspace.local(root)
    state = WorkspacePolicyState(
        workspace,
        WorkspacePolicy(
            thread_id="thread-1",
            permissions=("filesystem.read", "filesystem.list"),
            approval_policy="on-request",
        ),
    )
    scope = ExecutionScope(
        thread_id="thread-1",
        namespace="editor",
        run_id="run-1",
        workspace=workspace,
        permissions=workspace.permissions,
    )
    agent = _agent(
        "editor",
        tmp_path / "agent-state",
        workspace,
        [_tool_call("write", {"path": "blocked.txt", "content": "no"}), _response()],
        WriteTool(),
    )
    try:
        with execution_context(scope=scope, workspace_policy=state):
            with pytest.raises(PermissionError):
                require_permissions(("filesystem.write",))
            assert await agent.acall("Write a file", scope=scope) == "done"
        assert not (root / "blocked.txt").exists()
        records = agent._owned_threads[
            scope.thread_id
        ].resources.approval_store.pending(
            scope.namespace, scope.thread_id, scope.run_id
        )
        assert records == []
    finally:
        await agent.aclose()


def test_automatic_approval_policy_tracks_live_never_override(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = AgentWorkspace.local(root)
    state = WorkspacePolicyState(workspace, _policy("thread-1", workspace))
    scope = ExecutionScope(
        thread_id="thread-1",
        namespace="editor",
        workspace=workspace,
        permissions=workspace.permissions,
    )
    agent = _agent("editor", tmp_path / "agent-state", workspace, [], WriteTool())
    agent._bind_resources(scope.thread_id)
    try:
        with execution_context(scope=scope, workspace_policy=state):
            with agent._approval_context(_UNSET):
                assert agent._get_effective_approvals() is not None
                state.current = _policy("thread-1", workspace, mode="never", revision=1)
                assert agent._get_effective_approvals() is None
            assert get_workspace_approvals(agent) is None
    finally:
        import asyncio

        asyncio.run(agent.aclose())


def test_tool_revision_fingerprint_is_stable_and_tracks_implementation():
    def original(value: str) -> str:
        return value.upper()

    def changed(value: str) -> str:
        return value.lower()

    first_model, second_model = Mock(), Mock()
    first_model.model_type = second_model.model_type = "chat_completion"
    first_tool = mf.tool_config(
        name_override="record", required_permissions=("process.execute",)
    )(original)
    changed_tool = mf.tool_config(
        name_override="record", required_permissions=("process.execute",)
    )(changed)
    first_agent = Agent(name="revision-test", model=first_model, tools=[first_tool])
    second_agent = Agent(name="revision-test", model=second_model, tools=[changed_tool])
    try:
        first = first_agent.tool_library.registry.definitions()[0]
        changed_definition = second_agent.tool_library.registry.definitions()[0]
        assert _definition_revision(first) == _definition_revision(first)
        assert _definition_revision(first) != _definition_revision(changed_definition)
    finally:
        import asyncio

        asyncio.run(first_agent.aclose())
        asyncio.run(second_agent.aclose())


def test_on_request_never_silently_disables_without_owned_journal(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    workspace = AgentWorkspace.local(root)
    state = WorkspacePolicyState(workspace, _policy("thread-1", workspace))
    scope = ExecutionScope(
        thread_id="thread-1",
        namespace="editor",
        workspace=workspace,
        permissions=workspace.permissions,
    )
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="editor", model=model, workspace=workspace, tools=[WriteTool()])
    try:
        with execution_context(scope=scope, workspace_policy=state):
            with pytest.raises(ValueError, match="managed approval store"):
                get_workspace_approvals(agent)
    finally:
        import asyncio

        asyncio.run(agent.aclose())


def test_selected_nonforeground_tool_is_rejected_by_workspace_approval_policy(
    tmp_path,
):
    @mf.tool_config(background=True, required_permissions=("process.execute",))
    def execute_in_background(command: str) -> str:
        return command

    root = tmp_path / "workspace"
    root.mkdir()
    workspace = AgentWorkspace.local(root)
    state = WorkspacePolicyState(workspace, _policy("thread-1", workspace))
    scope = ExecutionScope(
        thread_id="thread-1",
        namespace="editor",
        workspace=workspace,
        permissions=workspace.permissions,
    )
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="editor",
        model=model,
        workspace=workspace,
        agent_dir=tmp_path / "agent-state",
        tools=[execute_in_background],
    )
    agent._bind_resources(scope.thread_id)
    try:
        with execution_context(scope=scope, workspace_policy=state):
            with pytest.raises(ValueError, match="non-foreground tool"):
                get_workspace_approvals(agent)
    finally:
        import asyncio

        asyncio.run(agent.aclose())
