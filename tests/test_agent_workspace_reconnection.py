"""Agent checkpoints retain workspace identity without retaining authority."""

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.nn.modules.agent.conversation import AgentConversationMixin
from msgflux.nn.modules.agent.lifecycle import AgentLifecycleMixin
from msgflux.runtime import (
    AgentWorkspace,
    ExecutionEnvironment,
    ExecutionScope,
    InMemoryWorkspaceBackend,
    PermissionSet,
    execution_context,
)


class _Agent(AgentConversationMixin, AgentLifecycleMixin):
    def __init__(self, workspace):
        self.workspace = workspace

    def get_module_name(self):
        return "agent:test"


@pytest.mark.asyncio
async def test_reconnect_validates_cwd_and_releases_only_failed_binding():
    backend = InMemoryWorkspaceBackend({"/work/file.txt": b"ok"})
    opened = await backend.open("project")
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        await AgentWorkspace.reconnect(
            backend, "project", opened.identity, cwd="/missing"
        )
    assert opened.state == "open"
    assert opened.filesystem.requires_binding
    # Reconnection owns its own binding and no grants are restored by identity.
    workspace = await AgentWorkspace.reconnect(
        backend,
        "project",
        opened.identity,
        cwd="/work",
        permissions=None,
    )
    assert workspace.permissions == PermissionSet()
    with pytest.raises(PermissionError):
        workspace.read_text("file.txt")
    await workspace.aclose()
    await opened.aclose()


def test_checkpoint_workspace_reference_is_persisted_and_checked():
    backend = InMemoryWorkspaceBackend()
    import asyncio

    binding = asyncio.run(backend.open("project"))
    workspace = AgentWorkspace.from_environment(
        ExecutionEnvironment.from_binding(binding),
        cwd="/",
    )
    agent = _Agent(workspace)
    scope = ExecutionScope(workspace=workspace, run_id="run-1", thread_id="thread-1")
    messages = ChatMessages()
    messages.begin_turn(turn_id="run-1")
    with execution_context(scope=scope):
        state = agent._build_checkpoint_state(messages, status="running")
        assert state["runtime"]["extensions"]["workspace_reference"] == {
            "version": 1,
            "workspace_id": workspace.workspace_id,
            "identity": {
                "backend": workspace.identity.backend,
                "resource_id": workspace.identity.resource_id,
                "generation": workspace.identity.generation,
                "config_revision": workspace.identity.config_revision,
            },
            "cwd": "/",
        }
        agent._validate_checkpoint_workspace(state)

        changed_binding = asyncio.run(backend.open("other-project"))
        changed = AgentWorkspace.from_environment(
            ExecutionEnvironment.from_binding(changed_binding), cwd="/"
        )
        # A changed cwd is paused before resume can dispatch model/tools.
        cwd_override = workspace.with_cwd("/different")
        with execution_context(scope=scope.with_overrides(workspace=cwd_override)):
            with pytest.raises(TaskPauseRequestedError, match="cwd"):
                agent._validate_checkpoint_workspace(state)
        # Old checkpoints remain resumable without adding a workspace requirement.
        agent._validate_checkpoint_workspace({"runtime": {}})

    with execution_context(scope=scope.with_overrides(workspace=changed)):
        with pytest.raises(ValueError, match="does not match"):
            agent._validate_checkpoint_workspace(state)
    asyncio.run(binding.aclose())
    asyncio.run(changed_binding.aclose())
