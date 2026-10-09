"""Live workspace policies remain thread-local and clip each operation."""

import pytest
from unittest.mock import Mock

from msgflux.nn import Agent
from msgflux.runtime.context import (
    ExecutionScope,
    execution_context,
    get_execution_scope,
)
from msgflux.runtime.permissions import PermissionSet, require_permissions
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.workspace.api import AgentWorkspace


def _agent(name, workspace):
    model = Mock()
    model.model_type = "chat_completion"
    return Agent(name=name, model=model, workspace=workspace)


def _scope(thread_id, workspace, permissions=None):
    return ExecutionScope(
        thread_id=thread_id,
        workspace=workspace,
        permissions=permissions,
    )


@pytest.mark.asyncio
async def test_service_policy_change_clips_permissions_inside_an_active_scope(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    agent = _agent("live-policy", workspace)
    service = AgentService(store=SQLiteServiceStore())
    service.register("assistant", lambda _thread: AgentSession(agent))
    try:
        thread = await service.open_thread("assistant", thread_id="live-thread")
        lease = await service.acquire_session(thread.thread_id)
        session = lease.session
        scope = session.scope(thread.thread_id, run_id="live-run")
        try:
            with session.context(scope):
                require_permissions(("filesystem.write",))
                await service.update_workspace_policy(
                    thread.thread_id, permissions="read-only"
                )
                assert (
                    "filesystem.write" not in get_execution_scope().permissions.grants
                )
                with pytest.raises(PermissionError, match=r"filesystem\.write"):
                    require_permissions(("filesystem.write",))

                await service.update_workspace_policy(
                    thread.thread_id, permissions="full-access"
                )
                require_permissions(("filesystem.write",))
        finally:
            await lease.aclose()
    finally:
        await service.aclose()
        workspace.require_active()
        service.store.close()


@pytest.mark.asyncio
async def test_borrowed_workspace_policy_state_is_isolated_by_thread(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    agents = {}

    def factory(thread):
        agent = _agent(f"agent-{thread.thread_id}", workspace)
        agents[thread.thread_id] = agent
        return AgentSession(agent)

    service = AgentService(store=SQLiteServiceStore())
    service.register("assistant", factory)
    try:
        first = await service.open_thread("assistant", thread_id="policy-one")
        second = await service.open_thread("assistant", thread_id="policy-two")
        first_lease = await service.acquire_session(first.thread_id)
        second_lease = await service.acquire_session(second.thread_id)
        first_session = first_lease.session
        second_session = second_lease.session
        assert agents[first.thread_id].workspace.shares_environment(workspace)
        assert agents[second.thread_id].workspace.shares_environment(workspace)

        try:
            await service.update_workspace_policy(
                first.thread_id, permissions="read-only"
            )
            with first_session.context(first_session.scope(first.thread_id)):
                with pytest.raises(PermissionError, match=r"filesystem\.write"):
                    require_permissions(("filesystem.write",))
            with second_session.context(second_session.scope(second.thread_id)):
                require_permissions(("filesystem.write",))
        finally:
            await first_lease.aclose()
            await second_lease.aclose()
    finally:
        await service.aclose()
        workspace.require_active()
        service.store.close()


@pytest.mark.asyncio
async def test_child_readonly_workspace_ceiling_survives_parent_full_access(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    agent = _agent("child-ceiling", workspace)
    service = AgentService(store=SQLiteServiceStore())
    service.register("assistant", lambda _thread: AgentSession(agent))
    try:
        thread = await service.open_thread("assistant", thread_id="child-thread")
        lease = await service.acquire_session(thread.thread_id)
        session = lease.session
        try:
            await service.update_workspace_policy(
                thread.thread_id, permissions="full-access"
            )
            parent_scope = session.scope(thread.thread_id, run_id="parent-run")
            child_workspace = workspace.with_cwd("/child")
            readonly = PermissionSet(frozenset({"filesystem.read", "filesystem.list"}))
            with session.context(parent_scope):
                require_permissions(("filesystem.write",))
                with execution_context(
                    scope=_scope(thread.thread_id, child_workspace, readonly)
                ):
                    assert get_execution_scope().workspace.cwd == "/child"
                    require_permissions(("filesystem.read",))
                    with pytest.raises(PermissionError, match=r"filesystem\.write"):
                        require_permissions(("filesystem.write",))
                require_permissions(("filesystem.write",))
        finally:
            await lease.aclose()
    finally:
        await service.aclose()
        workspace.require_active()
        service.store.close()
