"""Runtime clipping for durable workspace policy updates."""

import pytest
import msgspec

from msgflux.runtime.context import (
    ExecutionScope,
    _get_execution_scope_ceiling,
    execution_context,
    get_execution_context,
    get_execution_scope,
)
from msgflux.runtime.permissions import (
    PermissionSet,
    ResourcePermission,
    require_permissions,
)
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.runtime.workspace.policy import WorkspacePolicy, WorkspacePolicyState


def _scope(thread_id, workspace=None, permissions=None):
    return ExecutionScope(
        thread_id=thread_id,
        workspace=workspace,
        permissions=permissions,
    )


def _policy(thread_id, permissions, *, resources=(), revision=0):
    return WorkspacePolicy(
        thread_id=thread_id,
        permissions=tuple(permissions),
        resources=tuple(resources),
        revision=revision,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"thread_id": " "}, "thread_id"),
        ({"permissions": ("bad*grant",)}, "Permissions"),
        ({"approval_policy": "always"}, "approval_policy"),
        ({"revision": -1}, "revision"),
        ({"revision": True}, "revision"),
        ({"updated_at": " "}, "updated_at"),
    ],
)
def test_workspace_policy_rejects_invalid_values(kwargs, message):
    values = {
        "thread_id": "thread-1",
        "permissions": ("filesystem.read",),
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        WorkspacePolicy(**values)


def test_policy_permission_set_normalizes_and_round_trips_resources():
    item = ResourcePermission("workspace:workspace-1:/src/a.py", "filesystem.write")
    policy = WorkspacePolicy(
        thread_id="thread-1",
        permissions=("filesystem.write", "filesystem.read", "filesystem.read"),
        resources=(item, item),
        approval_policy="on-request",
    )
    assert policy.permissions == ("filesystem.read", "filesystem.write")
    assert policy.resources == (item,)
    assert policy.permission_set() == PermissionSet(
        frozenset({"filesystem.read", "filesystem.write"}), frozenset({item})
    )
    assert (
        msgspec.convert(msgspec.to_builtins(policy), type=WorkspacePolicy, strict=True)
        == policy
    )


def test_active_call_observes_policy_restriction_and_later_expansion(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    full = workspace.permissions
    ceiling = PermissionSet(full.grants | {"custom.invoke"}, full.resources)
    state = WorkspacePolicyState(
        workspace,
        _policy("thread-1", full.grants, resources=full.resources),
    )
    with execution_context(
        scope=_scope("thread-1", workspace, ceiling),
        workspace_policy=state,
    ):
        require_permissions(("filesystem.write",))
        state.current = _policy(
            "thread-1", ("filesystem.read", "filesystem.list"), revision=1
        )
        assert get_execution_scope().permissions == PermissionSet(
            frozenset({"filesystem.read", "filesystem.list", "custom.invoke"})
        )
        assert _get_execution_scope_ceiling().permissions == ceiling
        with pytest.raises(PermissionError, match=r"filesystem\.write"):
            require_permissions(("filesystem.write",))
        require_permissions(("custom.invoke",))

        state.current = _policy("thread-1", full.grants, revision=2)
        require_permissions(("filesystem.write",))
        assert get_execution_context()["workspace_policy"] is state


def test_child_read_only_scope_remains_a_ceiling_after_parent_expansion(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    full = workspace.permissions
    state = WorkspacePolicyState(
        workspace,
        _policy("thread-1", full.grants, resources=full.resources),
    )
    readonly = PermissionSet(frozenset({"filesystem.read", "filesystem.list"}))
    with execution_context(
        scope=_scope("thread-1", workspace, full),
        workspace_policy=state,
    ):
        with execution_context(scope=_scope("thread-1", workspace, readonly)):
            with pytest.raises(PermissionError, match=r"filesystem\.write"):
                require_permissions(("filesystem.write",))
            state.current = _policy("thread-1", full.grants, revision=1)
            with pytest.raises(PermissionError, match=r"filesystem\.write"):
                require_permissions(("filesystem.write",))
        require_permissions(("filesystem.write",))


def test_exact_resources_are_clipped_and_workspace_driver_is_immutable(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    write_a = workspace.permission("/a.txt", "filesystem.write")
    write_b = workspace.permission("/b.txt", "filesystem.write")
    ceiling = PermissionSet(resources=frozenset({write_a, write_b}))
    state = WorkspacePolicyState(
        workspace,
        _policy("thread-1", (), resources=(write_a,)),
    )
    with execution_context(
        scope=_scope("thread-1", workspace, ceiling),
        workspace_policy=state,
    ):
        require_permissions((), (write_a,))
        with pytest.raises(PermissionError, match="resource permissions"):
            require_permissions((), (write_b,))
        other_root = tmp_path / "other"
        other_root.mkdir()
        other = AgentWorkspace.local(other_root)
        with pytest.raises(ValueError, match="workspace driver"):
            with execution_context(scope=_scope("thread-1", other, ceiling)):
                get_execution_scope()


def test_policy_state_isolated_by_thread_and_cannot_be_rebound(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    read = PermissionSet(frozenset({"filesystem.read"}))
    write = PermissionSet(frozenset({"filesystem.write"}))
    first = WorkspacePolicyState(workspace, _policy("thread-1", read.grants))
    second = WorkspacePolicyState(workspace, _policy("thread-2", write.grants))

    with execution_context(
        scope=_scope("thread-1", workspace, read), workspace_policy=first
    ):
        assert get_execution_scope().permissions == read
        with pytest.raises(ValueError, match="replace workspace policy"):
            with execution_context(
                scope=_scope("thread-2", workspace, write), workspace_policy=second
            ):
                pass
    with execution_context(
        scope=_scope("thread-2", workspace, write), workspace_policy=second
    ):
        assert get_execution_scope().permissions == write


def test_workspace_policy_without_workspace_still_clips_runtime_permissions():
    state = WorkspacePolicyState(None, _policy("thread-1", ("custom.read",)))
    ceiling = PermissionSet(frozenset({"custom.read", "custom.write"}))
    with execution_context(
        scope=_scope("thread-1", permissions=ceiling), workspace_policy=state
    ):
        require_permissions(("custom.read",))
        with pytest.raises(PermissionError, match=r"custom\.write"):
            require_permissions(("custom.write",))


def test_policy_state_rejects_thread_switch_and_nonincreasing_revision(tmp_path):
    workspace = AgentWorkspace.local(tmp_path)
    state = WorkspacePolicyState(workspace, _policy("thread-1", ()))
    with pytest.raises(ValueError, match="thread identity"):
        state.current = _policy("thread-2", (), revision=1)
    with pytest.raises(ValueError, match="revision must increase"):
        state.current = _policy("thread-1", (), revision=0)
