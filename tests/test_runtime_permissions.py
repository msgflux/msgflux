import asyncio

import pytest

from msgflux.runtime import (
    ExecutionScope,
    PermissionSet,
    execution_context,
    get_execution_scope,
)


def test_permission_set_is_immutable_and_exact():
    source = ["filesystem.read", "process.execute"]
    grants = PermissionSet(source)
    source.append("filesystem.write")
    assert grants.missing(["filesystem.write"]) == ("filesystem.write",)
    assert grants.intersect(
        PermissionSet(["filesystem.read", "network.connect"])
    ).grants == {"filesystem.read"}
    with pytest.raises(ValueError):
        PermissionSet(["filesystem.*"])
    with pytest.raises(TypeError):
        PermissionSet("filesystem.read")


def test_nested_scope_cannot_widen_or_change_principal():
    root = ExecutionScope(
        principal="user:1", permissions=PermissionSet(["filesystem.read"])
    )
    with execution_context(scope=root):
        with execution_context(
            scope=ExecutionScope(
                permissions=PermissionSet(["filesystem.read", "process.execute"])
            )
        ) as child:
            assert child.permissions.grants == {"filesystem.read"}
            assert child.principal == "user:1"
        with execution_context(
            scope=ExecutionScope(permissions=PermissionSet())
        ) as child:
            assert child.permissions.grants == set()
        with pytest.raises(ValueError, match="principal"):
            with execution_context(scope=ExecutionScope(principal="admin")):
                pass
    assert get_execution_scope().permissions is None


def test_scope_identity_serialization_does_not_restore_authority():
    root = ExecutionScope(
        run_id="run", principal="user:1", permissions=PermissionSet(["filesystem.read"])
    )
    snapshot = root.to_dict()
    assert "permissions" not in snapshot
    assert "principal" not in snapshot
    with execution_context(scope=ExecutionScope(**snapshot)) as restored:
        assert restored.permissions == PermissionSet()
        assert restored.principal is None


@pytest.mark.asyncio
async def test_concurrent_roots_do_not_share_grants():
    async def run(name):
        with execution_context(scope=ExecutionScope(permissions=PermissionSet([name]))):
            await asyncio.sleep(0)
            return get_execution_scope().permissions.grants

    assert await asyncio.gather(run("read"), run("write")) == [{"read"}, {"write"}]
