"""Strict request and typed client contracts for workspace policy routes."""

import msgspec
import pytest
import httpx2

from msgflux.nn import Agent
from msgflux.runtime.permissions import PermissionSet, ResourcePermission
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    SQLiteServiceStore,
    ServiceConflictError,
)
from msgflux.runtime.service.http.app import create_service_app
from msgflux.runtime.service.http.client import (
    AgentServiceClient,
    AgentServiceHTTPError,
)
from msgflux.runtime.service.http.session import AgentSessionClient
from msgflux.runtime.service.http.records import WorkspacePolicyRequest
from msgflux.runtime.workspace.policy import WorkspacePolicy
from msgflux.runtime.workspace.api import AgentWorkspace
from unittest.mock import Mock


def test_workspace_policy_request_round_trips_named_and_exact_grants():
    request = WorkspacePolicyRequest(
        permissions=("filesystem.read", "process.execute"),
        resources=(ResourcePermission("workspace:repo:/README.md", "filesystem.read"),),
        approval_policy="on-request",
        expected_revision=4,
    )

    restored = msgspec.json.decode(
        msgspec.json.encode(request), type=WorkspacePolicyRequest, strict=True
    )
    assert restored == request
    assert PermissionSet(
        frozenset(restored.permissions), frozenset(restored.resources)
    ).resources == frozenset(request.resources)


@pytest.mark.parametrize(
    "payload",
    [
        b'{"permissions":["filesystem.read"],"unknown":true}',
        b'{"permissions":"all"}',
        b'{"approval_policy":"yolo"}',
        b'{"expected_revision":-1}',
        b'{"expected_revision":true}',
    ],
)
def test_workspace_policy_request_rejects_malformed_values(payload):
    with pytest.raises((msgspec.DecodeError, ValueError, TypeError)):
        msgspec.json.decode(payload, type=WorkspacePolicyRequest, strict=True)


def test_workspace_policy_request_exact_permission_validation():
    request = msgspec.json.decode(
        b'{"permissions":["filesystem.*"]}',
        type=WorkspacePolicyRequest,
        strict=True,
    )
    with pytest.raises(ValueError, match="exact capability"):
        PermissionSet(frozenset(request.permissions))


def test_workspace_policy_preset_rejects_even_empty_resource_list():
    request = msgspec.json.decode(
        b'{"permissions":"full-access","resources":[]}',
        type=WorkspacePolicyRequest,
        strict=True,
    )
    assert isinstance(request.permissions, str)
    assert request.resources == ()


def test_permission_set_wire_values_preserve_only_exact_resource_ids():
    permissions = PermissionSet(
        frozenset({"filesystem.read"}),
        frozenset({ResourcePermission("workspace:repo:/README.md", "filesystem.read")}),
    )
    request = WorkspacePolicyRequest(
        permissions=tuple(sorted(permissions.grants)),
        resources=tuple(permissions.resources),
    )
    decoded = msgspec.json.decode(
        msgspec.json.encode(request), type=WorkspacePolicyRequest
    )
    assert decoded.resources == tuple(permissions.resources)


def test_policy_response_wire_round_trip_is_typed():
    policy = WorkspacePolicy(
        thread_id="thread-1",
        permissions=("filesystem.read",),
        resources=(ResourcePermission("workspace:repo:/README.md", "filesystem.read"),),
        approval_policy="on-request",
        revision=2,
        updated_at="2026-10-07T12:00:00+00:00",
    )
    decoded = msgspec.json.decode(msgspec.json.encode(policy), type=WorkspacePolicy)
    assert decoded == policy


@pytest.mark.asyncio
async def test_authenticated_policy_routes_typed_client_ceiling_and_cas(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    journal = SQLiteServiceStore(tmp_path / "service.sqlite3")

    def factory(_thread):
        model = Mock()
        model.model_type = "chat_completion"
        workspace = AgentWorkspace.local(root, read_only=True)
        agent = Agent(name="policy-http", model=model, workspace=workspace)
        return AgentSession(agent)

    service = AgentService(store=journal)
    service.register("agent", factory)
    app = create_service_app(service, token="secret")
    transport = httpx2.ASGITransport(app=app)
    http = httpx2.AsyncClient(transport=transport, base_url="http://test")
    client = AgentServiceClient("http://test", token="secret", client=http)
    try:
        denied = await http.put(
            "/v1/threads/missing/workspace-policy",
            content=b"{ malformed",
            headers={"content-type": "application/json"},
        )
        assert denied.status_code == 401

        session = await AgentSessionClient.open(
            client,
            agent_id="agent",
            thread_id="policy-thread",
            cwd=root,
        )
        initial = await session.workspace_policy()
        assert isinstance(initial, WorkspacePolicy)
        assert initial.permissions == ("filesystem.list", "filesystem.read")

        updated = await session.update_workspace_policy(
            permissions="full-access",
            approval_policy="on-request",
            expected_revision=initial.revision,
        )
        assert updated.revision == initial.revision + 1
        assert updated.permissions == initial.permissions
        assert updated.approval_policy == "on-request"
        assert await client.workspace_policy(session.thread_id) == updated
        assert journal.workspace_policy(session.thread_id) == updated

        with pytest.raises(ServiceConflictError):
            await session.update_workspace_policy(
                permissions="read-only",
                expected_revision=initial.revision,
            )

        invalid = await http.put(
            f"/v1/threads/{session.thread_id}/workspace-policy",
            headers={"Authorization": "Bearer secret"},
            json={"permissions": ["filesystem.*"]},
        )
        assert invalid.status_code == 422
        conflict_preset = await http.put(
            f"/v1/threads/{session.thread_id}/workspace-policy",
            headers={"Authorization": "Bearer secret"},
            json={"permissions": "full-access", "resources": []},
        )
        assert conflict_preset.status_code == 422
    finally:
        await client.aclose()
        await http.aclose()
        await service.aclose()
        journal.close()


@pytest.mark.asyncio
async def test_policy_get_without_workspace_is_empty_and_update_is_rejected():
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(name="no-workspace", model=model)
    service = AgentService(store=SQLiteServiceStore())
    service.register("agent", lambda _thread: AgentSession(agent))
    app = create_service_app(service, token="secret")
    transport = httpx2.ASGITransport(app=app)
    http = httpx2.AsyncClient(transport=transport, base_url="http://test")
    client = AgentServiceClient("http://test", token="secret", client=http)
    try:
        session = await AgentSessionClient.open(
            client, agent_id="agent", thread_id="no-workspace-thread"
        )
        current = await session.workspace_policy()
        assert current.permissions == ()
        assert current.resources == ()
        assert current.revision == 0
        with pytest.raises(AgentServiceHTTPError) as error:
            await session.update_workspace_policy(permissions="full-access")
        assert error.value.status_code == 422
        assert service.store.workspace_policy(session.thread_id) is None
    finally:
        await client.aclose()
        await http.aclose()
        await service.aclose()
        service.store.close()
