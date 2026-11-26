"""HTTP/SDK contract for generic extension inventory and stopped-session updates."""

import httpx2
import pytest

pytest.importorskip("litestar")

from msgflux.runtime.service.http import (
    AgentServiceClient,
    AgentSessionClient,
    create_service_app,
)
from msgflux.runtime.service import ServiceConflictError
from tests.test_service_extension_controls import _setup


@pytest.mark.asyncio
async def test_http_and_both_client_facades_update_a_registered_arbitrary_extension():
    service, _ = _setup()
    app = create_service_app(service, token="secret")
    transport = httpx2.ASGITransport(app=app)
    try:
        async with httpx2.AsyncClient(
            transport=transport, base_url="http://test"
        ) as http:
            assert (await http.get("/v1/threads/t/extensions")).status_code == 401
            client = AgentServiceClient("http://test", token="secret", client=http)
            session = await AgentSessionClient.open(
                client, agent_id="main", thread_id="t"
            )
            before = await session.thread_extensions()
            changed = await session.update_extensions(
                scope="tree", enable=["marker"], expected_revision=before.revision
            )
            assert changed.revision == 1
            assert all("marker" in a.extensions for a in changed.agents)
            assert await client.thread_extensions("t") == changed
            with pytest.raises(ServiceConflictError):
                await client.update_extensions(
                    "t", scope="tree", disable=["marker"], expected_revision=0
                )
            changed = await client.update_extensions(
                "t", agent_ids=["root"], disable=["marker"], expected_revision=1
            )
            assert changed.revision == 2
            assert ["marker" in a.extensions for a in changed.agents] == [False, True]
            headers = {"Authorization": "Bearer secret"}
            for data in (
                {"scope": "tree", "enable": ["missing"]},
                {"scope": "tree", "enable": ["marker"], "factory": "evil:code"},
                {"scope": "tree", "agent_ids": ["root"], "disable": ["marker"]},
                {"scope": "tree", "enable": ["marker"], "expected_revision": True},
                {"scope": "tree", "enable": "marker"},
            ):
                response = await http.patch(
                    "/v1/threads/t/extensions", headers=headers, json=data
                )
                assert response.status_code == 422, response.text
            assert (await session.thread_extensions()) == changed
            assert (
                await http.get("/v1/threads/missing/extensions", headers=headers)
            ).status_code == 404
    finally:
        await service.aclose()
        service.store.close()
