"""HTTP observers release presentation history after producing their first frame."""

import gc
import weakref

import pytest

pytest.importorskip("litestar")

from msgflux.core.dotdict import dotdict
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.http import create_service_app
from tests.test_agent_service_http_integration import _agent, _response


@pytest.mark.asyncio
async def test_http_observer_releases_snapshot_messages_before_waiting_for_events(
    monkeypatch,
):
    import msgflux.runtime.service.http.app as app_module

    references = []
    original_encode = app_module.encode_snapshot

    def record_snapshot(snapshot):
        references.append(weakref.ref(snapshot.messages))
        return original_encode(snapshot)

    monkeypatch.setattr(app_module, "encode_snapshot", record_snapshot)
    agent = _agent(lambda **_kwargs: _response("seeded"), name="snapshot-lifetime")
    service = AgentService(store=SQLiteServiceStore())
    service.register("main", lambda _thread: AgentSession(agent))
    response = None
    try:
        thread = await service.open_thread("main")
        await service.prompt(thread.thread_id, "history " * 1000, request_id="seed")
        assert (await service.wait(thread.thread_id, "seed")).status == "completed"
        app = create_service_app(service, token="secret")
        route = next(route for route in app.routes if route.path.endswith("/watch"))
        handler = route.route_handler_map["GET"][0]
        response = await handler.fn(dotdict(headers={}), thread.thread_id)
        first = await anext(response.iterator)
        assert b"event: snapshot" in first and b"history " in first
        gc.collect()
        assert len(references) == 1
        assert references[0]() is None
        # The subscription remains open after its initial history was released.
        assert len(service._watchers) == 1
    finally:
        if response is not None:
            await response.background()
        await service.aclose()
        service.store.close()
