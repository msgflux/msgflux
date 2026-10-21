"""Watch attachment releases dependencies even when its caller cancels."""

import asyncio
import gc
import weakref
from unittest.mock import Mock

import pytest

from msgflux.nn import Agent
from msgflux.runtime.event_hub import EventHub, get_event_hub
from msgflux.runtime.events import ExecutionEvent
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore


@pytest.mark.asyncio
async def test_cancelled_attach_disposes_observer_while_lease_release_continues():
    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "main",
        lambda _thread: AgentSession(
            Agent(name="watch-cancel", model=Mock(model_type="chat_completion"))
        ),
    )
    thread = await service.open_thread("main")
    release_entered, finish_release, release_done = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    original_unpin = service._cache._unpin

    async def delayed_unpin(key, generation):
        release_entered.set()
        await finish_release.wait()
        await original_unpin(key, generation)
        release_done.set()

    service._cache._unpin = delayed_unpin
    context = service.watch(thread.thread_id)
    attach = asyncio.create_task(context.__aenter__())
    try:
        await asyncio.wait_for(release_entered.wait(), 2)
        assert get_event_hub()._watchers.get(thread.thread_id)
        attach.cancel()
        with pytest.raises(asyncio.CancelledError):
            await attach
        assert not get_event_hub()._watchers.get(thread.thread_id)
        finish_release.set()
        await asyncio.wait_for(release_done.wait(), 2)
        assert await service.release_session(thread.thread_id)
    finally:
        finish_release.set()
        await asyncio.gather(attach, return_exceptions=True)
        await service.aclose()
        service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["attach", "unused_close", "failed_attach"])
async def test_watcher_drops_snapshot_readers_after_attach_or_close(action):
    class Readers:
        def messages(self):
            if action == "failed_attach":
                raise ValueError("snapshot unavailable")
            return ["saved history"]

        def approvals(self):
            return ()

    readers = Readers()
    reference = weakref.ref(readers)
    watcher = EventHub().watch(
        "thread", load_messages=readers.messages, load_approvals=readers.approvals
    )
    del readers
    try:
        if action == "attach":
            await watcher.__aenter__()
            assert watcher.snapshot.messages == ["saved history"]
        elif action == "failed_attach":
            with pytest.raises(ValueError, match="snapshot unavailable"):
                await watcher.__aenter__()
            assert not watcher._hub._watchers
        else:
            await watcher.aclose()
        gc.collect()
        assert reference() is None
    finally:
        await watcher.aclose()


@pytest.mark.asyncio
async def test_shutdown_detaches_own_watchers_and_preserves_queued_events():
    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "main",
        lambda _thread: AgentSession(
            Agent(name="watch-shutdown", model=Mock(model_type="chat_completion"))
        ),
    )
    thread = await service.open_thread("main")
    hub = get_event_hub()
    async with hub.watch(thread.thread_id) as other:
        try:
            async with service.watch(thread.thread_id) as watcher:
                event = ExecutionEvent(type="run.end", timestamp="now", run_id="one")
                hub.publish(thread.thread_id, event)
                await service.aclose()
                assert service._watchers == set()
                assert hub._watchers[thread.thread_id] == {other}
                assert await anext(watcher) is event
                with pytest.raises(StopAsyncIteration):
                    await anext(watcher)
                assert await anext(other) is event
        finally:
            await service.aclose()
            service.store.close()
