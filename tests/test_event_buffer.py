"""Bounds apply before loop callbacks, including worker-thread publishers."""

import asyncio
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from msgflux.exceptions import EventBufferOverflowError
from msgflux.runtime.event_buffer import _EventBuffer


@pytest.mark.asyncio
async def test_overflow_releases_payloads_before_consumer_observes_error():
    class Payload:
        def __init__(self):
            self.content = bytearray(1024 * 1024)

    buffer = _EventBuffer(1)
    payload = Payload()
    reference = weakref.ref(payload)
    assert buffer.put(payload)
    del payload
    assert reference() is not None
    assert not buffer.put("overflow")
    # No event-loop yield or consumer get: a queued wake must not own payloads.
    assert reference() is None
    with pytest.raises(EventBufferOverflowError):
        await buffer.get()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "2"])
async def test_invalid_buffer_limit(limit):
    with pytest.raises(ValueError, match="positive integer"):
        _EventBuffer(limit)


@pytest.mark.asyncio
async def test_thread_burst_is_bounded_before_loop_can_run(monkeypatch):
    loop = asyncio.get_running_loop()
    original = loop.call_soon_threadsafe
    wakes = []

    def schedule(callback, *args, **kwargs):
        wakes.append(callback)
        return original(callback, *args, **kwargs)

    buffer = _EventBuffer(4)
    monkeypatch.setattr(loop, "call_soon_threadsafe", schedule)
    # Joining here intentionally keeps the loop from draining callbacks.
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(buffer.put, range(1000)))
    assert len(buffer._items) == 0
    assert len(wakes) == 1
    buffer.close()
    with pytest.raises(EventBufferOverflowError) as error:
        await buffer.get()
    assert error.value.limit == 4
    assert await buffer.get() is None
    assert buffer.put("late") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 4])
async def test_close_preserves_accepted_order_without_a_sentinel_slot(limit):
    buffer = _EventBuffer(limit)
    for item in range(4):
        assert buffer.put(item)
    buffer.close()
    buffer.close()
    assert [await buffer.get() for _ in range(4)] == list(range(4))
    assert await buffer.get() is None
    assert await buffer.get() is None


@pytest.mark.asyncio
async def test_worker_and_consumer_handshake_has_no_lost_wakeup():
    buffer = _EventBuffer(1)
    for item in range(100):
        consumer = asyncio.create_task(buffer.get())
        await asyncio.to_thread(buffer.put, item)
        assert await asyncio.wait_for(consumer, timeout=1) == item
    buffer.close()


@pytest.mark.asyncio
async def test_closed_loop_publication_drops_references_without_raising(monkeypatch):
    buffer = _EventBuffer(1)

    def closed(*args):
        raise RuntimeError("Event loop is closed")

    monkeypatch.setattr(buffer._loop, "call_soon_threadsafe", closed)
    assert not buffer.put("late")
    assert len(buffer._items) == 0
    assert await buffer.get() is None


@pytest.mark.asyncio
async def test_loop_closed_with_a_wakeup_already_scheduled(monkeypatch):
    buffer = _EventBuffer()
    assert buffer.put("accepted before close")
    assert buffer._wake_pending
    with monkeypatch.context() as patch:
        patch.setattr(buffer._loop, "is_closed", lambda: True)
        assert not buffer.put("late")
        assert len(buffer._items) == 0
    assert await buffer.get() is None


@pytest.mark.asyncio
async def test_thread_publication_racing_close_keeps_only_accepted_prefix():
    with ThreadPoolExecutor(max_workers=1) as pool:
        for _ in range(20):
            buffer = _EventBuffer()
            barrier = threading.Barrier(2)

            def publish(target, start):
                accepted = []
                start.wait(timeout=2)
                for item in range(100):
                    if target.put(item):
                        accepted.append(item)
                return accepted

            producer = pool.submit(publish, buffer, barrier)
            barrier.wait(timeout=2)
            buffer.close()
            accepted = producer.result(timeout=2)
            assert [await buffer.get() for _ in accepted] == accepted
            assert await buffer.get() is None
            assert not buffer.put("after close")
