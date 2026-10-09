import asyncio
import gc
import weakref
from types import SimpleNamespace

import pytest

from msgflux.runtime.service.records import (
    ServiceBusyError,
    ServiceRecoveryRequiredError,
)
from msgflux.runtime.service.session_cache import _SessionCache


def _thread(thread_id: str) -> SimpleNamespace:
    return SimpleNamespace(thread_id=thread_id)


@pytest.mark.asyncio
async def test_acquire_single_flight_and_cancelled_waiter_does_not_cancel_load():
    entered = asyncio.Event()
    finish = asyncio.Event()
    calls = 0
    session = SimpleNamespace(on_close=lambda: None)

    async def load(_thread):
        nonlocal calls
        calls += 1
        entered.set()
        await finish.wait()
        return session

    cache = _SessionCache(load, lambda *_: None)
    first = asyncio.create_task(cache.acquire(_thread("one")))
    await entered.wait()
    second = asyncio.create_task(cache.acquire(_thread("one")))
    await asyncio.sleep(0)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second

    finish.set()
    lease = await first
    assert lease.session is session
    assert calls == 1
    await lease.aclose()
    assert await cache.release("one") is True


@pytest.mark.asyncio
async def test_cancelled_load_waiters_do_not_leave_unretrieved_failure():
    entered = asyncio.Event()
    fail = asyncio.Event()
    observed = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: observed.append(context))

    async def load(_thread):
        entered.set()
        await fail.wait()
        raise RuntimeError("factory failed")

    cache = _SessionCache(load, lambda *_: None)
    try:
        waiters = [asyncio.create_task(cache.acquire("lost")) for _ in range(2)]
        await entered.wait()
        load_task = cache._loads["lost"]
        settled = asyncio.Event()
        load_task.add_done_callback(lambda _task: settled.set())
        for waiter in waiters:
            waiter.cancel()
        for waiter in waiters:
            with pytest.raises(asyncio.CancelledError):
                await waiter

        fail.set()
        await settled.wait()
        await asyncio.sleep(0)
        assert "lost" not in cache._loads
        # Python 3.14 reports shielded failures even when another callback has
        # retrieved the exception. That diagnostic is expected; unobserved or
        # unrelated task failures must still fail this regression.
        assert all(
            context.get("message") == "RuntimeError exception in shielded future"
            and context.get("future") is load_task
            and context.get("exception") is load_task.exception()
            for context in observed
        )
    finally:
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_distinct_threads_load_concurrently():
    entered = {key: asyncio.Event() for key in ("one", "two")}
    finish = asyncio.Event()
    calls = 0

    async def load(thread):
        nonlocal calls
        calls += 1
        entered[thread.thread_id].set()
        await finish.wait()
        return SimpleNamespace(on_close=lambda: None)

    cache = _SessionCache(load, lambda *_: None)
    tasks = [asyncio.create_task(cache.acquire(_thread(key))) for key in entered]
    await asyncio.wait_for(
        asyncio.gather(*(event.wait() for event in entered.values())), 1
    )
    assert calls == 2
    finish.set()
    leases = await asyncio.gather(*tasks)
    await asyncio.gather(*(lease.aclose() for lease in leases))
    await asyncio.gather(*(cache.release(key) for key in entered))


@pytest.mark.asyncio
async def test_release_rejects_pins_and_acquire_waits_for_close_generation():
    close_entered = asyncio.Event()
    close_finish = asyncio.Event()
    loads = 0
    closes = 0

    class Session:
        async def on_close(self):
            nonlocal closes
            closes += 1
            close_entered.set()
            await close_finish.wait()

    async def load(_thread):
        nonlocal loads
        loads += 1
        return Session()

    cache = _SessionCache(load, lambda *_: None)
    lease = await cache.acquire("thread")
    with pytest.raises(ServiceBusyError):
        await cache.release("thread")
    await lease.aclose()

    release = asyncio.create_task(cache.release("thread"))
    await close_entered.wait()
    acquire = asyncio.create_task(cache.acquire("thread"))
    await asyncio.sleep(0)
    assert loads == 1
    close_finish.set()
    assert await release is True
    next_lease = await acquire
    assert loads == 2
    assert closes == 1
    await next_lease.aclose()
    # Keep the second session's close callback from blocking.
    close_finish.set()
    await cache.release("thread")


@pytest.mark.asyncio
async def test_close_failure_quarantines_generation_and_blocks_reload():
    calls = 0
    failure = OSError("close failed")

    class Session:
        async def on_close(self):
            raise failure

    async def load(_thread):
        nonlocal calls
        calls += 1
        return Session()

    cache = _SessionCache(load, lambda *_: None)
    lease = await cache.acquire("thread")
    session = lease.session
    await lease.aclose()

    with pytest.raises(ServiceRecoveryRequiredError) as raised:
        await cache.release("thread")
    assert raised.value.__cause__ is failure
    assert "close failed" in str(raised.value)
    assert cache.sessions["thread"] is session
    with pytest.raises(ServiceRecoveryRequiredError) as raised:
        await cache.acquire("thread")
    assert raised.value.__cause__ is failure
    assert calls == 1

    errors = await cache.close_all()
    assert len(errors) == 1
    assert errors[0].__cause__ is failure
    assert cache.sessions["thread"] is session


@pytest.mark.asyncio
async def test_idle_reason_rejects_release_until_host_state_is_idle():
    busy = True
    cache = _SessionCache(
        lambda _thread: SimpleNamespace(on_close=lambda: None),
        lambda *_: "background task is active" if busy else None,
    )
    lease = await cache.acquire("thread")
    await lease.aclose()
    with pytest.raises(ServiceBusyError, match="background task"):
        await cache.release("thread")
    busy = False
    assert await cache.release("thread") is True


@pytest.mark.asyncio
async def test_loader_reentry_for_same_thread_fails_without_deadlock():
    cache = None

    async def load(thread):
        await cache.acquire(thread)

    cache = _SessionCache(load, lambda *_: None)
    with pytest.raises(RuntimeError, match="re-entered"):
        await asyncio.wait_for(cache.acquire("thread"), timeout=1)


@pytest.mark.asyncio
async def test_cancelled_close_all_waiter_does_not_cancel_shutdown():
    close_entered = asyncio.Event()
    close_finish = asyncio.Event()

    class Session:
        async def on_close(self):
            close_entered.set()
            await close_finish.wait()

    cache = _SessionCache(
        lambda _thread: Session(),
        lambda *_: None,
    )
    lease = await cache.acquire("thread")
    await lease.aclose()
    shutdown = asyncio.create_task(cache.close_all())
    await close_entered.wait()
    shutdown.cancel()
    with pytest.raises(asyncio.CancelledError):
        await shutdown
    close_finish.set()
    assert await cache.close_all() == []
    assert dict(cache.sessions) == {}


@pytest.mark.asyncio
async def test_cancelled_lease_close_still_releases_pin():
    cache = _SessionCache(
        lambda _thread: SimpleNamespace(on_close=lambda: None),
        lambda *_: None,
    )
    lease = await cache.acquire("thread")
    await cache._lock.acquire()
    close_lease = asyncio.create_task(lease.aclose())
    await asyncio.sleep(0)
    close_lease.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close_lease
    cache._lock.release()

    await lease.aclose()
    assert await cache.release("thread") is True


@pytest.mark.asyncio
async def test_closed_lease_drops_session_and_cache_references():
    class Session:
        def on_close(self):
            return None

    session = Session()
    cache = _SessionCache(lambda _thread, value=session: value, lambda *_: None)
    lease = await cache.acquire("thread")
    session_ref = weakref.ref(session)
    cache_ref = weakref.ref(cache)

    await lease.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        _ = lease.session
    await lease.aclose()  # Closing a lease is idempotent.
    assert await cache.release("thread") is True

    del session
    del cache
    gc.collect()
    assert session_ref() is None
    assert cache_ref() is None


@pytest.mark.asyncio
async def test_unloaded_thread_generations_do_not_accumulate_per_thread_state():
    cache = _SessionCache(
        lambda _thread: SimpleNamespace(on_close=lambda: None),
        lambda *_: None,
    )
    for index in range(25):
        key = f"thread-{index}"
        lease = await cache.acquire(key)
        await lease.aclose()
        assert await cache.release(key) is True

    assert cache._generation == 25
    assert cache._entries == {}
    assert cache._loads == {}
    assert not hasattr(cache, "_generations")


@pytest.mark.asyncio
async def test_close_all_disposes_late_load_without_publishing_it():
    entered = asyncio.Event()
    finish = asyncio.Event()
    closed = 0

    class Session:
        async def on_close(self):
            nonlocal closed
            closed += 1

    async def load(_thread):
        entered.set()
        await finish.wait()
        return Session()

    cache = _SessionCache(load, lambda *_: None)
    acquire = asyncio.create_task(cache.acquire("thread"))
    await entered.wait()
    shutdown = asyncio.create_task(cache.close_all())
    await asyncio.sleep(0)
    finish.set()

    with pytest.raises(ServiceRecoveryRequiredError):
        await acquire
    assert await shutdown == []
    assert closed == 1
    assert dict(cache.sessions) == {}
