import asyncio
import gc
import weakref

import pytest

from msgflux.runtime.service.cache_policy import SessionCachePolicy
from msgflux.runtime.service.records import (
    ServiceBusyError,
    ServiceRecoveryRequiredError,
)
from msgflux.runtime.service.session_cache import _SessionCache


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _Session:
    def __init__(self, name, closed):
        self.name = name
        self.closed = closed

    async def on_close(self):
        self.closed.append(self.name)


def test_policy_defaults_validation_and_immutability():
    policy = SessionCachePolicy()
    assert policy.max_loaded == 64
    assert policy.idle_timeout == 300.0
    with pytest.raises(AttributeError):
        policy.max_loaded = 3

    for max_loaded in (0, -1, True, 1.5):
        with pytest.raises((TypeError, ValueError)):
            SessionCachePolicy(max_loaded=max_loaded)
    for timeout in (-0.1, float("inf"), float("nan"), True):
        with pytest.raises((TypeError, ValueError)):
            SessionCachePolicy(idle_timeout=timeout)
    assert SessionCachePolicy(max_loaded=None, idle_timeout=None) == SessionCachePolicy(
        max_loaded=None, idle_timeout=None
    )
    assert SessionCachePolicy(idle_timeout=0).idle_timeout == 0.0


@pytest.mark.asyncio
async def test_capacity_is_reserved_before_factory_and_busy_slots_reject_load():
    calls = []
    closed = []

    async def load(thread):
        calls.append(thread)
        return _Session(thread, closed)

    cache = _SessionCache(
        load,
        lambda *_: None,
        busy_reason=lambda *_: None,
        policy=SessionCachePolicy(max_loaded=1, idle_timeout=None),
    )
    first = await cache.acquire("first")
    with pytest.raises(ServiceBusyError, match="full"):
        await cache.acquire("second")
    assert calls == ["first"]

    await first.aclose()
    second = await cache.acquire("second")
    assert calls == ["first", "second"]
    assert closed == ["first"]
    await second.aclose()
    await cache.close_all()


@pytest.mark.asyncio
async def test_loading_slot_counts_toward_capacity_without_second_factory():
    entered, finish = asyncio.Event(), asyncio.Event()
    calls = []

    async def load(thread):
        calls.append(thread)
        if thread == "first":
            entered.set()
            await finish.wait()
        return _Session(thread, [])

    cache = _SessionCache(
        load,
        lambda *_: None,
        busy_reason=lambda *_: None,
        policy=SessionCachePolicy(max_loaded=1, idle_timeout=None),
    )
    first_task = asyncio.create_task(cache.acquire("first"))
    await entered.wait()
    with pytest.raises(ServiceBusyError, match="full"):
        await cache.acquire("second")
    assert calls == ["first"]
    finish.set()
    lease = await first_task
    await lease.aclose()
    await cache.close_all()


@pytest.mark.asyncio
async def test_capacity_pressure_evicts_lru_and_skips_busy_candidate():
    clock = _Clock()
    closed = []
    busy = {"oldest": True}
    calls = []

    async def load(thread):
        calls.append(thread)
        return _Session(thread, closed)

    cache = _SessionCache(
        load,
        lambda *_: None,
        busy_reason=lambda thread, _session: (
            "active work" if busy.get(thread) else None
        ),
        clock=clock,
        policy=SessionCachePolicy(max_loaded=2, idle_timeout=None),
    )
    leases = [await cache.acquire("oldest"), await cache.acquire("next")]
    for lease in leases:
        await lease.aclose()
    clock.now = 1.0
    third = await cache.acquire("third")
    assert closed == ["next"]
    assert calls == ["oldest", "next", "third"]
    assert set(cache.sessions) == {"oldest", "third"}
    await third.aclose()
    await cache.close_all()


@pytest.mark.asyncio
async def test_quarantine_consumes_slot_but_pressure_can_close_another_entry():
    calls = []
    closed = []
    failure = OSError("cannot close bad")

    class Session(_Session):
        async def on_close(self):
            if self.name == "bad":
                closed.append(self.name)
                raise failure
            await super().on_close()

    async def load(thread):
        calls.append(thread)
        return Session(thread, closed)

    cache = _SessionCache(
        load,
        lambda *_: None,
        busy_reason=lambda *_: None,
        policy=SessionCachePolicy(max_loaded=2, idle_timeout=None),
    )
    bad, good = await cache.acquire("bad"), await cache.acquire("good")
    await bad.aclose()
    await good.aclose()
    with pytest.raises(ServiceRecoveryRequiredError) as raised:
        await cache.release("bad")
    assert raised.value.__cause__ is failure

    third = await cache.acquire("third")
    assert calls == ["bad", "good", "third"]
    assert set(cache.sessions) == {"bad", "third"}
    assert closed == ["bad", "good"]
    await third.aclose()
    await cache.close_all()


@pytest.mark.asyncio
async def test_idle_timeout_zero_preserves_pending_load_handoff_and_cancellation():
    entered, finish = asyncio.Event(), asyncio.Event()
    closed = []

    async def load(thread):
        entered.set()
        await finish.wait()
        return _Session(thread, closed)

    cache = _SessionCache(
        load,
        lambda *_: None,
        busy_reason=lambda *_: None,
        policy=SessionCachePolicy(max_loaded=1, idle_timeout=None),
    )
    first = asyncio.create_task(cache.acquire("thread"))
    await entered.wait()
    cancelled = asyncio.create_task(cache.acquire("thread"))
    await asyncio.sleep(0)
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled

    finish.set()
    lease = await first
    assert lease.session.name == "thread"
    cache._policy = SessionCachePolicy(max_loaded=1, idle_timeout=0)
    await cache._sweep_once()
    assert "thread" in cache.sessions
    await lease.aclose()
    await cache._sweep_once()
    assert "thread" not in cache.sessions
    assert closed == ["thread"]
    await cache.close_all()


@pytest.mark.asyncio
async def test_repeated_cancellation_does_not_leak_load_waiter_reservation():
    entered, finish = asyncio.Event(), asyncio.Event()

    async def load(_thread):
        entered.set()
        await finish.wait()
        return _Session("thread", [])

    cache = _SessionCache(
        load,
        lambda *_: None,
        policy=SessionCachePolicy(max_loaded=1, idle_timeout=None),
    )
    first = asyncio.create_task(cache.acquire("thread"))
    await entered.wait()
    second = asyncio.create_task(cache.acquire("thread"))
    while cache._load_waiters.get("thread") != 2:
        await asyncio.sleep(0)

    await cache._lock.acquire()
    second.cancel()
    await asyncio.sleep(0)
    second.cancel()
    cache._lock.release()
    with pytest.raises(asyncio.CancelledError):
        await second

    assert cache._load_waiters["thread"] == 1
    finish.set()
    lease = await first
    await lease.aclose()
    await cache.close_all()


@pytest.mark.asyncio
async def test_fake_clock_idle_timeout_and_activity_finished_reset():
    clock = _Clock()
    closed = []
    busy = {"thread": True}
    cache = _SessionCache(
        lambda thread: _Session(thread, closed),
        lambda *_: None,
        busy_reason=lambda thread, _session: "worker active" if busy[thread] else None,
        clock=clock,
        policy=SessionCachePolicy(max_loaded=None, idle_timeout=None),
    )
    lease = await cache.acquire("thread")
    await lease.aclose()
    cache._policy = SessionCachePolicy(max_loaded=None, idle_timeout=10)
    clock.now = 10
    await cache._sweep_once()
    assert "thread" in cache.sessions

    busy["thread"] = False
    clock.now = 12
    cache.activity_finished("thread")
    await cache._sweep_once()
    assert "thread" in cache.sessions
    clock.now = 22
    await cache._sweep_once()
    assert "thread" not in cache.sessions
    assert closed == ["thread"]
    await cache.close_all()


@pytest.mark.asyncio
async def test_idle_timeout_starts_after_busy_work_is_confirmed_finished():
    clock = _Clock()
    closed = []
    fenced = []
    busy = {"thread": True}
    cache = _SessionCache(
        lambda thread: _Session(thread, closed),
        lambda thread, _session: fenced.append(thread),
        busy_reason=lambda thread, _session: "worker active" if busy[thread] else None,
        clock=clock,
        policy=SessionCachePolicy(max_loaded=None, idle_timeout=None),
    )
    lease = await cache.acquire("thread")
    await lease.aclose()
    cache._policy = SessionCachePolicy(max_loaded=None, idle_timeout=10)

    await cache._sweep_once()  # The worker is still active, so no idle clock yet.
    clock.now = 4
    busy["thread"] = False
    clock.now = 9
    await cache._sweep_once()  # The next probe is scheduled for t=10.
    assert "thread" in cache.sessions
    assert fenced == []

    clock.now = 10
    await cache._sweep_once()  # Quiescence is confirmed; the TTL begins now.
    assert fenced == []
    clock.now = 19
    await cache._sweep_once()
    assert "thread" in cache.sessions
    clock.now = 20
    await cache._sweep_once()
    assert "thread" not in cache.sessions
    assert closed == ["thread"]
    assert fenced == ["thread"]
    await cache.close_all()


@pytest.mark.asyncio
async def test_cleaner_is_lazy_and_shutdown_stops_it_before_closing_sessions():
    closed = []
    cache = _SessionCache(
        lambda thread: _Session(thread, closed),
        lambda *_: None,
        busy_reason=lambda *_: None,
        policy=SessionCachePolicy(max_loaded=None, idle_timeout=60),
    )
    assert cache._cleaner_task is None
    lease = await cache.acquire("thread")
    await lease.aclose()
    cleaner = cache._cleaner_task
    assert cleaner is not None and not cleaner.done()

    assert await cache.close_all() == []
    assert cleaner.done()
    assert closed == ["thread"]


@pytest.mark.asyncio
async def test_shutdown_waits_for_late_factory_cleanup_and_reports_quarantine():
    factory_entered = asyncio.Event()
    factory_release = asyncio.Event()
    close_entered = asyncio.Event()
    close_release = asyncio.Event()
    failure = OSError("late session cleanup failed")

    class Session:
        async def on_close(self):
            close_entered.set()
            await close_release.wait()
            raise failure

    async def load(_thread):
        factory_entered.set()
        await factory_release.wait()
        return Session()

    cache = _SessionCache(load, lambda *_: None)
    acquire = asyncio.create_task(cache.acquire("late"))
    await factory_entered.wait()
    shutdown = asyncio.create_task(cache.close_all())
    while not cache._closed:
        await asyncio.sleep(0)

    factory_release.set()
    await close_entered.wait()
    assert "late" in cache._loads
    assert not shutdown.done()

    close_release.set()
    with pytest.raises(ServiceRecoveryRequiredError):
        await acquire
    errors = await shutdown
    assert len(errors) == 1
    assert errors[0].__cause__ is failure
    assert "late" in cache.sessions


@pytest.mark.asyncio
async def test_live_session_is_not_retained_by_idle_cleaner_after_release():
    class LiveSession:
        async def on_close(self):
            pass

    sessions = []

    def load(_thread):
        session = LiveSession()
        sessions.append(session)
        return session

    cache = _SessionCache(
        load,
        lambda *_: None,
        busy_reason=lambda *_: None,
        policy=SessionCachePolicy(max_loaded=None, idle_timeout=60),
    )
    lease = await cache.acquire("thread")
    session_ref = weakref.ref(lease.session)
    await lease.aclose()
    await asyncio.sleep(0)  # Let the cleaner start its independent idle probe.
    assert await cache.release("thread") is True
    sessions.clear()
    del lease
    gc.collect()
    assert session_ref() is None
    assert cache._cleaner_task is not None and not cache._cleaner_task.done()
    await cache.close_all()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"),
    reason="asyncio.eager_task_factory requires Python 3.12+",
)
async def test_eager_factory_preserves_load_reservation_before_running_factory():
    observed = []
    cache = None

    async def load(thread):
        observed.append(
            (
                cache._loads[thread] is asyncio.current_task(),
                cache._lock.locked(),
            )
        )
        return _Session(thread, [])

    cache = _SessionCache(load, lambda *_: None)
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        lease = await cache.acquire("thread")
        assert observed == [(True, False)]
        await lease.aclose()
        await cache.release("thread")
    finally:
        loop.set_task_factory(previous_factory)
        await cache.close_all()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not hasattr(asyncio, "eager_task_factory"),
    reason="asyncio.eager_task_factory requires Python 3.12+",
)
async def test_eager_factory_defers_probe_and_close_gates_until_reserved():
    observed_probes = []
    observed_fences = []
    closed = []
    cache = None

    def busy_reason(thread, _session):
        entry = cache._entries[thread]
        observed_probes.append(
            (
                entry.close_task is asyncio.current_task(),
                entry.probing,
                cache._lock.locked(),
            )
        )
        return None

    def fence_reason(thread, _session):
        entry = cache._entries[thread]
        observed_fences.append(
            (entry.close_task is asyncio.current_task(), cache._lock.locked())
        )
        return None

    cache = _SessionCache(
        lambda thread: _Session(thread, closed),
        fence_reason,
        busy_reason=busy_reason,
        policy=SessionCachePolicy(max_loaded=2, idle_timeout=None),
    )
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        lease = await cache.acquire("thread")
        await lease.aclose()
        cache._policy = SessionCachePolicy(max_loaded=2, idle_timeout=0)
        await cache._sweep_once()
        assert observed_probes[0] == (True, True, False)
        assert observed_fences == [(True, False)]
        assert closed == ["thread"]
    finally:
        loop.set_task_factory(previous_factory)
        await cache.close_all()
