"""Tests for process-local execution event observation."""

import asyncio
import threading

import pytest

from msgflux.runtime.event_hub import BackgroundTaskSnapshot, EventHub
from msgflux.runtime.events import EventType, ExecutionEvent
from msgflux.exceptions import EventBufferOverflowError


@pytest.mark.asyncio
@pytest.mark.parametrize("parts", [("olá ", "🌍", "!"), (b"hello ", b"world", b"!")])
@pytest.mark.parametrize(
    "kind,field,chunks_field",
    [
        ("message.delta", "streaming_message", "message_chunks"),
        ("reasoning.delta", "reasoning", "reasoning_chunks"),
        ("reasoning_summary.delta", "reasoning_summary", "reasoning_summary_chunks"),
    ],
)
async def test_reconnect_consolidates_text_without_mutating_old_snapshots(
    parts, kind, field, chunks_field
):
    hub = EventHub()
    for part in parts[:2]:
        hub.publish("thread", make_event(kind, {"delta": part}))
    async with hub.watch("thread") as first:
        initial = getattr(first.snapshot.active_run, field)
        assert initial == parts[0] + parts[1]
    run = next(iter(hub._threads["thread"].runs.values()))
    assert getattr(run, chunks_field) == [initial]
    async with hub.watch("thread") as second:
        assert getattr(second.snapshot.active_run, field) is initial
    hub.publish("thread", make_event(kind, {"delta": parts[2]}))
    async with hub.watch("thread") as third:
        assert getattr(third.snapshot.active_run, field) == initial + parts[2]
    assert getattr(first.snapshot.active_run, field) == parts[0] + parts[1]
    hub.publish("thread", make_event("run.end"))
    assert not hub._threads


@pytest.mark.asyncio
async def test_mixed_projection_payloads_keep_existing_last_value_semantics():
    hub = EventHub()
    for part in ("text", b"bytes", {"structured": True}):
        hub.publish("thread", make_event("message.delta", {"delta": part}))
    async with hub.watch("thread") as watcher:
        assert watcher.snapshot.streaming_message == {"structured": True}


def make_event(
    event_type: str,
    data=None,
    *,
    run_id: str = "run_1",
    source_path=("agent:root",),
):
    return ExecutionEvent(
        type=event_type,
        timestamp="2026-08-27T00:00:00+00:00",
        data=data or {},
        run_id=run_id,
        source_path=source_path,
    )


@pytest.mark.asyncio
async def test_watch_snapshot_combines_live_projection_and_future_events():
    hub = EventHub()
    thread_id = "thread_1"
    hub.publish(
        thread_id,
        make_event(
            EventType.RUN_START,
            {"namespace": "root"},
        ),
    )
    hub.publish(thread_id, make_event(EventType.REASONING_DELTA, {"delta": "raw"}))
    hub.publish(
        thread_id,
        make_event(EventType.REASONING_SUMMARY_DELTA, {"delta": "summary"}),
    )
    hub.publish(thread_id, make_event(EventType.MESSAGE_START))
    hub.publish(thread_id, make_event(EventType.MESSAGE_DELTA, {"delta": "hel"}))
    hub.publish(
        thread_id,
        make_event(
            EventType.TOOL_START,
            {
                "tool_call_id": "call_1",
                "tool_name": "lookup",
                "arguments": {"query": "sku-1"},
            },
        ),
    )
    hub.publish(
        thread_id,
        make_event(
            EventType.TASK_START,
            {"task_id": "task_1", "tool_name": "research", "status": "queued"},
        ),
    )

    async with hub.watch(
        thread_id,
        namespace="root",
        load_messages=lambda: ["durable"],
    ) as watcher:
        snapshot = watcher.snapshot
        assert snapshot.messages == ["durable"]
        assert snapshot.active_run.streaming_message == "hel"
        assert snapshot.active_run.reasoning == "raw"
        assert snapshot.active_run.reasoning_summary == "summary"
        assert snapshot.running_tools[0].tool_name == "lookup"
        assert snapshot.background_tasks == (
            BackgroundTaskSnapshot(
                task_id="task_1",
                tool_name="research",
                status="queued",
            ),
        )

        expected = make_event(EventType.MESSAGE_DELTA, {"delta": "lo"})
        hub.publish(thread_id, expected)
        assert await asyncio.wait_for(watcher.__anext__(), timeout=1) == expected


@pytest.mark.asyncio
async def test_watch_closes_snapshot_subscription_race():
    hub = EventHub()
    thread_id = "thread_race"
    started = threading.Event()
    publisher = None
    expected = make_event(EventType.RUN_START, {"namespace": "root"})

    def load_messages():
        nonlocal publisher

        def publish():
            started.set()
            hub.publish(thread_id, expected)

        publisher = threading.Thread(target=publish)
        publisher.start()
        assert started.wait(timeout=1)
        return ["snapshot"]

    async with hub.watch(thread_id, load_messages=load_messages) as watcher:
        assert watcher.snapshot.messages == ["snapshot"]
        assert await asyncio.wait_for(watcher.__anext__(), timeout=1) == expected

    assert publisher is not None
    publisher.join(timeout=1)


def test_hub_keeps_only_active_projection_without_event_log():
    hub = EventHub()
    thread_id = "thread_cleanup"
    hub.publish(thread_id, make_event(EventType.RUN_START))
    assert thread_id in hub._threads

    hub.publish(thread_id, make_event(EventType.RUN_END))

    assert thread_id not in hub._threads
    assert hub._watchers == {}


@pytest.mark.asyncio
async def test_parent_terminal_event_clears_nested_projection_and_keeps_watcher():
    hub = EventHub()
    thread_id = "nested_cleanup"
    child_path = ("agent:root", "tool:lookup")
    hub.publish(
        thread_id,
        make_event(EventType.RUN_START, source_path=("agent:root",)),
    )
    hub.publish(
        thread_id,
        make_event(
            EventType.MESSAGE_DELTA,
            {"delta": "nested output"},
            source_path=child_path,
        ),
    )

    async with hub.watch(thread_id) as watcher:
        assert len(watcher.snapshot.active_runs) == 2
        terminal = make_event(EventType.RUN_END, source_path=("agent:root",))
        hub.publish(thread_id, terminal)

        assert await watcher.__anext__() == terminal
        assert not hub._threads[thread_id].active
        assert thread_id in hub._watchers

    assert thread_id not in hub._threads


def test_tool_lifecycle_tracks_tool_without_creating_an_extra_run():
    hub = EventHub()
    thread_id = "tool_projection"
    source_path = ("agent:root", "tool:lookup")
    hub.publish(
        thread_id,
        make_event(
            EventType.TOOL_START,
            {
                "tool_call_id": "call-1",
                "tool_name": "lookup",
                "arguments": {"query": "status"},
            },
            source_path=source_path,
        ),
    )

    state = hub._threads[thread_id]
    assert state.runs == {}
    assert len(state.tools) == 1

    hub.publish(
        thread_id,
        make_event(
            EventType.TOOL_END,
            {"tool_call_id": "call-1"},
            source_path=source_path,
        ),
    )
    assert not hub._threads


@pytest.mark.asyncio
async def test_watch_is_isolated_by_thread_id():
    hub = EventHub()
    expected = make_event(EventType.RUN_START)

    async with hub.watch("thread_a") as watcher:
        hub.publish("thread_b", make_event(EventType.RUN_START, run_id="run_b"))
        hub.publish("thread_a", expected)
        assert await asyncio.wait_for(watcher.__anext__(), timeout=1) == expected


@pytest.mark.asyncio
async def test_overflow_detaches_only_slow_watcher_and_allows_reconnect():
    hub = EventHub()
    events = [make_event(EventType.MESSAGE_DELTA, {"delta": str(i)}) for i in range(4)]
    async with (
        hub.watch("t", event_buffer_limit=2) as slow,
        hub.watch("t") as fast,
    ):
        for event in events:
            hub.publish("t", event)
        assert slow not in hub._watchers["t"]
        assert [await anext(fast) for _ in events] == events
        with pytest.raises(EventBufferOverflowError):
            await anext(slow)
        with pytest.raises(StopAsyncIteration):
            await anext(slow)
        async with hub.watch("t", event_buffer_limit=2) as reconnected:
            assert reconnected.snapshot.streaming_message == "0123"
            hub.publish("t", make_event(EventType.RUN_END))
            assert (await anext(reconnected)).type == EventType.RUN_END


@pytest.mark.asyncio
async def test_full_watcher_closes_without_queuefull_or_hanging():
    hub = EventHub()
    async with hub.watch("t", event_buffer_limit=1) as watcher:
        hub.publish("t", make_event(EventType.RUN_START))
        await watcher.aclose()
        assert (await anext(watcher)).type == EventType.RUN_START
        with pytest.raises(StopAsyncIteration):
            await anext(watcher)
        with pytest.raises(StopAsyncIteration):
            await anext(watcher)
