"""Tests for process-local execution event observation."""

import asyncio
import threading

import pytest

from msgflux.runtime.event_hub import BackgroundTaskSnapshot, EventHub
from msgflux.runtime.events import EventType, ExecutionEvent


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
async def test_watch_is_isolated_by_thread_id():
    hub = EventHub()
    expected = make_event(EventType.RUN_START)

    async with hub.watch("thread_a") as watcher:
        hub.publish("thread_b", make_event(EventType.RUN_START, run_id="run_b"))
        hub.publish("thread_a", expected)
        assert await asyncio.wait_for(watcher.__anext__(), timeout=1) == expected
