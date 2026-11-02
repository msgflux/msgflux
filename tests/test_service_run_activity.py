"""Activity observes existing work without hydrating it or retaining its data."""

import asyncio
import gc
import json
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest

from msgflux.runtime.events import ExecutionEvent, EventType
from msgflux.runtime.service import (
    AdmissionReceipt,
    AgentService,
    AgentSession,
    SQLiteServiceStore,
)
from msgflux.runtime.service.activity import RunActivityTracker
from msgflux.runtime.service.local.cli import _show_run_result
from msgflux.runtime.service.records import ServiceThread


def _receipt():
    return AdmissionReceipt("thread", "request", "run", "running")


def test_activity_bounds_labels_and_drops_original_payload():
    class Payload:
        pass

    payload = Payload()
    ref = weakref.ref(payload)
    tracker = RunActivityTracker()
    event = ExecutionEvent(
        EventType.TOOL_END,
        "timestamp",
        {"result": payload, "arguments": "private", "tool_name": "tool" * 100},
        source_path=tuple("source" * 100 for _ in range(50)),
    )
    tracker.observe(event)
    del payload, event
    gc.collect()
    assert ref() is None
    for _ in range(20):
        tracker.observe(ExecutionEvent(EventType.MODEL_REQUEST, "stamp"))
    before = tracker.snapshot(_receipt(), interrupted=False)
    assert len(before.recent_events) == 8
    tracker.observe(
        ExecutionEvent(EventType.MESSAGE_DELTA, "latest", {"delta": "secret"})
    )
    after = tracker.snapshot(_receipt(), interrupted=False)
    assert after.recent_events == before.recent_events
    assert after.last_event_type == EventType.MESSAGE_DELTA
    assert after.last_event_at == "latest"
    assert "secret" not in msgspec.json.encode(after).decode()
    assert "private" not in msgspec.json.encode(after).decode()


def test_activity_label_lengths_are_bounded():
    tracker = RunActivityTracker()
    tracker.observe(
        ExecutionEvent(
            "x" * 1000,
            "t" * 1000,
            {"tool_name": "n" * 1000},
            source_path=("p" * 1000,) * 30,
        )
    )
    item = tracker.snapshot(_receipt(), interrupted=False).recent_events[0]
    assert len(item.event_type) == len(item.timestamp) == 64
    assert len(item.tool_name) == 128
    assert len(item.source_path) == 8
    assert all(len(part) == 128 for part in item.source_path)


def test_activity_interrupt_age_preserves_first_request():
    tracker = RunActivityTracker()
    with patch(
        "msgflux.runtime.service.activity.time.monotonic", side_effect=[10, 20, 25, 30]
    ):
        tracker.observe(ExecutionEvent(EventType.MODEL_REQUEST, "request-time"))
        tracker.request_interrupt()
        first = tracker.snapshot(_receipt(), interrupted=True)
        tracker.request_interrupt()
        second = tracker.snapshot(_receipt(), interrupted=True)
    assert first.seconds_since_event == 15
    assert first.seconds_since_interrupt == 5
    assert second.seconds_since_event == 20
    assert second.seconds_since_interrupt == 10
    assert first.interrupt_requested_at == second.interrupt_requested_at
    assert second.interrupt_requested


@pytest.mark.asyncio
async def test_activity_uses_admission_without_instantiating_factory_or_checkpoint():
    store = SQLiteServiceStore()
    store.bind_thread(ServiceThread(thread_id="thread", agent_id="main"))
    record = store.admit("thread", "request", "private prompt", "main")
    service = AgentService(store=store)

    def forbidden(_thread):
        pytest.fail("Activity instantiated the Agent")

    service.register("main", forbidden)
    try:
        activity = await service.run_activity("thread", record.receipt.run_id)
        assert activity.status == "accepted" and not activity.local_worker
        assert activity.recent_events == () and activity.last_event_type is None
        assert activity.interrupt_requested is None
        assert "private" not in msgspec.json.encode(activity).decode()
        assert service._sessions == {}
        with pytest.raises(KeyError):
            await service.run_activity("thread", "missing")
        with pytest.raises(KeyError):
            await service.run_activity("unknown", record.receipt.run_id)
    finally:
        await service.aclose()
        store.close()


def test_activity_cli_plain_and_json(capsys):
    tracker = RunActivityTracker()
    tracker.observe(ExecutionEvent(EventType.MODEL_REQUEST, "stamp"))
    tracker.request_interrupt()
    activity = tracker.snapshot(_receipt(), interrupted=True)
    _show_run_result(SimpleNamespace(action="activity", json=False), activity)
    output = capsys.readouterr().out
    assert "Last observed event: model.request" in output
    assert "settlement is pending" in output
    _show_run_result(SimpleNamespace(action="activity", json=True), activity)
    assert json.loads(capsys.readouterr().out)["last_event_type"] == "model.request"


@pytest.mark.asyncio
async def test_http_activity_observes_pending_cancellation_until_cleanup_finishes():
    pytest.importorskip("litestar")
    from contextlib import aclosing

    from msgflux.runtime.service.http import AgentServiceClient
    from tests.test_agent_service_http_integration import (
        TOKEN,
        _agent,
        _response,
        _server,
    )

    entered, cleanup_entered, release = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )

    async def answer(**_kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_entered.set()
            await release.wait()
            raise
        return _response("unreachable")

    agent = _agent(answer)
    try:
        async with _server(lambda _thread: AgentSession(agent)) as (service, url):
            async with aclosing(AgentServiceClient(url, token=TOKEN)) as client:
                thread = await client.open_thread("main")
                receipt = await client.prompt(
                    thread.thread_id, "private prompt", request_id="run"
                )
                await asyncio.wait_for(entered.wait(), 2)

                async def requested():
                    while True:
                        result = await client.run_activity(
                            thread.thread_id, receipt.run_id
                        )
                        if result.last_event_type == EventType.MODEL_REQUEST:
                            return result
                        await asyncio.sleep(0)

                active = await asyncio.wait_for(requested(), 2)
                assert active.local_worker and active.status == "running"
                assert active.seconds_since_event >= 0
                # Polling must not enter the ordinary session/checkpoint path.
                with patch.object(
                    service, "acquire_session", side_effect=AssertionError("hydration")
                ):
                    assert (
                        await client.run_activity(thread.thread_id, receipt.run_id)
                    ).local_worker
                await client.interrupt(thread.thread_id, receipt.run_id)
                await asyncio.wait_for(cleanup_entered.wait(), 2)
                cancelling = await client.run_activity(thread.thread_id, receipt.run_id)
                assert cancelling.interrupt_requested and cancelling.local_worker
                assert cancelling.interrupt_requested_at is not None
                assert cancelling.status == "running"
                assert "private prompt" not in msgspec.json.encode(cancelling).decode()
                release.set()
                settled = await asyncio.wait_for(
                    service.wait(thread.thread_id, "run"), 2
                )
                assert settled.status == "interrupted"
                final = await client.run_activity(thread.thread_id, receipt.run_id)
                assert final.status == "interrupted" and not final.local_worker
                assert final.recent_events == () and final.last_event_type is None
                assert final.interrupt_requested is None
    finally:
        release.set()
        await agent.aclose()
