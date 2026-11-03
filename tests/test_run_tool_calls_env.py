"""Server settings own retention/defaults independently of observing clients."""

import asyncio
import json
import os
import subprocess
import sys

import httpx2
import pytest

from msgflux.envs import envs
from msgflux.runtime.events import ExecutionEvent, EventType, emit_event
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.runtime.service.activity import RunActivityTracker, _ActivitySettings
from msgflux.runtime.service.http.client import AgentServiceClient
from tests.test_agent_service_http_integration import _agent, _response


def configure(monkeypatch, *, capacity=50, limit=10, preview=1024):
    monkeypatch.setattr(envs, "service_activity_tool_call_capacity", capacity)
    monkeypatch.setattr(envs, "service_activity_tool_call_limit", limit)
    monkeypatch.setattr(envs, "service_activity_argument_preview_bytes", preview)


def observe(tracker, index, arguments):
    tracker.observe(
        ExecutionEvent(
            EventType.TOOL_START,
            "stamp",
            {
                "tool_name": "grep",
                "tool_call_id": f"call-{index}",
                "arguments": arguments,
            },
        )
    )


def test_configurable_capacity_default_and_unicode_budget(monkeypatch):
    configure(monkeypatch, capacity=3, limit=2, preview=12)
    tracker = RunActivityTracker()
    for index in range(20):
        observe(tracker, index, {"pattern": "🔎" * 100})
    assert [call.tool_call_id for call in tracker.tool_calls()] == [
        "call-18",
        "call-19",
    ]
    assert len(tracker.tool_calls(3)) == 3
    assert all(
        len(call.arguments_preview.encode()) <= 12 for call in tracker.tool_calls(3)
    )
    with pytest.raises(ValueError):
        tracker.tool_calls(4)
    configure(monkeypatch, capacity=10, limit=8, preview=4096)
    # Existing workers keep an immutable snapshot, even when ambient settings change.
    assert len(tracker.tool_calls()) == 2


@pytest.mark.parametrize("preview", [0, 1, 2, 3, 15])
def test_zero_and_tiny_preview_budgets_are_honored(monkeypatch, preview):
    configure(monkeypatch, preview=preview)
    tracker = RunActivityTracker()
    observe(tracker, 1, {"command": "🔎" * 100})
    call = tracker.tool_calls()[0]
    assert len(call.arguments_preview.encode()) <= preview
    assert call.arguments_truncated is True
    if preview == 0:
        assert call.arguments_preview == ""


def test_larger_preview_budget_can_show_a_longer_single_argument(monkeypatch):
    configure(monkeypatch, preview=4096)
    tracker = RunActivityTracker()
    observe(tracker, 1, {"command": "x" * 2000})
    call = tracker.tool_calls()[0]
    assert json.loads(call.arguments_preview) == {"command": "x" * 2000}
    assert not call.arguments_truncated


def test_default_query_is_capped_when_retention_is_smaller(monkeypatch):
    configure(monkeypatch, capacity=2, limit=10)
    tracker = RunActivityTracker()
    for index in range(5):
        observe(tracker, index, {})
    assert len(tracker.tool_calls()) == 2


@pytest.mark.parametrize(
    "capacity,limit,preview",
    [
        (0, 10, 1024),
        (-1, 10, 1024),
        (True, 10, 1024),
        (10, 0, 1024),
        (10, 10, -1),
        (10, 10, False),
        (sys.maxsize + 1, 10, 1024),
    ],
)
def test_invalid_settings_fail_when_service_is_constructed(
    monkeypatch, capacity, limit, preview
):
    configure(monkeypatch, capacity=capacity, limit=limit, preview=preview)
    store = SQLiteServiceStore()
    try:
        with pytest.raises(ValueError):
            AgentService(store=store)
        assert store.threads() == ()
    finally:
        store.close()


@pytest.mark.asyncio
async def test_worker_uses_service_snapshot_not_client_environment(monkeypatch):
    configure(monkeypatch, capacity=3, limit=2, preview=24)
    service = AgentService(store=SQLiteServiceStore())
    configure(monkeypatch, capacity=100, limit=40, preview=4096)
    ready, release = asyncio.Event(), asyncio.Event()

    async def answer(**kwargs):
        for index in range(10):
            emit_event(
                EventType.TOOL_START,
                {
                    "tool_name": "grep",
                    "tool_call_id": f"call-{index}",
                    "arguments": {"pattern": "x" * 100},
                },
            )
        ready.set()
        await release.wait()
        return _response("done")

    agent = _agent(answer)
    service.register("main", lambda _: AgentSession(agent))
    try:
        thread = await service.open_thread("main")
        receipt = await service.prompt(thread.thread_id, "test", request_id="one")
        await asyncio.wait_for(ready.wait(), 3)

        async def observed():
            while True:
                result = await service.run_tool_calls(thread.thread_id, receipt.run_id)
                if result.tool_calls and result.tool_calls[-1].tool_call_id == "call-9":
                    return result
                await asyncio.sleep(0)

        result = await asyncio.wait_for(observed(), 3)
        assert len(result.tool_calls) == 2
        assert all(
            len(call.arguments_preview.encode()) <= 24 for call in result.tool_calls
        )
        with pytest.raises(ValueError):
            await service.run_tool_calls(thread.thread_id, receipt.run_id, limit=4)
    finally:
        release.set()
        await service.aclose()
        service.store.close()
        await agent.aclose()


@pytest.mark.asyncio
async def test_client_does_not_apply_local_limits_to_remote_server(monkeypatch):
    configure(monkeypatch, capacity=1, limit=1)
    queries = []

    def respond(request):
        queries.append(dict(request.url.params))
        return httpx2.Response(
            200, json={"thread_id": "t", "run_id": "r", "status": "running"}
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as raw:
        client = AgentServiceClient("http://test", token="secret", client=raw)
        await client.run_tool_calls("t", "r")
        await client.run_tool_calls("t", "r", limit=200)
    assert queries == [{}, {"limit": "200"}]


def test_process_environment_overrides_defaults(tmp_path):
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("MSGFLUX_", "MSGTRACE_"))
    }
    environment.update(
        MSGFLUX_SERVICE_ACTIVITY_TOOL_CALL_CAPACITY="17",
        MSGFLUX_SERVICE_ACTIVITY_TOOL_CALL_LIMIT="4",
        MSGFLUX_SERVICE_ACTIVITY_ARGUMENT_PREVIEW_BYTES="93",
    )
    script = 'import json; from msgflux.runtime.service.activity import _ActivitySettings; from msgspec import to_builtins; print("SETTINGS="+json.dumps(to_builtins(_ActivitySettings.from_environment())))'
    result = subprocess.run(  # noqa: S603 - fixed interpreter/script and controlled env
        [sys.executable, "-c", script],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    settings = next(
        line.removeprefix("SETTINGS=")
        for line in result.stdout.splitlines()
        if line.startswith("SETTINGS=")
    )
    assert json.loads(settings) == {
        "capacity": 17,
        "default_limit": 4,
        "preview_bytes": 93,
    }
