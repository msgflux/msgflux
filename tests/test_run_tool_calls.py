"""Recent tool calls are bounded projections, never outputs or retained state.

These tests pin the observation contract: the tracker keeps bounded previews of
started calls only, the service reads them without loading a session or a
checkpoint, and no argument, result or model text survives as a live reference.
"""

import asyncio
import gc
import json
import os
import weakref
from unittest.mock import AsyncMock, Mock

import httpx2
import msgspec
import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.envs import envs
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.events import EventType, ExecutionEvent, emit_event
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    RunToolCall,
    RunToolCalls,
    SQLiteServiceStore,
)
from msgflux.runtime.service.activity import (
    ARGUMENT_PREVIEW_BYTES,
    DEFAULT_TOOL_CALL_LIMIT,
    MAX_TOOL_CALLS,
    RunActivityTracker,
)
from msgflux.runtime.service.records import ServiceThread

# Documented defaults, also used as explicit expectations below.
CAPACITY = 50
DEFAULT_LIMIT = 10
PREVIEW_BYTES = 1024

_HOST_ENVIRONMENT_KEYS = (
    "MSGFLUX_SERVICE_ACTIVITY_TOOL_CALL_CAPACITY",
    "MSGFLUX_SERVICE_ACTIVITY_TOOL_CALL_LIMIT",
    "MSGFLUX_SERVICE_ACTIVITY_ARGUMENT_PREVIEW_BYTES",
)


@pytest.fixture(autouse=True)
def pinned_observation_settings(monkeypatch):
    """Keep assertions independent of observation envs configured by the host."""
    monkeypatch.setattr(envs, "service_activity_tool_call_capacity", CAPACITY)
    monkeypatch.setattr(envs, "service_activity_tool_call_limit", DEFAULT_LIMIT)
    monkeypatch.setattr(envs, "service_activity_argument_preview_bytes", PREVIEW_BYTES)


def _start(
    name="read_file",
    *,
    arguments=None,
    call_id="call-1",
    run_id="run",
    source=("tool:read_file",),
    stamp="2026-01-01T00:00:00+00:00",
    include_arguments=True,
):
    data = {"tool_name": name, "tool_call_id": call_id}
    if include_arguments:
        data["arguments"] = {} if arguments is None else arguments
    return ExecutionEvent(
        EventType.TOOL_START, stamp, data, run_id=run_id, source_path=source
    )


def _observe_starts(tracker, count, *, name="read_file"):
    for index in range(count):
        tracker.observe(
            _start(
                f"{name}-{index}",
                call_id=f"call-{index}",
                stamp=f"2026-01-01T00:00:{index:02d}+00:00",
                include_arguments=False,
            )
        )


def _preview_bytes(call):
    encoded = call.arguments_preview.encode("utf-8")
    # A byte-level cut must never emit a half-encoded character or raise.
    assert call.arguments_preview == encoded.decode("utf-8")
    return encoded


def test_module_constants_expose_the_documented_defaults():
    if any(key in os.environ for key in _HOST_ENVIRONMENT_KEYS):
        pytest.skip("tool-call observation defaults are configured by the host")
    assert MAX_TOOL_CALLS == CAPACITY
    assert DEFAULT_TOOL_CALL_LIMIT == DEFAULT_LIMIT
    assert ARGUMENT_PREVIEW_BYTES == PREVIEW_BYTES


def test_tracker_returns_the_bounded_oldest_to_newest_tail():
    tracker = RunActivityTracker()
    _observe_starts(tracker, DEFAULT_LIMIT + 5)

    calls = tracker.tool_calls()
    assert isinstance(calls, tuple)
    assert [call.tool_name for call in calls] == [
        f"read_file-{index}" for index in range(5, DEFAULT_LIMIT + 5)
    ]
    assert all(isinstance(call, RunToolCall) for call in calls)
    assert [call.tool_name for call in tracker.tool_calls(limit=3)] == [
        "read_file-12",
        "read_file-13",
        "read_file-14",
    ]
    assert [call.tool_call_id for call in tracker.tool_calls(limit=1)] == ["call-14"]
    assert len(tracker.tool_calls(limit=CAPACITY)) == DEFAULT_LIMIT + 5


def test_tracker_default_limit_is_read_per_request_and_has_no_cursor():
    tracker = RunActivityTracker()
    _observe_starts(tracker, DEFAULT_LIMIT + 2)

    default = tracker.tool_calls()
    assert len(default) == DEFAULT_LIMIT
    assert default[-1].tool_name == f"read_file-{DEFAULT_LIMIT + 1}"
    # ``None`` means "server default"; it is never a silent unlimited read.
    assert tracker.tool_calls(limit=None) == default
    assert tracker.tool_calls(limit=DEFAULT_LIMIT) == default
    # Reading is idempotent: observers do not consume a shared cursor.
    assert tracker.tool_calls() == default
    assert tracker.tool_calls(limit=2) == default[-2:]
    assert tracker.tool_calls() == default


def test_tracker_retains_at_most_the_capacity():
    tracker = RunActivityTracker()
    overflow = CAPACITY + 7
    _observe_starts(tracker, overflow)

    retained = tracker.tool_calls(limit=CAPACITY)
    assert len(retained) == CAPACITY
    assert retained[0].tool_name == f"read_file-{overflow - CAPACITY}"
    assert retained[-1].tool_name == f"read_file-{overflow - 1}"


@pytest.mark.parametrize("bad_limit", [0, -1, CAPACITY + 1, True, False, "10", 10.0])
def test_tracker_rejects_limits_outside_the_strict_bounds(bad_limit):
    tracker = RunActivityTracker()
    tracker.observe(_start())
    with pytest.raises(ValueError):
        tracker.tool_calls(limit=bad_limit)


def test_tracker_records_bounded_metadata_labels():
    tracker = RunActivityTracker()
    tracker.observe(
        _start(
            "n" * 1000,
            call_id="c" * 1000,
            run_id="r" * 1000,
            stamp="t" * 1000,
            source=tuple(f"s{index}" * 200 for index in range(20)),
            arguments={"path": "src"},
        )
    )

    call = tracker.tool_calls()[0]
    assert call.tool_name.startswith("nnnn") and len(call.tool_name) <= 128
    assert call.tool_call_id is not None and len(call.tool_call_id) <= 128
    assert call.run_id is not None and len(call.run_id) <= 128
    assert call.timestamp.startswith("tttt") and len(call.timestamp) <= 64
    assert len(call.source_path) == 8
    assert all(len(part) <= 128 for part in call.source_path)
    assert isinstance(call.arguments_preview, str)


def test_preview_is_bounded_in_utf8_bytes():
    tracker = RunActivityTracker()
    tracker.observe(
        _start(arguments={"path": "src", "text": "á" * 4000, "é": "∑" * 4000})
    )

    truncated = tracker.tool_calls()[0]
    assert truncated.arguments_truncated is True
    assert len(_preview_bytes(truncated)) <= PREVIEW_BYTES

    tracker.observe(
        _start("stat", call_id="call-2", arguments={"path": "src/agent.py", "limit": 5})
    )
    small = tracker.tool_calls()[-1]
    assert small.arguments_truncated is False
    assert json.loads(small.arguments_preview) == {"path": "src/agent.py", "limit": 5}


def test_preview_bounds_width_depth_strings_and_cycles():
    cycle = {"name": "root", "level": 1}
    cycle["self"] = {"name": "child", "parent": cycle}
    items = [0, 1]
    items.append(items)
    tracker = RunActivityTracker()
    tracker.observe(
        _start(
            arguments={
                "wide": {f"key{index}": index for index in range(5000)},
                "deep": {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}},
                "long": "x" * 10000,
                "list": list(range(5000)),
                "cycle": cycle,
                "recursive_list": items,
            }
        )
    )

    call = tracker.tool_calls()[0]
    assert call.arguments_truncated is True
    assert len(_preview_bytes(call)) <= PREVIEW_BYTES
    # Bounded traversal keeps the projection small even for very wide input.
    assert "key4999" not in call.arguments_preview


def test_preview_masks_sensitive_named_fields():
    tracker = RunActivityTracker()
    tracker.observe(
        _start(
            arguments={
                "path": "src/agent.py",
                "password": "hunter2-LEAK",
                "api_key": "sk-LEAK",
                "Authorization": "Bearer LEAK",
                "cookie": "session=LEAK",
                "client_secret": "LEAK",
                "nested": {"auth_token": "LEAK", "ok": True},
            }
        )
    )

    call = tracker.tool_calls()[0]
    for secret in ("hunter2-LEAK", "sk-LEAK", "Bearer LEAK", "session=LEAK"):
        assert secret not in call.arguments_preview
    # Masking is per field: unrelated arguments keep their value.
    assert "src/agent.py" in call.arguments_preview


def test_preview_never_calls_repr_or_str_of_arbitrary_values():
    class Unprintable:
        def __repr__(self):
            raise AssertionError("repr must not be called")

        def __str__(self):
            raise AssertionError("str must not be called")

    tracker = RunActivityTracker()
    tracker.observe(
        _start(
            arguments={
                "text": "visible",
                "object": Unprintable(),
                "bytes": b"raw",
                "mapping": {"inner": Unprintable()},
                "huge_int": 10**5000,
                "infinity": float("inf"),
            }
        )
    )

    call = tracker.tool_calls()[0]
    assert call.arguments_truncated is True
    assert "visible" in call.arguments_preview
    assert "Unprintable" not in call.arguments_preview
    assert len(_preview_bytes(call)) <= PREVIEW_BYTES


def test_tracker_does_not_retain_argument_or_output_references():
    class Payload:
        pass

    arguments = {"body": Payload()}
    argument_ref = weakref.ref(arguments["body"])
    result = Payload()
    result_ref = weakref.ref(result)
    tracker = RunActivityTracker()
    event = _start(arguments=arguments)
    ended = ExecutionEvent(
        EventType.TOOL_END, "stamp", {"tool_name": "read_file", "result": result}
    )
    delta = ExecutionEvent(EventType.MESSAGE_DELTA, "stamp", {"delta": "PRIVATE"})
    tracker.observe(event)
    tracker.observe(ended)
    tracker.observe(delta)
    retained = tracker.tool_calls()
    del arguments, event, ended, delta, result
    gc.collect()

    assert argument_ref() is None
    assert result_ref() is None
    assert "Payload" not in retained[0].arguments_preview


def test_tracker_previews_are_isolated_from_later_mutation():
    arguments = {"path": "src/a.py", "items": [1, 2], "nested": {"key": "value"}}
    tracker = RunActivityTracker()
    tracker.observe(_start(arguments=arguments))
    before = tracker.tool_calls()[0].arguments_preview

    arguments["path"] = "etc/MUTATED"
    arguments["items"].append(3)
    arguments["nested"]["key"] = "MUTATED"
    arguments["injected"] = "MUTATED"
    arguments["items"].clear()

    after = tracker.tool_calls()[0].arguments_preview
    assert after == before
    assert "src/a.py" in after
    assert "MUTATED" not in after


def test_records_are_frozen_projections():
    tracker = RunActivityTracker()
    tracker.observe(_start(arguments={"path": "src"}))
    call = tracker.tool_calls()[0]
    with pytest.raises((AttributeError, TypeError)):
        call.tool_name = "other"


def test_other_events_keep_no_call_data():
    tracker = RunActivityTracker()
    tracker.observe(
        ExecutionEvent(
            EventType.TOOL_END,
            "stamp",
            {"tool_name": "read_file", "arguments": {"path": "LEAK"}, "result": "LEAK"},
        )
    )
    tracker.observe(ExecutionEvent(EventType.TOOL_UPDATE, "stamp", {"result": "LEAK"}))
    tracker.observe(ExecutionEvent(EventType.MESSAGE_DELTA, "stamp", {"delta": "LEAK"}))
    tracker.observe(
        ExecutionEvent(EventType.MODEL_REQUEST, "stamp", {"arguments": "LEAK"})
    )
    tracker.observe(
        ExecutionEvent(EventType.TOOL_BLOCKED, "stamp", {"tool_name": "read_file"})
    )
    tracker.observe(ExecutionEvent(EventType.TOOL_START, "stamp", {"arguments": {}}))
    tracker.observe(ExecutionEvent(EventType.TOOL_START, "stamp", {"tool_name": ""}))
    tracker.observe(ExecutionEvent(EventType.TOOL_START, "stamp", None))

    assert tracker.tool_calls() == ()
    assert "LEAK" not in msgspec.json.encode(tracker.tool_calls()).decode()


def test_tracker_capacity_and_default_follow_configured_settings(monkeypatch):
    monkeypatch.setattr(envs, "service_activity_tool_call_capacity", 3)
    monkeypatch.setattr(envs, "service_activity_tool_call_limit", 9)
    tracker = RunActivityTracker()
    _observe_starts(tracker, 6)

    # The configured default is clamped to the configured capacity.
    assert [call.tool_name for call in tracker.tool_calls()] == [
        "read_file-3",
        "read_file-4",
        "read_file-5",
    ]
    assert tracker.tool_calls(limit=None) == tracker.tool_calls()
    assert [call.tool_name for call in tracker.tool_calls(limit=2)] == [
        "read_file-4",
        "read_file-5",
    ]
    with pytest.raises(ValueError):
        tracker.tool_calls(limit=4)


def test_preview_bytes_configuration_bounds_and_suppresses_previews(monkeypatch):
    monkeypatch.setattr(envs, "service_activity_argument_preview_bytes", 8)
    small = RunActivityTracker()
    small.observe(_start(arguments={"path": "src/agent.py"}))
    call = small.tool_calls()[0]
    assert call.arguments_truncated is True
    assert len(_preview_bytes(call)) <= 8

    monkeypatch.setattr(envs, "service_activity_argument_preview_bytes", 0)
    suppressed = RunActivityTracker()
    suppressed.observe(_start(arguments={"path": "src/agent.py"}))
    empty = suppressed.tool_calls()[0]
    assert empty.arguments_preview == ""
    assert empty.arguments_truncated is True


def test_records_encode_without_output_payloads():
    calls = RunToolCalls(
        thread_id="thread",
        run_id="run",
        status="running",
        local_worker=True,
        tool_calls=(
            RunToolCall(
                tool_name="grep",
                timestamp="stamp",
                tool_call_id="call-1",
                run_id="run",
                source_path=("tool:grep",),
                arguments_preview='{"path":"src"}',
                arguments_truncated=True,
            ),
        ),
    )
    decoded = msgspec.json.decode(msgspec.json.encode(calls), type=RunToolCalls)
    assert decoded == calls
    assert decoded.local_worker is True
    assert decoded.tool_calls[0].arguments_truncated is True


def _pending_service():
    store = SQLiteServiceStore()
    store.bind_thread(ServiceThread(thread_id="thread", agent_id="main"))
    record = store.admit("thread", "request", "private prompt", "main")
    service = AgentService(store=store)

    def forbidden(_thread):
        pytest.fail("run_tool_calls hydrated an Agent session")

    service.register("main", forbidden)
    return service, store, record.receipt.run_id


@pytest.mark.asyncio
async def test_service_reads_no_calls_without_a_local_worker_or_hydration():
    service, store, run_id = _pending_service()
    try:
        calls = await service.run_tool_calls("thread", run_id)
        assert calls == RunToolCalls(
            thread_id="thread",
            run_id=run_id,
            status="accepted",
            local_worker=False,
            tool_calls=(),
        )
        assert await service.run_tool_calls("thread", run_id, limit=3) == calls
        assert await service.run_tool_calls("thread", run_id, limit=None) == calls
        assert service._sessions == {}
        assert "private prompt" not in msgspec.json.encode(calls).decode()
        with pytest.raises(KeyError):
            await service.run_tool_calls("thread", "missing")
        with pytest.raises(KeyError):
            await service.run_tool_calls("unknown", run_id)
    finally:
        await service.aclose()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_limit", [0, -1, CAPACITY + 1, True, "10", 10.0])
async def test_service_rejects_limits_outside_the_strict_bounds(bad_limit):
    service, store, run_id = _pending_service()
    try:
        with pytest.raises(ValueError):
            await service.run_tool_calls("thread", run_id, limit=bad_limit)
    finally:
        await service.aclose()
        store.close()


@pytest.mark.asyncio
async def test_service_uses_the_capacity_configured_at_start(monkeypatch):
    monkeypatch.setattr(envs, "service_activity_tool_call_capacity", 3)
    monkeypatch.setattr(envs, "service_activity_tool_call_limit", 9)
    service, store, run_id = _pending_service()
    try:
        assert (await service.run_tool_calls("thread", run_id)).tool_calls == ()
        assert (
            await service.run_tool_calls("thread", run_id, limit=3)
        ).tool_calls == ()
        with pytest.raises(ValueError):
            await service.run_tool_calls("thread", run_id, limit=4)
    finally:
        await service.aclose()
        store.close()


def _live_agent(answer):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="main", model=model, tools=[], checkpoint_store=InMemoryCheckpointStore()
    )
    agent.generator.aforward = AsyncMock(side_effect=answer)
    return agent


def _response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


@pytest.mark.asyncio
async def test_live_worker_tracks_the_default_tail_then_discards_it():
    entered, release = asyncio.Event(), asyncio.Event()

    async def answer(**_kwargs):
        for index in range(DEFAULT_LIMIT + 3):
            emit_event(
                EventType.TOOL_START,
                {
                    "tool_name": "grep",
                    "tool_call_id": f"call-{index}",
                    "arguments": {"path": "src", "api_key": "PRIVATE-KEY"},
                },
            )
            emit_event(
                EventType.TOOL_END,
                {"tool_name": "grep", "result": "PRIVATE-OUTPUT"},
            )
        entered.set()
        await release.wait()
        return _response("done")

    agent = _live_agent(answer)
    store = SQLiteServiceStore()
    service = AgentService(store=store)
    service.register("main", lambda _thread: AgentSession(agent))
    try:
        thread = await service.open_thread("main")
        receipt = await service.prompt(
            thread.thread_id, "PRIVATE-PROMPT", request_id="1"
        )
        await asyncio.wait_for(entered.wait(), 5)

        async def observed():
            while True:
                calls = await service.run_tool_calls(thread.thread_id, receipt.run_id)
                if len(calls.tool_calls) == DEFAULT_LIMIT:
                    return calls
                await asyncio.sleep(0)

        calls = await asyncio.wait_for(observed(), 5)
        assert calls.local_worker is True
        assert calls.tool_calls[0].tool_call_id == "call-3"
        assert calls.tool_calls[-1].tool_call_id == f"call-{DEFAULT_LIMIT + 2}"
        encoded = msgspec.json.encode(calls)
        assert b"PRIVATE-KEY" not in encoded and b"PRIVATE-OUTPUT" not in encoded
        assert b"PRIVATE-PROMPT" not in encoded

        release.set()
        await asyncio.wait_for(service.wait(thread.thread_id, "1"), 5)
        settled = await service.run_tool_calls(thread.thread_id, receipt.run_id)
        assert settled.local_worker is False
        assert settled.tool_calls == ()
    finally:
        release.set()
        await service.aclose()
        store.close()
        await agent.aclose()


@pytest.mark.asyncio
async def test_http_route_checks_auth_and_limit_without_hydration():
    pytest.importorskip("litestar")
    from litestar.testing import AsyncTestClient

    from msgflux.runtime.service.http.app import create_service_app

    service, store, run_id = _pending_service()
    app = create_service_app(service, token="secret")
    headers = {"Authorization": "Bearer secret"}
    path = f"/v1/threads/thread/runs/{run_id}/tool-calls"
    try:
        async with AsyncTestClient(app=app) as client:
            default = await client.get(path, headers=headers)
            assert default.status_code == 200
            assert default.json() == {
                "thread_id": "thread",
                "run_id": run_id,
                "status": "accepted",
                "local_worker": False,
                "tool_calls": [],
            }
            limited = await client.get(f"{path}?limit=3", headers=headers)
            assert limited.status_code == 200
            assert limited.json() == default.json()
            assert (await client.get(path)).status_code == 401
            for bad in ("0", str(CAPACITY + 1), "bad"):
                response = await client.get(f"{path}?limit={bad}", headers=headers)
                assert response.status_code == 422
                assert response.json()["code"] == "invalid_request"
            missing = await client.get(
                "/v1/threads/thread/runs/missing/tool-calls", headers=headers
            )
            assert missing.status_code == 404
            assert missing.json()["code"] == "not_found"
        assert service._sessions == {}
    finally:
        await service.aclose()
        store.close()


@pytest.mark.asyncio
async def test_client_omits_the_limit_unless_explicitly_requested():
    from msgflux.runtime.service.http.client import AgentServiceClient

    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(
            200,
            json={
                "thread_id": "t",
                "run_id": "r",
                "status": "running",
                "local_worker": True,
                "tool_calls": [
                    {
                        "tool_name": "grep",
                        "timestamp": "stamp",
                        "tool_call_id": "call-1",
                        "run_id": "r",
                        "source_path": ["tool:grep"],
                        "arguments_preview": '{"path":"src"}',
                        "arguments_truncated": False,
                    }
                ],
            },
            request=request,
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http:
        client = AgentServiceClient("http://service", token="secret", client=http)
        default = await client.run_tool_calls("t", "r")
        assert requests[-1].url.path == "/v1/threads/t/runs/r/tool-calls"
        assert "limit" not in requests[-1].url.params
        assert default.thread_id == "t" and default.run_id == "r"
        assert default.local_worker is True
        assert [call.tool_name for call in default.tool_calls] == ["grep"]
        assert default.tool_calls[0].source_path == ("tool:grep",)

        # ``None`` leaves the server default in charge of capacity and bounds.
        assert await client.run_tool_calls("t", "r", limit=None) == default
        assert "limit" not in requests[-1].url.params

        assert await client.run_tool_calls("t", "r", limit=3) == default
        assert requests[-1].url.params.get("limit") == "3"

        seen = len(requests)
        for bad in (0, -1, True, "10", 10.0):
            with pytest.raises(ValueError):
                await client.run_tool_calls("t", "r", limit=bad)
        assert len(requests) == seen
