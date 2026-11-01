"""Real-socket regression for native SSE over negotiated HTTP compression.

Every test drives a real Uvicorn TCP listener. An ASGI transport buffers the
response body, which would hide whether an event reached the client before the
producer finished, so these tests inject a real ``httpx2.AsyncClient`` that
advertises one encoding and observe the wire headers through a response hook.
"""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager

import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")

pytestmark = pytest.mark.filterwarnings(
    "error::pytest.PytestUnraisableExceptionWarning"
)

import httpx2

from msgflux.runtime.events import EventType, emit_event
from msgflux.runtime.service import AgentSession
from msgflux.runtime.service.http import (
    AgentServiceClient,
    AgentServiceProtocolError,
    ServiceHTTPConfig,
)
from tests.test_agent_service_http_integration import (
    TOKEN,
    _agent,
    _no_watchers,
    _response,
    _server,
)

# Bounded values only guard against hangs and size regressions; no test asserts
# throughput or elapsed time.
_TIMEOUT = 15
_MIB = 1 << 20
_UNSET = object()
_TERMINAL = {EventType.RUN_END, EventType.RUN_ERROR}
# The service extra ships both codecs, so both are exercised unconditionally.
_ENCODINGS = ["zstd", "gzip"]


@asynccontextmanager
async def _streaming_client(url, encoding, *, max_event_size=_UNSET, seen=None):
    """Client over real TCP advertising one encoding; always closes its pool."""
    event_hooks = None
    if seen is not None:

        async def capture(response):
            if response.request.url.path.endswith("/watch"):
                seen.append(response)

        event_hooks = {"response": [capture]}
    http = httpx2.AsyncClient(
        headers={"Accept-Encoding": encoding},
        event_hooks=event_hooks,
        trust_env=False,
    )
    options = {} if max_event_size is _UNSET else {"max_event_size": max_event_size}
    try:
        yield AgentServiceClient(url, token=TOKEN, client=http, **options)
    finally:
        await http.aclose()


@asynccontextmanager
async def _watching(client, thread_id):
    """Enter a watch with a bounded hang guard and always exit it cleanly."""
    context = client.watch(thread_id)
    # A buffered transport could never produce the snapshot before the stream
    # ends, so a bounded enter timeout fails loudly instead of hanging.
    watcher = await asyncio.wait_for(context.__aenter__(), _TIMEOUT)
    try:
        yield watcher
    finally:
        await context.__aexit__(None, None, None)


async def _enter_watch_fails(client, thread_id):
    """Enter a watch expected to be refused, still unwinding on failure."""
    async with AsyncExitStack() as stack:
        await asyncio.wait_for(
            stack.enter_async_context(client.watch(thread_id)), _TIMEOUT
        )


def _content_size(snapshot):
    return sum(len(str(item.get("content"))) for item in snapshot.messages or ())


async def _seed_history(client, service, thread_id, chunks, *, prefix="seed"):
    """Ask the fake model once per chunk so the history grows legitimately."""
    for index, chunk in enumerate(chunks):
        request_id = f"{prefix}-{index}"
        await client.prompt(thread_id, chunk, request_id=request_id)
        settled = await asyncio.wait_for(service.wait(thread_id, request_id), _TIMEOUT)
        assert settled.status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", _ENCODINGS)
async def test_compressed_watch_streams_snapshot_and_progress_before_completion(
    encoding,
):
    entered, release = asyncio.Event(), asyncio.Event()
    marker = f"progress-{encoding}"
    request_id = f"stream-{encoding}"
    thread_id = f"compressed-{encoding}"

    async def answer(**_kwargs):
        # Deterministic mid-run progress while the producer stays blocked.
        emit_event(EventType.COMMENTARY_DELTA, {"delta": marker})
        entered.set()
        await release.wait()
        return _response(f"answer-{encoding}")

    seen = []
    agent = _agent(answer)
    try:
        async with _server(
            lambda _thread: AgentSession(agent),
            http_config=ServiceHTTPConfig(compression=True),
        ) as (service, url):
            async with _streaming_client(url, encoding, seen=seen) as client:
                thread = await client.open_thread("main", thread_id=thread_id)
                async with _watching(client, thread.thread_id) as watcher:
                    # The negotiated encoding is confirmed on the wire. Entering
                    # the watcher already returned the snapshot while no prompt
                    # was issued, which a buffering transport cannot do.
                    assert len(seen) == 1
                    assert seen[0].headers["content-encoding"] == encoding
                    assert watcher.snapshot.thread_id == thread.thread_id
                    assert watcher.snapshot.active_runs == ()

                    receipt = await client.prompt(
                        thread.thread_id, "hello", request_id=request_id
                    )
                    await asyncio.wait_for(entered.wait(), _TIMEOUT)

                    progress = []
                    while True:
                        if release.is_set():
                            raise AssertionError("run settled before progress")
                        event = await asyncio.wait_for(anext(watcher), _TIMEOUT)
                        progress.append(event)
                        if event.type == EventType.COMMENTARY_DELTA:
                            assert event.data["delta"] == marker
                            break
                    # Progress arrives while the run is still in flight.
                    assert not release.is_set()
                    assert not any(item.type in _TERMINAL for item in progress)
                    running = await client.receipt(thread.thread_id, request_id)
                    assert running.status == "running"

                    release.set()
                    tail = []
                    while True:
                        event = await asyncio.wait_for(anext(watcher), _TIMEOUT)
                        tail.append(event)
                        if (
                            event.type == EventType.RUN_END
                            and event.run_id == receipt.run_id
                        ):
                            break
                    assert any(item.type == EventType.MESSAGE_END for item in tail)
                    assert f"answer-{encoding}" in str(tail)
                await asyncio.wait_for(_no_watchers(thread_id), _TIMEOUT)
                settled = await asyncio.wait_for(
                    service.wait(thread.thread_id, request_id), _TIMEOUT
                )
                assert settled.status == "completed"
                assert agent.generator.aforward.await_count == 1
    finally:
        release.set()


@pytest.mark.asyncio
async def test_watcher_close_detaches_observer_without_interrupting_producer():
    entered, release = asyncio.Event(), asyncio.Event()
    calls = []

    async def answer(**_kwargs):
        calls.append("started")
        entered.set()
        await release.wait()
        calls.append("released")
        return _response("finished")

    agent = _agent(answer)
    thread_id = "compressed-close"
    try:
        async with _server(
            lambda _thread: AgentSession(agent),
            http_config=ServiceHTTPConfig(compression=True),
        ) as (service, url):
            async with _streaming_client(url, "zstd") as client:
                thread = await client.open_thread("main", thread_id=thread_id)
                async with AsyncExitStack() as stack:
                    watcher = await asyncio.wait_for(
                        stack.enter_async_context(client.watch(thread.thread_id)),
                        _TIMEOUT,
                    )
                    assert watcher.snapshot.thread_id == thread.thread_id
                    receipt = await client.prompt(
                        thread.thread_id, "go", request_id="close"
                    )
                    await asyncio.wait_for(entered.wait(), _TIMEOUT)

                    # Leaving the watcher detaches the observer while the
                    # producer is still blocked; a repeated close is idempotent.
                    await stack.aclose()
                    await watcher.aclose()
                    with pytest.raises(StopAsyncIteration):
                        await watcher.__anext__()

                    assert not release.is_set()
                    await asyncio.wait_for(_no_watchers(thread_id), _TIMEOUT)
                    assert not release.is_set()

                    release.set()
                    settled = await asyncio.wait_for(
                        service.wait(thread.thread_id, "close"), _TIMEOUT
                    )
                    assert settled.status == "completed"
                    assert settled.run_id == receipt.run_id
                    assert calls == ["started", "released"]
                    snapshot = await client.snapshot(thread.thread_id)
                    assert "finished" in str(snapshot.messages)
    finally:
        release.set()


@pytest.mark.asyncio
async def test_large_compressed_snapshot_respects_default_and_explicit_event_limits():
    marker = "large-history-marker"
    thread_id = "compressed-large"
    chunks = [f"{marker}-{index}:" + "0123456789abcdef" * 12_000 for index in range(8)]
    assert sum(len(chunk) for chunk in chunks) > _MIB
    agent = _agent(lambda **_kwargs: _response("noted"))
    async with _server(
        lambda _thread: AgentSession(agent),
        http_config=ServiceHTTPConfig(compression=True),
    ) as (service, url):
        seen = []
        # Default ``max_event_size=None``: the native 1 MiB cap must not apply
        # to a legitimately large compressed snapshot.
        async with _streaming_client(url, "zstd", seen=seen) as client:
            thread = await client.open_thread("main", thread_id=thread_id)
            await _seed_history(client, service, thread.thread_id, chunks)
            async with _watching(client, thread.thread_id) as watcher:
                assert seen[0].headers["content-encoding"] == "zstd"
                assert _content_size(watcher.snapshot) > _MIB
                assert marker in str(watcher.snapshot.messages)
            await asyncio.wait_for(_no_watchers(thread.thread_id), _TIMEOUT)

            # An explicit 1 MiB cap still rejects the decompressed snapshot.
            async with _streaming_client(url, "zstd", max_event_size=_MIB) as bounded:
                with pytest.raises(AgentServiceProtocolError):
                    await _enter_watch_fails(bounded, thread.thread_id)
            # A further prompt gives the server a write to observe the
            # disconnect on, so a leaked observer would surface here.
            await _seed_history(
                client, service, thread.thread_id, ["ping"], prefix="after"
            )
            await asyncio.wait_for(_no_watchers(thread.thread_id), _TIMEOUT)

            # A larger limit accepts the same snapshot, and reconnecting starts
            # from a fresh snapshot instead of replaying missed deltas.
            async with _streaming_client(
                url, "zstd", max_event_size=8 * _MIB
            ) as generous:
                async with _watching(generous, thread.thread_id) as watcher:
                    assert _content_size(watcher.snapshot) > _MIB
                async with _watching(generous, thread.thread_id) as reconnected:
                    assert reconnected.snapshot.thread_id == thread.thread_id
                    assert reconnected.snapshot.active_runs == ()
                    assert marker in str(reconnected.snapshot.messages)
            await asyncio.wait_for(_no_watchers(thread.thread_id), _TIMEOUT)
