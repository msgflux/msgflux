"""The CodingSession and HTTP session facades share the same async contract."""

import asyncio
import socket
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import httpx2
import pytest

pytest.importorskip("litestar")
uvicorn = pytest.importorskip("uvicorn")

from msgflux.coding import CodingSession
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import (
    AgentApprovals,
    AgentWorkspace,
    ExecutionScope,
    InMemoryApprovalStore,
)
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    ServiceRecoveryRequiredError,
    SQLiteServiceStore,
)
from msgflux.runtime.service.records import EventRecord, SnapshotRecord
from msgflux.runtime.service.http import (
    AgentServiceClient,
    AgentServiceHTTPError,
    AgentSessionClient,
)
from msgflux.runtime.service.http import create_service_app
from msgflux.runtime.events import _hub_event_sink
from msgflux.tools.builtin import WriteTool
from msgflux.utils.msgspec import msgspec_dumps

TOKEN = "coding-session-parity-token"


def _text(content):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    return response


def _tool(name, arguments):
    calls = ToolCallAggregator()
    calls.process(0, "call-1", name, msgspec_dumps(arguments))
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _agent(*, answer, tools=None, approvals=None, agent_dir=None, workspace=None):
    model = Mock()
    model.model_type = "chat_completion"
    agent = Agent(
        name="main",
        model=model,
        tools=tools,
        approvals=approvals,
        agent_dir=agent_dir,
        workspace=workspace,
        checkpoint_store=None if agent_dir else InMemoryCheckpointStore(),
    )
    agent.generator.aforward = AsyncMock(side_effect=answer)
    return agent


@asynccontextmanager
def _mark_foreign_attempt(service, thread_id, request_id):
    record = service.store.admit(thread_id, request_id, "in flight", "parity-test")
    assert service.store.claim(record, "previous-owner")


@asynccontextmanager
async def _session(kind, agent, thread_id, *, foreign_request=None):
    store = SQLiteServiceStore()
    service = AgentService(store=store)
    service.register(
        "main",
        lambda _thread: AgentSession(
            agent,
            approval_reviewer="human",
            scope_factory=lambda scope: scope.with_overrides(principal="executor"),
        ),
    )
    if kind == "local":
        try:
            thread = await service.open_thread("main", thread_id=thread_id)
            if foreign_request is not None:
                _mark_foreign_attempt(service, thread.thread_id, foreign_request)
            yield await CodingSession.from_service(service, thread.thread_id)
        finally:
            await service.aclose()
            store.close()
        return

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            create_service_app(service, token=TOKEN),
            host="127.0.0.1",
            port=port,
            log_level="error",
            access_log=False,
            ws="none",
            lifespan="on",
            timeout_graceful_shutdown=2,
        )
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("HTTP server exited before startup")
            await asyncio.sleep(0.01)
        async with httpx2.AsyncClient() as http:
            client = AgentServiceClient(
                f"http://127.0.0.1:{port}", token=TOKEN, client=http
            )
            session = await AgentSessionClient.open(client, thread_id=thread_id)
            if foreign_request is not None:
                _mark_foreign_attempt(service, session.thread_id, foreign_request)
            yield session
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            listener.close()
            await service.aclose()
            store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["local", "http"])
async def test_empty_history_is_typed_and_does_not_create_managed_thread_dirs(
    tmp_path, kind
):
    agent_dir = tmp_path / kind / "agent"
    agent_dir.mkdir(parents=True)
    agent = _agent(answer=lambda **_kwargs: _text("unused"), agent_dir=agent_dir)
    async with _session(kind, agent, "lazy-thread") as session:
        assert await session.runs() == ()
        assert await session.latest_run() is None
        assert not (agent_dir / "threads").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["local", "http"])
async def test_prompt_wait_receipt_and_run_summaries_share_types(kind):
    agent = _agent(answer=lambda **_kwargs: _text("finished"))
    async with _session(kind, agent, "typed-thread") as session:
        admitted = await session.prompt("hello", request_id="typed-request")
        settled = await asyncio.wait_for(session.wait("typed-request"), timeout=5)
        receipt = await session.receipt("typed-request")
        runs = await session.runs()
        latest = await session.latest_run()
        snapshot = await session.snapshot()

        assert type(admitted) is type(settled) is type(receipt)
        assert admitted.run_id == settled.run_id == receipt.run_id
        assert settled.status == receipt.status == "completed"
        assert runs and type(runs[0]) is type(latest)
        assert runs[0].run_id == latest.run_id == admitted.run_id
        assert isinstance(snapshot, SnapshotRecord)
        assert snapshot.thread_id == session.thread_id
        assert isinstance(snapshot.messages, tuple)
        assert all(isinstance(message, dict) for message in snapshot.messages)
        assert isinstance(snapshot.active_runs, tuple)
        assert all(isinstance(run, dict) for run in snapshot.active_runs)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["local", "http"])
async def test_approval_resume_returns_receipt_for_same_run_and_is_observed_by_watch(
    kind,
):
    calls = []

    def lookup(value):
        calls.append(value)
        return "found"

    agent = _agent(
        answer=None,
        tools=[lookup],
        approvals=AgentApprovals(
            InMemoryApprovalStore(), {"lookup": "v1"}, "policy-v1"
        ),
    )
    agent.generator.aforward = AsyncMock(
        side_effect=[_tool("lookup", {"value": "key"}), _text("done")]
    )
    async with _session(kind, agent, "approval-thread") as session:
        admitted = await session.prompt("look it up", request_id="approval-request")
        paused = await asyncio.wait_for(session.wait("approval-request"), timeout=5)
        assert paused.status == "paused"
        (review,) = await session.approval_reviews(admitted.run_id)
        await session.decide_approval(
            admitted.run_id,
            review.request_id,
            approved=True,
            expected_revision=review.revision,
        )

        async with session.watch() as watcher:
            resume_receipt = await session.resume(admitted.run_id)
            assert resume_receipt.run_id == admitted.run_id
            events = []
            while True:
                event = await asyncio.wait_for(anext(watcher), timeout=5)
                events.append(event)
                if event.run_id == admitted.run_id and event.type == "run.end":
                    break

        settled = await asyncio.wait_for(session.wait("approval-request"), timeout=5)
        assert settled.status == "completed"
        assert calls == ["key"]
        assert all(isinstance(event, EventRecord) for event in events)
        assert all(isinstance(event.data, dict) for event in events)
        assert events[-1].run_id == admitted.run_id
        assert events[-1].type == "run.end"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["local", "http"])
async def test_closing_watch_observer_leaves_service_producer_running(kind):
    entered, release = asyncio.Event(), asyncio.Event()

    async def answer(**_kwargs):
        entered.set()
        await release.wait()
        return _text("survived observer")

    agent = _agent(answer=answer)
    async with _session(kind, agent, "observer-thread") as session:
        async with session.watch() as watcher:
            admitted = await session.prompt("wait", request_id="observer-request")
            await asyncio.wait_for(entered.wait(), timeout=5)
        assert (await session.receipt("observer-request")).status == "running"
        release.set()
        settled = await asyncio.wait_for(session.wait("observer-request"), timeout=5)
        assert settled.status == "completed"
        assert admitted.run_id == settled.run_id


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["local", "http"])
async def test_reconnect_snapshot_and_events_cover_run_after_observer_closes(kind):
    entered, release = asyncio.Event(), asyncio.Event()

    async def answer(**_kwargs):
        entered.set()
        await release.wait()
        return _text("completed after reconnect")

    agent = _agent(answer=answer)
    async with _session(kind, agent, "reconnect-thread") as session:
        first_context = session.watch()
        first = await first_context.__aenter__()
        try:
            admitted = await session.prompt("continue", request_id="reconnect-request")
            first_event = await asyncio.wait_for(anext(first), timeout=5)
            assert first_event.run_id == admitted.run_id
            await asyncio.wait_for(entered.wait(), timeout=5)
        finally:
            await first_context.__aexit__(None, None, None)

        second_context = session.watch()
        second = await second_context.__aenter__()
        try:
            snapshot = second.snapshot
            assert isinstance(snapshot, SnapshotRecord)
            assert snapshot.thread_id == session.thread_id
            assert any(
                isinstance(run, dict) and run.get("run_id") == admitted.run_id
                for run in snapshot.active_runs
            )
            release.set()
            events = []
            while True:
                event = await asyncio.wait_for(anext(second), timeout=5)
                events.append(event)
                if event.run_id == admitted.run_id and event.type == "run.end":
                    break
        finally:
            await second_context.__aexit__(None, None, None)

        assert events
        assert all(isinstance(event, EventRecord) for event in events)
        assert all(event.run_id == admitted.run_id for event in events)
        assert events[-1].type == "run.end"
        assert (await session.wait("reconnect-request")).status == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["local", "http"])
async def test_read_only_workspace_errors_are_delivered_and_do_not_write(
    tmp_path, kind
):
    root = tmp_path / kind / "workspace"
    root.mkdir(parents=True)
    workspace = AgentWorkspace.local(root, read_only=True)
    requests = 0
    model_feedback = []

    async def answer(**kwargs):
        nonlocal requests
        requests += 1
        if requests == 1:
            return _tool("write", {"path": "blocked.txt", "content": "no"})
        model_feedback.extend(kwargs["messages"].to_chatml())
        return _text("write was blocked")

    agent = _agent(answer=answer, tools=[WriteTool()], workspace=workspace)
    try:
        async with _session(kind, agent, "readonly-thread") as session:
            watch_context = session.watch()
            watcher = await watch_context.__aenter__()
            try:
                admitted = await session.prompt("write a file", request_id="readonly")
                events = []
                while True:
                    event = await asyncio.wait_for(anext(watcher), timeout=5)
                    events.append(event)
                    if event.run_id == admitted.run_id and event.type == "run.end":
                        break
            finally:
                await watch_context.__aexit__(None, None, None)
            settled = await asyncio.wait_for(session.wait("readonly"), timeout=5)
            assert settled.status == "completed"
            assert admitted.run_id == settled.run_id
            assert not (root / "blocked.txt").exists()
            assert agent.generator.aforward.await_count == 2
            assert "workspace is read-only" in str(model_feedback).lower()
            tool_end = next(event for event in events if event.type == "tool.end")
            assert all(isinstance(event, EventRecord) for event in events)
            assert "workspace is read-only" in tool_end.data["error"].lower()
    finally:
        await workspace.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["local", "http"])
async def test_resuming_terminal_run_returns_its_receipt_without_a_new_execution(kind):
    agent = _agent(answer=lambda **_kwargs: _text("finished"))
    async with _session(kind, agent, "terminal-resume-thread") as session:
        admitted = await session.prompt("finish", request_id="terminal-request")
        settled = await asyncio.wait_for(session.wait("terminal-request"), timeout=5)
        resumed = await session.resume(admitted.run_id)

        assert settled.status == resumed.status == "completed"
        assert resumed.run_id == settled.run_id == admitted.run_id
        assert await session.receipt("terminal-request") == resumed
        assert agent.generator.aforward.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["local", "http"])
async def test_invalid_run_and_unknown_request_fail_without_creating_managed_state(
    tmp_path, kind
):
    agent_dir = tmp_path / kind / "managed-agent"
    agent_dir.mkdir(parents=True)
    agent = _agent(answer=lambda **_kwargs: _text("unused"), agent_dir=agent_dir)
    async with _session(kind, agent, "invalid-id-thread") as session:
        if kind == "local":
            with pytest.raises(ValueError):
                await session.resume(" ")
        else:
            with pytest.raises(AgentServiceHTTPError) as invalid_run_error:
                await session.resume(" ")
            assert invalid_run_error.value.status_code == 422
        with pytest.raises(ServiceRecoveryRequiredError):
            await session.resume("unknown-run")

        if kind == "local":
            with pytest.raises(KeyError):
                await session.receipt("unknown-request")
            with pytest.raises(KeyError):
                await session.wait("unknown-request")
        else:
            with pytest.raises(AgentServiceHTTPError) as receipt_error:
                await session.receipt("unknown-request")
            assert receipt_error.value.status_code == 404
            with pytest.raises(AgentServiceHTTPError) as wait_error:
                await session.wait("unknown-request", poll_interval=0.01)
            assert wait_error.value.status_code == 404

        assert not (agent_dir / "threads").exists()
        assert not (agent_dir / "runtime").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["local", "http"])
async def test_cancelling_waiter_does_not_cancel_service_owned_execution(kind):
    entered, release = asyncio.Event(), asyncio.Event()

    async def answer(**_kwargs):
        entered.set()
        await release.wait()
        return _text("producer completed")

    agent = _agent(answer=answer)
    async with _session(kind, agent, "wait-cancel-thread") as session:
        admitted = await session.prompt("keep working", request_id="wait-cancel")
        await asyncio.wait_for(entered.wait(), timeout=5)
        waiter = asyncio.create_task(session.wait("wait-cancel"))
        await asyncio.sleep(0)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        assert (await session.receipt("wait-cancel")).status == "running"
        release.set()
        settled = await asyncio.wait_for(session.wait("wait-cancel"), timeout=5)
        assert settled.status == "completed"
        assert settled.run_id == admitted.run_id
        assert agent.generator.aforward.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["local", "http"])
async def test_foreign_owned_wait_has_bounded_known_recovery_behavior(kind):
    agent = _agent(answer=lambda **_kwargs: _text("unused"))
    async with _session(
        kind,
        agent,
        "foreign-wait-thread",
        foreign_request="foreign-request",
    ) as session:
        with pytest.raises(ServiceRecoveryRequiredError):
            if kind == "local":
                await session.wait("foreign-request")
            else:
                # Keep the public 50 ms poll interval; the owner check is
                # returned immediately by the receipt endpoint's wait mode.
                await session.wait("foreign-request")
        assert (await session.receipt("foreign-request")).status == "running"
        assert agent.generator.aforward.await_count == 0


@pytest.mark.asyncio
async def test_local_watcher_conversion_failure_closes_observer_not_producer():
    entered, release = asyncio.Event(), asyncio.Event()

    async def answer(**_kwargs):
        entered.set()
        await release.wait()
        return _text("producer survived conversion error")

    agent = _agent(answer=answer)
    async with _session("local", agent, "conversion-failure-thread") as session:
        watch_context = session.watch()
        watcher = await watch_context.__aenter__()
        try:
            admitted = await session.prompt(
                "stay alive", request_id="conversion-failure"
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            _hub_event_sink().emit(
                "test.unsupported_payload",
                {"payload": object()},
                scope=ExecutionScope(
                    thread_id=session.thread_id,
                    namespace=session.namespace,
                    run_id=admitted.run_id,
                ),
            )

            with pytest.raises(TypeError):
                while True:
                    await asyncio.wait_for(anext(watcher), timeout=5)

            # Conversion failure closes the wrapped observer, and repeated
            # closes (including the context manager exit) remain harmless.
            await watcher.aclose()
            await watcher.aclose()
            with pytest.raises(StopAsyncIteration):
                await anext(watcher)
            assert (await session.receipt("conversion-failure")).status == "running"

            release.set()
            settled = await asyncio.wait_for(
                session.wait("conversion-failure"), timeout=5
            )
            assert settled.status == "completed"
            assert settled.run_id == admitted.run_id
            assert agent.generator.aforward.await_count == 1
        finally:
            release.set()
            await watch_context.__aexit__(None, None, None)
