"""Real socket integration for service lifetime and frontend reconnection."""

import asyncio
import gc
import os
import socket
import weakref
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest

pytest.importorskip("litestar")
uvicorn = pytest.importorskip("uvicorn")

from msgflux.data.stores import InMemoryCheckpointStore, SQLiteCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.runtime import (
    AgentApprovals,
    AgentWorkspace,
    DockerWorkspaceBackend,
    InMemoryApprovalStore,
    PermissionSet,
)
from msgflux.runtime.context import get_execution_scope
from msgflux.runtime.event_hub import get_event_hub
from msgflux.runtime.service import AgentService, AgentSession, SQLiteServiceStore
from msgflux.tools.builtin import BashTool, ReadFileTool
from msgflux.runtime.service.http import (
    AgentServiceClient,
    AgentServiceHTTPError,
    create_service_app,
)

TOKEN = "socket-integration-token"


def _response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


def _agent(
    answer, *, name="main", tools=(), checkpoints=None, approvals=None, agent_dir=None
):
    model = Mock()
    model.model_type = "chat_completion"
    if checkpoints is None and agent_dir is None:
        checkpoints = InMemoryCheckpointStore()
    agent = Agent(
        name=name,
        model=model,
        tools=list(tools),
        checkpoint_store=checkpoints,
        approvals=approvals,
        agent_dir=agent_dir,
    )
    agent.generator.aforward = AsyncMock(side_effect=answer)
    return agent


@asynccontextmanager
async def _server(factory, *, journal_path=":memory:", event_buffer_limit=1024):
    journal = SQLiteServiceStore(journal_path)
    service = AgentService(store=journal)
    service.register("main", factory)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = listener.getsockname()[1]
    config = uvicorn.Config(
        create_service_app(service, token=TOKEN, event_buffer_limit=event_buffer_limit),
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
        ws="none",
        lifespan="on",
        timeout_graceful_shutdown=2,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[listener]))

    async def started():
        while not server.started:
            if task.done():
                await task
                raise RuntimeError("Server exited before startup")
            await asyncio.sleep(0.01)

    try:
        await asyncio.wait_for(started(), 5)
        yield service, f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 5)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            listener.close()
            await service.aclose()
            journal.close()


async def _root_terminal(watcher, run_id):
    events = []
    async for event in watcher:
        events.append(event)
        if (
            event.run_id == run_id
            and len(event.source_path) == 1
            and event.type in {"run.end", "run.error", "run.paused", "run.interrupted"}
        ):
            return events
    raise AssertionError("Connection ended before the run settled")


async def _no_watchers(thread_id):
    while get_event_hub()._watchers.get(thread_id):
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_two_frontends_detach_reconnect_and_deduplicate_over_real_tcp():
    entered, release = asyncio.Event(), asyncio.Event()

    async def answer(**_kwargs):
        entered.set()
        await release.wait()
        return _response("Olá — 日本語 — 🌋")

    agent = _agent(answer)
    async with _server(lambda _thread: AgentSession(agent)) as (service, url):
        first = AgentServiceClient(url, token=TOKEN)
        second = AgentServiceClient(url, token=TOKEN)
        try:
            thread = await first.open_thread("main", thread_id="socket-two-fronts")
            assert await first.agents() == ("main",)
            assert thread in await second.threads()
            async with second.watch(thread.thread_id) as remaining:
                assert remaining.snapshot.thread_id == thread.thread_id
                async with first.watch(thread.thread_id) as detached:
                    receipt = await first.prompt(
                        thread.thread_id, "hello", request_id="same"
                    )
                    event = await asyncio.wait_for(anext(detached), 3)
                    assert event.type == "run.start"
                    await asyncio.wait_for(entered.wait(), 3)
                assert not release.is_set()
                assert (
                    await second.receipt(thread.thread_id, "same")
                ).status == "running"
                release.set()
                events = await asyncio.wait_for(
                    _root_terminal(remaining, receipt.run_id), 3
                )
                assert any(
                    event.type == "message.end" and "日本語" in str(event.data)
                    for event in events
                )
                assert (
                    await service.wait(thread.thread_id, "same")
                ).status == "completed"
            async with first.watch(thread.thread_id) as reattached:
                assert any("Olá" in str(item) for item in reattached.snapshot.messages)
                assert reattached.snapshot.active_runs == ()
            duplicate = await second.prompt(
                thread.thread_id, "hello", request_id="same"
            )
            assert duplicate.run_id == receipt.run_id
            assert duplicate.status == "completed"
            assert agent.generator.aforward.await_count == 1
            await asyncio.wait_for(_no_watchers(thread.thread_id), 3)
        finally:
            release.set()
            await first.aclose()
            await second.aclose()


@pytest.mark.asyncio
async def test_tcp_sse_watcher_spans_managed_release_and_reload(tmp_path):
    agents = []
    observed_history = []
    agent_dir = tmp_path / "tcp-watch-reload-agent"

    def factory(_thread):
        generation = len(agents)

        async def answer(**kwargs):
            if generation:
                observed_history.extend(kwargs["messages"].to_chatml())
            return _response(f"socket-answer-{generation}")

        agent = _agent(answer, name="main", agent_dir=agent_dir)
        agents.append(agent)
        return AgentSession(agent)

    async with _server(factory) as (service, url):
        client = AgentServiceClient(url, token=TOKEN)
        watcher_context = None
        try:
            thread = await client.open_thread("main", thread_id="tcp-watch-reload")
            first = await client.prompt(
                thread.thread_id, "remember silver pine", request_id="one"
            )
            assert (
                await service.wait(thread.thread_id, first.request_id)
            ).status == "completed"
            old_agent = agents[0]
            old_agent_ref = weakref.ref(old_agent)
            old_resources = old_agent._owned_threads[thread.thread_id].resources

            context = client.watch(thread.thread_id)
            watcher = await context.__aenter__()
            watcher_context = context
            assert any(
                item.get("role") == "user"
                and "remember silver pine" in item.get("content", "")
                for item in watcher.snapshot.messages
            )
            assert await service.release_session(thread.thread_id) is True
            assert old_resources._closed
            agents[0] = None
            del old_agent
            gc.collect()
            assert old_agent_ref() is None

            second = await client.prompt(
                thread.thread_id, "what did I ask?", request_id="two"
            )
            assert (
                await service.wait(thread.thread_id, second.request_id)
            ).status == "completed"
            assert await service.release_session(thread.thread_id) is True
            assert any(
                item.get("role") == "user"
                and "remember silver pine" in item.get("content", "")
                for item in observed_history
            )

            events = await asyncio.wait_for(_root_terminal(watcher, second.run_id), 3)
            assert any(
                event.type == "run.end" and event.run_id == second.run_id
                for event in events
            )
            await watcher_context.__aexit__(None, None, None)
            watcher_context = None
            await asyncio.wait_for(_no_watchers(thread.thread_id), 3)
        finally:
            if watcher_context is not None:
                await watcher_context.__aexit__(None, None, None)
            await client.aclose()


@pytest.mark.asyncio
async def test_remote_interrupt_targets_one_thread_and_preserves_other_run():
    entered = {key: asyncio.Event() for key in ("one", "two")}
    release = asyncio.Event()

    def factory(binding):
        thread_id = binding.thread_id

        async def answer(**_kwargs):
            entered[thread_id].set()
            if thread_id == "one":
                signal = get_execution_scope().abort_signal
                await signal.wait()
                signal.raise_if_aborted()
            else:
                await release.wait()
            return _response("done")

        return AgentSession(_agent(answer))

    async with _server(factory) as (service, url):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            for thread in entered:
                await client.open_thread("main", thread_id=thread)
            first = await client.prompt("one", "first", request_id="one")
            second = await client.prompt("two", "second", request_id="two")
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in entered.values())), 3
            )
            assert await client.interrupt("two", first.run_id) is False
            assert await client.interrupt("one", first.run_id) is True
            assert (await service.wait("one", "one")).status == "interrupted"
            assert (await client.receipt("two", "two")).status == "running"
            release.set()
            assert (await service.wait("two", "two")).status == "completed"
            assert (await client.receipt("two", "two")).run_id == second.run_id
        finally:
            release.set()
            await client.aclose()


@pytest.mark.asyncio
async def test_remote_steering_reaches_next_model_request_after_tool():
    entered, release = asyncio.Event(), asyncio.Event()
    history = []

    async def lookup(query: str):
        entered.set()
        await release.wait()
        return query

    calls = ToolCallAggregator()
    calls.process(0, "lookup-1", "lookup", '{"query":"value"}')
    tool_response = ModelResponse()
    tool_response.set_response_type("tool_call")
    tool_response.add(calls)

    async def answer(**kwargs):
        history.append(kwargs["messages"].to_chatml())
        return tool_response if len(history) == 1 else _response("steered")

    agent = _agent(answer, tools=[lookup])
    async with _server(lambda _thread: AgentSession(agent)) as (service, url):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            thread = await client.open_thread("main")
            receipt = await client.prompt(
                thread.thread_id, "look up", request_id="steer"
            )
            await asyncio.wait_for(entered.wait(), 3)
            notice = await client.steer(
                thread.thread_id, receipt.run_id, "Use the existing tests"
            )
            assert notice["notification_id"]
            release.set()
            assert (await service.wait(thread.thread_id, "steer")).status == "completed"
            assert "Use the existing tests" in str(history[-1])
        finally:
            release.set()
            await client.aclose()


@pytest.mark.asyncio
async def test_real_stream_errors_and_failed_provider_remain_observable():
    async def failure(**_kwargs):
        raise RuntimeError("offline provider")

    agent = _agent(failure)
    async with _server(lambda _thread: AgentSession(agent)) as (service, url):
        client = AgentServiceClient(url, token=TOKEN)
        invalid = AgentServiceClient(url, token="wrong")
        try:
            with pytest.raises(AgentServiceHTTPError) as error:
                async with invalid.watch("missing"):
                    pass
            assert error.value.status_code == 401
            with pytest.raises(AgentServiceHTTPError) as error:
                async with client.watch("missing"):
                    pass
            assert error.value.status_code == 404
            thread = await client.open_thread("main")
            async with client.watch(thread.thread_id) as observer:
                receipt = await client.prompt(
                    thread.thread_id, "fail", request_id="failed"
                )
                events = await asyncio.wait_for(
                    _root_terminal(observer, receipt.run_id), 3
                )
                assert events[-1].type == "run.error"
            assert (await service.wait(thread.thread_id, "failed")).status == "failed"
            assert (
                await client.receipt(thread.thread_id, "failed")
            ).error == "offline provider"
            await asyncio.wait_for(_no_watchers(thread.thread_id), 3)
        finally:
            await invalid.aclose()
            await client.aclose()


@pytest.mark.asyncio
async def test_remote_approval_pause_resumes_same_run_after_host_decision():
    approvals = InMemoryApprovalStore()
    invoked = []

    def change(value: str):
        invoked.append(value)
        return value

    calls = ToolCallAggregator()
    calls.process(0, "change-1", "change", '{"value":"approved"}')
    tool = ModelResponse()
    tool.set_response_type("tool_call")
    tool.add(calls)
    responses = iter([tool, _response("done")])
    agent = _agent(
        lambda **_kwargs: next(responses),
        tools=[change],
        approvals=AgentApprovals(approvals, {"change": "v1"}, "socket-policy"),
    )
    factory = lambda _thread: AgentSession(
        agent, scope_factory=lambda scope: scope.with_overrides(principal="socket-user")
    )
    async with _server(factory) as (service, url):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            thread = await client.open_thread("main")
            receipt = await client.prompt(
                thread.thread_id, "change", request_id="approval"
            )
            assert (await service.wait(thread.thread_id, "approval")).status == "paused"
            assert invoked == []
            snapshot = await client.snapshot(thread.thread_id)
            assert snapshot.approvals
            approval = approvals.pending("main", thread.thread_id, receipt.run_id)[0]
            await agent.adecide_approval(
                approval.request_id, approved=True, decided_by="socket-user"
            )
            async with client.watch(thread.thread_id) as observer:
                resumed = await client.resume_checkpoint(
                    thread.thread_id, receipt.run_id
                )
                assert resumed.run_id == receipt.run_id
                await asyncio.wait_for(_root_terminal(observer, receipt.run_id), 3)
            assert (
                await service.wait(thread.thread_id, "approval")
            ).status == "completed"
            assert invoked == ["approved"]
        finally:
            await client.aclose()


@pytest.mark.asyncio
async def test_server_restart_restores_history_and_receipt_without_repeating_input(
    tmp_path,
):
    journal_path = tmp_path / "service.sqlite3"
    checkpoint_path = tmp_path / "checkpoints.sqlite3"
    checkpoints = SQLiteCheckpointStore(str(checkpoint_path))
    first_agent = _agent(
        lambda **_kwargs: _response("saved cobalt"), checkpoints=checkpoints
    )
    async with _server(
        lambda _thread: AgentSession(first_agent), journal_path=journal_path
    ) as (service, url):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            thread = await client.open_thread("main", thread_id="persistent-http")
            original = await client.prompt(
                thread.thread_id, "remember cobalt", request_id="old"
            )
            assert (await service.wait(thread.thread_id, "old")).status == "completed"
        finally:
            await client.aclose()
    checkpoints.close()

    seen = []

    async def answer(**kwargs):
        seen.append(kwargs["messages"].to_chatml())
        return _response("cobalt")

    reopened = SQLiteCheckpointStore(str(checkpoint_path))
    second_agent = _agent(answer, checkpoints=reopened)
    try:
        async with _server(
            lambda _thread: AgentSession(second_agent), journal_path=journal_path
        ) as (service, url):
            client = AgentServiceClient(url, token=TOKEN)
            try:
                async with client.watch(thread.thread_id) as observer:
                    assert "saved cobalt" in str(observer.snapshot.messages)
                old = await client.receipt(thread.thread_id, "old")
                assert old.run_id == original.run_id and old.status == "completed"
                duplicate = await client.prompt(
                    thread.thread_id, "remember cobalt", request_id="old"
                )
                assert duplicate == old
                assert second_agent.generator.aforward.await_count == 0
                await client.prompt(thread.thread_id, "what word?", request_id="new")
                assert (
                    await service.wait(thread.thread_id, "new")
                ).status == "completed"
                assert "saved cobalt" in str(seen)
                assert second_agent.generator.aforward.await_count == 1
            finally:
                await client.aclose()
    finally:
        reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_kind", ["local", "docker"])
async def test_workspace_read_and_real_bash_results_are_portable_over_sse(
    tmp_path, backend_kind
):
    (tmp_path / "note.txt").write_text("workspace-value", encoding="utf-8")
    if backend_kind == "docker":
        image = os.environ.get("MSGFLUX_TEST_DOCKER_IMAGE")
        if not image:
            pytest.skip("Set MSGFLUX_TEST_DOCKER_IMAGE to a trusted local image ID")
        workspace = await AgentWorkspace.open(
            DockerWorkspaceBackend(tmp_path, image=image),
            "http-docker",
            permissions=PermissionSet(
                [
                    "process.execute",
                    "process.workspace",
                    "filesystem.read",
                ]
            ),
            write_guarantee="cooperative_compare",
        )
        expected_cwd = "/workspace"
    else:
        workspace = AgentWorkspace.local(tmp_path)
        expected_cwd = str(tmp_path)
    calls = ToolCallAggregator()
    calls.process(
        0, "read-file", "read", '{"path":"note.txt","offset":null,"limit":null}'
    )
    calls.process(
        1, "bash-cmd", "bash", '{"command":"pwd; cat note.txt","timeout_ms":1000}'
    )
    tool = ModelResponse()
    tool.set_response_type("tool_call")
    tool.add(calls)
    responses = iter([tool, _response("done")])
    agent = _agent(
        lambda **_kwargs: next(responses), tools=[ReadFileTool(), BashTool()]
    )
    agent.workspace = workspace
    factory = lambda _thread: AgentSession(agent)
    try:
        async with _server(factory) as (service, url):
            client = AgentServiceClient(url, token=TOKEN)
            try:
                thread = await client.open_thread("main")
                async with client.watch(thread.thread_id) as observer:
                    receipt = await client.prompt(
                        thread.thread_id, "inspect", request_id="workspace"
                    )
                    events = await asyncio.wait_for(
                        _root_terminal(observer, receipt.run_id), 5
                    )
                assert (
                    await service.wait(thread.thread_id, "workspace")
                ).status == "completed"
                tool_events = [event for event in events if event.type == "tool.end"]
                assert len(tool_events) == 2
                assert "workspace-value" in str(tool_events)
                shell = next(
                    event for event in tool_events if event.data["tool_name"] == "bash"
                )
                assert isinstance(shell.data["result"], dict)
                assert expected_cwd in str(shell.data["result"])
            finally:
                await client.aclose()
    finally:
        await workspace.aclose()


@pytest.mark.asyncio
async def test_observer_overflow_reports_reconnect_without_canceling_run():
    agent = _agent(lambda **_kwargs: _response("buffered answer"))
    async with _server(lambda _thread: AgentSession(agent), event_buffer_limit=1) as (
        service,
        url,
    ):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            thread = await client.open_thread("main")
            async with client.watch(thread.thread_id) as observer:
                await client.prompt(thread.thread_id, "hello", request_id="overflow")
                with pytest.raises(AgentServiceHTTPError, match="reconnect"):
                    await asyncio.wait_for(_root_terminal(observer, "any"), 3)
            assert (
                await service.wait(thread.thread_id, "overflow")
            ).status == "completed"
            async with client.watch(thread.thread_id) as reattached:
                assert "buffered answer" in str(reattached.snapshot.messages)
            await asyncio.wait_for(_no_watchers(thread.thread_id), 3)
        finally:
            await client.aclose()
