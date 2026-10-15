"""Controlled provider and SQLite failures through real service HTTP sockets."""

import asyncio
from contextlib import closing
import sqlite3

import httpx2
import msgspec
import pytest

pytest.importorskip("litestar")
pytest.importorskip("uvicorn")

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.models.chat_transport import HTTPChatTransport
from msgflux.models.providers.openai import OpenAIChatCompletion
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime.service import AgentSession, ServiceRecoveryRequiredError
from msgflux.runtime.service.http import AgentServiceClient, AgentSessionClient
from tests.models._stream_disconnect import partial_text, tool_call
from tests.test_agent_service_http_integration import (
    TOKEN,
    _agent,
    _response,
    _root_terminal,
    _server,
)


class _DroppingStream(httpx2.AsyncByteStream):
    def __init__(self, api_mode, incomplete_tool):
        self.api_mode = api_mode
        self.incomplete_tool = incomplete_tool
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield partial_text(self.api_mode)
        if self.incomplete_tool:
            yield tool_call(self.api_mode, complete=False)
        self.entered.set()
        await self.release.wait()
        raise httpx2.ReadError("synthetic upstream connection dropped")

    async def aclose(self):
        self.closed = True


def _check_integrity(*paths):
    for path in paths:
        with closing(sqlite3.connect(path)) as connection:
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@pytest.mark.parametrize("api_mode", ["chat_completions", "responses"])
@pytest.mark.parametrize(
    "after_tool,incomplete_tool", [(False, False), (True, False), (False, True)]
)
@pytest.mark.asyncio
async def test_upstream_stream_drop_preserves_history_without_replaying_tool(
    tmp_path, monkeypatch, api_mode, after_tool, incomplete_tool
):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-fault-test-key")
    body = _DroppingStream(api_mode, incomplete_tool)
    sibling_entered, sibling_release = asyncio.Event(), asyncio.Event()
    requests = []
    effects = tmp_path / "effects.log"
    journal_path = tmp_path / "service.sqlite3"
    checkpoint_path = tmp_path / "broken.sqlite3"

    def record_effect() -> str:
        """Append one synthetic externally visible effect."""
        with effects.open("a", encoding="utf-8") as handle:
            handle.write("one\n")
        return "EFFECT_RECORDED"

    def handler(request):
        requests.append(msgspec.json.decode(request.content))
        if after_tool and len(requests) == 1:
            return httpx2.Response(200, content=tool_call(api_mode, complete=True))
        return httpx2.Response(200, stream=body)

    async def healthy_answer(**_kwargs):
        sibling_entered.set()
        await sibling_release.wait()
        return _response("healthy sibling")

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as upstream:
        model = OpenAIChatCompletion(
            model_id="gpt-4.1-mini",
            api_mode=api_mode,
            chat_transport=HTTPChatTransport(async_client=upstream, max_retries=0),
            retry=False,
        )

        def factory(thread):
            checkpoints = SQLiteCheckpointStore(
                tmp_path / f"{thread.thread_id}.sqlite3"
            )
            agent = (
                Agent(
                    name="main",
                    model=model,
                    tools=[record_effect],
                    checkpoint_store=checkpoints,
                    config={"stream": True},
                )
                if thread.thread_id == "broken"
                else _agent(healthy_answer, checkpoints=checkpoints)
            )
            return AgentSession(agent, on_close=checkpoints.close)

        try:
            async with _server(factory, journal_path=journal_path) as (_service, url):
                client = AgentServiceClient(url, token=TOKEN)
                try:
                    broken = await AgentSessionClient.open(
                        client, thread_id="broken", agent_id="main"
                    )
                    healthy = await AgentSessionClient.open(
                        client, thread_id="healthy", agent_id="main"
                    )
                    sibling = await healthy.prompt(
                        "continue independently", request_id="healthy"
                    )
                    await asyncio.wait_for(sibling_entered.wait(), 3)
                    async with broken.watch() as watcher:
                        receipt = await broken.prompt(
                            "record then answer", request_id="drop"
                        )
                        drain = asyncio.create_task(
                            _root_terminal(watcher, receipt.run_id)
                        )
                        try:
                            await asyncio.wait_for(body.entered.wait(), 5)
                            assert effects.exists() == after_tool
                            body.release.set()
                            events = await asyncio.wait_for(drain, 5)
                        finally:
                            if not drain.done():
                                drain.cancel()
                            await asyncio.gather(drain, return_exceptions=True)
                    settled = await asyncio.wait_for(broken.wait("drop"), 3)
                    assert settled.status == "failed"
                    assert "synthetic upstream connection dropped" in settled.error
                    assert events[-1].type == "run.error"
                    expected_delta = (
                        "commentary.delta" if incomplete_tool else "message.delta"
                    )
                    assert any(
                        event.type == expected_delta
                        and "partial-before-drop" in str(event.data)
                        for event in events
                    ), [(event.type, event.data) for event in events]
                    snapshot = await broken.snapshot()
                    assert "partial-before-drop" in str(snapshot.messages)
                    result = await client.run_result("broken", receipt.run_id)
                    assert result.status == "failed" and result.output is None
                    assert result.error == settled.error
                    duplicate = await broken.prompt(
                        "record then answer", request_id="drop"
                    )
                    assert (
                        duplicate.run_id == receipt.run_id
                        and duplicate.status == "failed"
                    )
                    assert len(requests) == (2 if after_tool else 1)
                    if after_tool:
                        assert effects.read_text() == "one\n"
                        sent = requests[-1]
                        history = (
                            sent["input"]
                            if api_mode == "responses"
                            else sent["messages"]
                        )
                        assert "EFFECT_RECORDED" in str(history)
                    assert (await healthy.receipt("healthy")).status == "running"
                    sibling_release.set()
                    assert (
                        await asyncio.wait_for(healthy.wait(sibling.request_id), 3)
                    ).status == "completed"
                    assert await client.active_runs() == ()
                finally:
                    body.release.set()
                    sibling_release.set()
                    await client.aclose()
            # Reopening the journal and checkpoints must not resend a failed input.
            async with _server(factory, journal_path=journal_path) as (_service, url):
                client = AgentServiceClient(url, token=TOKEN)
                try:
                    broken = await AgentSessionClient.open(
                        client, thread_id="broken", agent_id="main"
                    )
                    duplicate = await broken.prompt(
                        "record then answer", request_id="drop"
                    )
                    assert (
                        duplicate.run_id == receipt.run_id
                        and duplicate.status == "failed"
                    )
                    assert "partial-before-drop" in str(
                        (await broken.snapshot()).messages
                    )
                    assert len(requests) == (2 if after_tool else 1)
                    if after_tool:
                        assert effects.read_text() == "one\n"
                finally:
                    await client.aclose()
            assert body.closed
        finally:
            body.release.set()
            sibling_release.set()
            await model.aclose()
    _check_integrity(journal_path, checkpoint_path, tmp_path / "healthy.sqlite3")


def _tool_response():
    aggregate = ToolCallAggregator()
    aggregate.process(0, "effect-call", "record_effect", "{}")
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(aggregate)
    return response


@pytest.mark.parametrize(
    "error_type,message",
    [
        (sqlite3.OperationalError, "database or disk is full"),
        (PermissionError, "checkpoint directory is not writable"),
    ],
)
@pytest.mark.asyncio
async def test_terminal_checkpoint_write_failure_is_not_success_and_keeps_sibling_usable(
    tmp_path, monkeypatch, error_type, message
):
    effects = tmp_path / "effects.log"
    answers = {}

    def record_effect() -> str:
        """Record a synthetic effect before the terminal checkpoint write."""
        with effects.open("a", encoding="utf-8") as handle:
            handle.write("one\n")
        return "effect recorded"

    def factory(thread):
        checkpoints = SQLiteCheckpointStore(tmp_path / f"{thread.thread_id}.sqlite3")
        if thread.thread_id == "broken":
            commit = checkpoints.commit_state

            def fail_terminal(namespace, thread_id, run_id, state, **kwargs):
                if state.get("status") == "completed":
                    raise error_type(message)
                return commit(namespace, thread_id, run_id, state, **kwargs)

            monkeypatch.setattr(checkpoints, "commit_state", fail_terminal)
            responses = iter([_tool_response(), _response("not durably completed")])
            answer = lambda **_kwargs: next(responses)
        else:
            answer = lambda **_kwargs: _response("healthy answer")
        agent = _agent(answer, tools=[record_effect], checkpoints=checkpoints)
        answers[thread.thread_id] = agent.generator.aforward
        return AgentSession(agent, on_close=checkpoints.close)

    journal_path = tmp_path / "service.sqlite3"
    async with _server(factory, journal_path=journal_path) as (_service, url):
        client = AgentServiceClient(url, token=TOKEN)
        try:
            broken = await AgentSessionClient.open(
                client, thread_id="broken", agent_id="main"
            )
            async with broken.watch() as watcher:
                receipt = await broken.prompt(
                    "record one effect", request_id="write-failure"
                )
                events = await asyncio.wait_for(
                    _root_terminal(watcher, receipt.run_id), 5
                )
            failed = await asyncio.wait_for(broken.wait(receipt.request_id), 3)
            assert failed.status == "failed" and message in failed.error
            assert events[-1].type == "run.error"
            assert effects.read_text() == "one\n"
            result = await client.run_result("broken", receipt.run_id)
            assert result.status == "failed" and result.output is None
            inspection = await broken.inspect_run(receipt.run_id)
            assert inspection.checkpoint_status != "completed"
            assert any(message in reason for reason in inspection.reasons)
            duplicate = await broken.prompt(
                "record one effect", request_id="write-failure"
            )
            assert duplicate.run_id == receipt.run_id and duplicate.status == "failed"
            assert answers["broken"].await_count == 2
            healthy = await AgentSessionClient.open(
                client, thread_id="healthy", agent_id="main"
            )
            healthy_receipt = await healthy.prompt("keep working", request_id="healthy")
            assert (
                await asyncio.wait_for(healthy.wait(healthy_receipt.request_id), 3)
            ).status == "completed"
            assert await client.active_runs() == ()
        finally:
            await client.aclose()
    _check_integrity(
        journal_path, tmp_path / "broken.sqlite3", tmp_path / "healthy.sqlite3"
    )


@pytest.mark.asyncio
async def test_service_receipt_write_failure_requires_recovery_without_replaying_input(
    tmp_path, monkeypatch
):
    agents = {}

    def factory(thread):
        checkpoints = SQLiteCheckpointStore(tmp_path / f"{thread.thread_id}.sqlite3")
        agent = _agent(
            lambda **_kwargs: _response("checkpointed answer"), checkpoints=checkpoints
        )
        agents[thread.thread_id] = agent
        return AgentSession(agent, on_close=checkpoints.close)

    journal_path = tmp_path / "service.sqlite3"
    async with _server(factory, journal_path=journal_path) as (service, url):
        finish = service.store.finish

        def fail_finish(receipt, *args, **kwargs):
            if receipt.thread_id == "broken":
                raise sqlite3.OperationalError(
                    "synthetic service journal write failure"
                )
            return finish(receipt, *args, **kwargs)

        monkeypatch.setattr(service.store, "finish", fail_finish)
        client = AgentServiceClient(url, token=TOKEN)
        try:
            broken = await AgentSessionClient.open(
                client, thread_id="broken", agent_id="main"
            )
            receipt = await broken.prompt("save once", request_id="journal-failure")
            with pytest.raises(
                sqlite3.OperationalError, match="service journal write failure"
            ):
                await asyncio.wait_for(service.wait("broken", receipt.request_id), 3)
            with pytest.raises(ServiceRecoveryRequiredError, match="does not own"):
                await asyncio.wait_for(broken.wait(receipt.request_id), 3)
            saved = await broken.receipt(receipt.request_id)
            assert saved.status == "running"
            inspection = await broken.inspect_run(receipt.run_id)
            assert (
                not inspection.local_worker
                and inspection.checkpoint_status == "completed"
            )
            assert "Admission and checkpoint statuses differ." in inspection.reasons
            assert (await client.run_result("broken", receipt.run_id)).output is None
            duplicate = await broken.prompt("save once", request_id="journal-failure")
            assert duplicate.run_id == receipt.run_id
            assert agents["broken"].generator.aforward.await_count == 1
            healthy = await AgentSessionClient.open(
                client, thread_id="healthy", agent_id="main"
            )
            sibling = await healthy.prompt("still available", request_id="healthy")
            assert (
                await asyncio.wait_for(healthy.wait(sibling.request_id), 3)
            ).status == "completed"
            assert await client.active_runs() == ()
        finally:
            await client.aclose()
    _check_integrity(
        journal_path, tmp_path / "broken.sqlite3", tmp_path / "healthy.sqlite3"
    )
