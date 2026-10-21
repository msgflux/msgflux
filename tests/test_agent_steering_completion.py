"""Agent steering is delivered before ordinary terminal completion."""

import asyncio
import sqlite3
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.data.stores import SQLiteCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse, ModelStreamResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn.extensions import ToolTurnLimitExtension
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import NotificationContext
from msgflux.nn.modules.agent import Agent
from msgflux.runtime.agent_inbox import (
    AgentInbox,
    AgentInboxClosedError,
    SQLiteAgentInboxStore,
)
from msgflux.runtime.context import ExecutionScope, get_execution_context
from msgflux.runtime.events import EventType
from msgflux.tools.config import tool_config


def _text_response(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    return response


def _tool_response(name, call_id):
    calls = ToolCallAggregator()
    calls.process(0, call_id, name, "{}")
    response = ModelResponse()
    response.set_response_type("tool_call")
    response.add(calls)
    return response


def _scope(thread_id="thread", run_id="run"):
    return ExecutionScope(
        namespace="steering_agent", thread_id=thread_id, run_id=run_id
    )


def _agent(*, checkpoint_store, inbox_store, hooks=(), **kwargs):
    return Agent(
        name="steering_agent",
        model=Mock(model_type="chat_completion"),
        checkpoint_store=checkpoint_store,
        agent_inbox=AgentInbox(store=inbox_store),
        hooks=list(hooks),
        **kwargs,
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_user_steering_continues_same_run_and_commits_both_answers(
    tmp_path, asynchronous
):
    checkpoints = SQLiteCheckpointStore(tmp_path / f"checkpoints-{asynchronous}.sqlite")
    inbox_store = SQLiteAgentInboxStore(tmp_path / f"inbox-{asynchronous}.sqlite")
    requests = []
    before_end, after_end = [], []

    def publish_then_answer(**kwargs):
        requests.append(kwargs["messages"].to_chatml())
        if len(requests) == 1:
            get_execution_context()["agent_inbox"].user_message("the added requirement")
            return _text_response("first answer")
        return _text_response("newest answer")

    async def apublish_then_answer(**kwargs):
        return publish_then_answer(**kwargs)

    def before(context):
        before_end.append(context.outcome)
        return context

    def after(context):
        after_end.append(context.outcome)
        return context

    agent = _agent(
        checkpoint_store=checkpoints,
        inbox_store=inbox_store,
        hooks=[
            Hook(event="before_run_end", handler=before),
            Hook(event="after_run_end", handler=after),
        ],
    )
    if asynchronous:
        agent.generator.aforward = AsyncMock(side_effect=apublish_then_answer)
    else:
        agent.generator.forward = Mock(side_effect=publish_then_answer)

    try:
        if asynchronous:
            result = asyncio.run(agent.acall("start", scope=_scope()))
        else:
            result = agent("start", scope=_scope())

        assert result == "newest answer"
        assert len(requests) == 2
        assert str(requests[1]).count("the added requirement") == 1
        state = checkpoints.load_state("steering_agent", "thread", "run")
        assert state["status"] == "completed"
        items = state["messages"]["items"]
        assert sum(item.get("content") == "first answer" for item in items) == 1
        assert sum(item.get("content") == "newest answer" for item in items) == 1
        assert str(items).count("the added requirement") == 1
        assert before_end == after_end == ["completed"]
        assert inbox_store.load_notifications("steering_agent", "thread", "run") == []
    finally:
        checkpoints.close()
        inbox_store.close()


@pytest.mark.asyncio
async def test_task_notice_does_not_start_an_extra_model_request(tmp_path):
    checkpoints = SQLiteCheckpointStore(tmp_path / "task-checkpoints.sqlite")
    inbox_store = SQLiteAgentInboxStore(tmp_path / "task-inbox.sqlite")
    requests = []

    async def answer(**kwargs):
        requests.append(kwargs["messages"].to_chatml())
        get_execution_context()["agent_inbox"].publish(
            {"source": "task", "ref": "task-1", "status": "completed"}
        )
        return _text_response("finished")

    agent = _agent(checkpoint_store=checkpoints, inbox_store=inbox_store)
    agent.generator.aforward = AsyncMock(side_effect=answer)
    try:
        assert await agent.acall("start", scope=_scope()) == "finished"
        assert len(requests) == 1
        assert agent.generator.aforward.await_count == 1
    finally:
        checkpoints.close()
        inbox_store.close()


@pytest.mark.parametrize("terminal", ["return_direct", "tool_budget"])
@pytest.mark.asyncio
async def test_terminal_tool_policies_do_not_continue_for_steering(tmp_path, terminal):
    checkpoints = SQLiteCheckpointStore(tmp_path / f"{terminal}-checkpoints.sqlite")
    inbox_store = SQLiteAgentInboxStore(tmp_path / f"{terminal}-inbox.sqlite")
    tool_calls = []

    async def execute():
        """Complete the selected terminal tool path."""
        tool_calls.append("called")
        get_execution_context()["agent_inbox"].user_message("accepted before seal")
        return "direct result"

    if terminal == "return_direct":
        tool = tool_config(return_direct=True)(execute)
        extensions = ()
        config = {}
    else:
        tool = execute
        extensions = (ToolTurnLimitExtension(limit=1),)
        config = {}

    agent = _agent(
        checkpoint_store=checkpoints,
        inbox_store=inbox_store,
        tools=[tool],
        extensions=extensions,
        config=config,
    )
    agent.generator.aforward = AsyncMock(
        side_effect=[_tool_response(tool.__name__, f"call-{terminal}")]
    )
    try:
        result = await agent.acall("start", scope=_scope())
        assert tool_calls == ["called"]
        assert agent.generator.aforward.await_count == 1
        state = checkpoints.load_state("steering_agent", "thread", "run")
        assert state["status"] == "completed"
        pending = inbox_store.load_notifications("steering_agent", "thread", "run")
        assert len(pending) == 1
        assert pending[0]["metadata"]["content"] == "accepted before seal"
        if terminal == "return_direct":
            assert result["tool_responses"]["tool_calls"][0]["result"] == (
                "direct result"
            )
        else:
            assert result["stop_reason"] == "tool_turn_limit"
    finally:
        checkpoints.close()
        inbox_store.close()


@pytest.mark.asyncio
async def test_filtered_user_steering_pauses_without_redraining_forever(tmp_path):
    checkpoints = SQLiteCheckpointStore(tmp_path / "filtered-checkpoints.sqlite")
    inbox_store = SQLiteAgentInboxStore(tmp_path / "filtered-inbox.sqlite")
    filtered = []

    def discard_user_messages(context):
        assert isinstance(context, NotificationContext)
        filtered.extend(context.notifications)
        return replace(context, notifications=())

    async def answer(**_kwargs):
        get_execution_context()["agent_inbox"].user_message("cannot be filtered away")
        return _text_response("provider answer")

    agent = _agent(
        checkpoint_store=checkpoints,
        inbox_store=inbox_store,
        hooks=[Hook(event="transform_notifications", handler=discard_user_messages)],
    )
    agent.generator.aforward = AsyncMock(side_effect=answer)
    try:
        with pytest.raises(TaskPauseRequestedError):
            await asyncio.wait_for(agent.acall("start", scope=_scope()), timeout=3)
        assert agent.generator.aforward.await_count == 1
        assert len(filtered) == 1
        assert checkpoints.load_state("steering_agent", "thread", "run")["status"] == (
            "paused"
        )
    finally:
        checkpoints.close()
        inbox_store.close()


@pytest.mark.asyncio
async def test_stream_events_keep_message_boundaries_across_steering_continuation(
    tmp_path,
):
    checkpoints = SQLiteCheckpointStore(tmp_path / "stream-checkpoints.sqlite")
    inbox_store = SQLiteAgentInboxStore(tmp_path / "stream-inbox.sqlite")
    first_chunk, release_first = asyncio.Event(), asyncio.Event()
    model_calls = 0

    async def model_call(**_kwargs):
        nonlocal model_calls
        model_calls += 1
        response = ModelStreamResponse(mode="async")
        response.set_response_type("text_generation")

        async def produce():
            if model_calls == 1:
                response.add("first part")
                first_chunk.set()
                await release_first.wait()
                response.add(" answer")
            else:
                response.add("newest answer")
            response.finish()

        asyncio.create_task(produce())
        return response

    agent = _agent(
        checkpoint_store=checkpoints,
        inbox_store=inbox_store,
        config={"stream": True},
    )
    agent.generator.aforward = AsyncMock(side_effect=model_call)
    events = []
    try:
        stream = agent.stream_events("start", scope=_scope())
        consumer = asyncio.create_task(_collect_events(stream, events))
        await asyncio.wait_for(first_chunk.wait(), timeout=3)
        agent.agent_inbox.fork(
            namespace="steering_agent", thread_id="thread", run_id="run"
        ).user_message("steer while provider is streaming")
        release_first.set()
        await asyncio.wait_for(consumer, timeout=5)

        assert model_calls == 2
        assert sum(event.type == EventType.RUN_END for event in events) == 1
        assert [
            event.data["content"]
            for event in events
            if event.type == EventType.MESSAGE_END
        ] == ["first part answer", "newest answer"]
        assert events[-1].type == EventType.RUN_END
        state = checkpoints.load_state("steering_agent", "thread", "run")
        assert state["status"] == "completed"
        assert any("newest answer" in str(item) for item in state["messages"]["items"])
    finally:
        release_first.set()
        checkpoints.close()
        inbox_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_direct_stream_consume_follows_completed_stream_continuation(
    tmp_path, asynchronous
):
    checkpoints = SQLiteCheckpointStore(
        tmp_path / f"direct-stream-checkpoints-{asynchronous}.sqlite"
    )
    inbox_store = SQLiteAgentInboxStore(
        tmp_path / f"direct-stream-inbox-{asynchronous}.sqlite"
    )
    streams = []
    requests = []
    before_end, after_end = [], []

    def make_response(text):
        response = ModelStreamResponse(mode="async" if asynchronous else "sync")
        response.set_response_type("text_generation")
        response.add(text)
        response.finish()
        streams.append(response)
        return response

    def model_call(**kwargs):
        requests.append(kwargs["messages"].to_chatml())
        if len(requests) == 1:
            get_execution_context()["agent_inbox"].user_message(
                "continue completed stream"
            )
            return make_response("first streamed answer")
        return make_response("final streamed answer")

    async def amodel_call(**kwargs):
        return model_call(**kwargs)

    def before(context):
        before_end.append(context.outcome)
        return context

    def after(context):
        after_end.append(context.outcome)
        return context

    agent = _agent(
        checkpoint_store=checkpoints,
        inbox_store=inbox_store,
        hooks=[
            Hook(event="before_run_end", handler=before),
            Hook(event="after_run_end", handler=after),
        ],
        config={"stream": True},
    )
    agent.generator.forward = Mock(side_effect=model_call)
    agent.generator.aforward = AsyncMock(side_effect=amodel_call)
    try:
        if asynchronous:
            returned = await agent.acall("start", scope=_scope())
        else:
            returned = agent("start", scope=_scope())
        assert returned is streams[0]
        chunks = [chunk async for chunk in returned.consume()]
        assert chunks == ["first streamed answer", "final streamed answer"]
        assert returned._settled_output == "final streamed answer"
        assert len(requests) == len(streams) == 2
        assert str(requests[1]).count("continue completed stream") == 1
        assert before_end == after_end == ["completed"]
        state = checkpoints.load_state("steering_agent", "thread", "run")
        assert state["status"] == "completed"
        items = state["messages"]["items"]
        assert any("first streamed answer" in str(item) for item in items)
        assert any("final streamed answer" in str(item) for item in items)
    finally:
        checkpoints.close()
        inbox_store.close()


@pytest.mark.asyncio
async def test_accepted_steering_survives_terminal_checkpoint_failure_and_reopen(
    tmp_path,
):
    checkpoint_path = tmp_path / "recovery-checkpoints.sqlite"
    inbox_path = tmp_path / "recovery-inbox.sqlite"
    checkpoints = SQLiteCheckpointStore(checkpoint_path)
    inbox_store = SQLiteAgentInboxStore(inbox_path)
    scope = _scope()
    failed_requests = []

    async def fail_terminal_commit(**kwargs):
        failed_requests.append(kwargs["messages"].to_chatml())
        if len(failed_requests) == 1:
            get_execution_context()["agent_inbox"].user_message(
                "accepted across checkpoint failure"
            )
            return _text_response("first answer")
        return _text_response("answer before rejected commit")

    agent = _agent(checkpoint_store=checkpoints, inbox_store=inbox_store)
    agent.generator.aforward = AsyncMock(side_effect=fail_terminal_commit)
    with sqlite3.connect(checkpoint_path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_completed BEFORE INSERT ON checkpoints "
            "WHEN NEW.status = 'completed' BEGIN "
            "SELECT RAISE(ABORT, 'terminal write rejected'); END"
        )

    try:
        with pytest.raises(sqlite3.IntegrityError, match="terminal write rejected"):
            await agent.acall("start", scope=scope)
        failed = checkpoints.load_state("steering_agent", "thread", "run")
        assert failed["status"] == "failed"
        assert (
            str(failed["messages"]["items"]).count("accepted across checkpoint failure")
            == 1
        )
        assert str(failed_requests[1]).count("accepted across checkpoint failure") == 1
        assert len(failed_requests) == 2
    finally:
        await agent.aclose()
        checkpoints.close()
        inbox_store.close()

    with sqlite3.connect(checkpoint_path) as connection:
        connection.execute("DROP TRIGGER reject_completed")

    reopened_checkpoints = SQLiteCheckpointStore(checkpoint_path)
    reopened_inbox_store = SQLiteAgentInboxStore(inbox_path)
    recovered_requests = []
    before_resume = []

    def validate_before_resume(context):
        history = context.messages.to_chatml()
        assert str(history).count("accepted across checkpoint failure") == 1
        before_resume.append(history)
        return context

    async def recover(**kwargs):
        recovered_requests.append(kwargs["messages"].to_chatml())
        if len(recovered_requests) == 1:
            get_execution_context()["agent_inbox"].user_message(
                "accepted after recovery validation"
            )
            return _text_response("recovered intermediate answer")
        assert (
            str(recovered_requests[-1]).count("accepted across checkpoint failure") == 1
        )
        assert (
            str(recovered_requests[-1]).count("accepted after recovery validation") == 1
        )
        return _text_response("recovered answer")

    recovered = _agent(
        checkpoint_store=reopened_checkpoints,
        inbox_store=reopened_inbox_store,
        hooks=[Hook(event="before_resume", handler=validate_before_resume)],
    )
    recovered.generator.aforward = AsyncMock(side_effect=recover)
    try:
        assert await recovered.acall("explicit recovery", scope=scope) == (
            "recovered answer"
        )
        assert len(before_resume) == 1
        assert len(recovered_requests) == 2
        assert (
            str(recovered_requests[0]).count("accepted across checkpoint failure") == 1
        )
        completed = reopened_checkpoints.load_state("steering_agent", "thread", "run")
        assert completed["status"] == "completed"
        items = completed["messages"]["items"]
        assert str(items).count("accepted across checkpoint failure") == 1
        assert str(items).count("accepted after recovery validation") == 1
        assert any("first answer" in str(item) for item in items)
        assert any("recovered answer" in str(item) for item in items)
        with pytest.raises(AgentInboxClosedError):
            recovered.agent_inbox.fork(
                namespace="steering_agent", thread_id="thread", run_id="run"
            ).user_message("too late")
        assert (
            reopened_inbox_store.load_notifications("steering_agent", "thread", "run")
            == []
        )
    finally:
        await recovered.aclose()
        reopened_checkpoints.close()
        reopened_inbox_store.close()


async def _collect_events(stream, output):
    async for event in stream:
        output.append(event)
