from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.nn import Agent
from msgflux.nn.hooks import Hook
from msgflux.runtime import ExecutionScope
from msgflux.runtime.agent_inbox import AgentInbox, InMemoryAgentInboxStore
from msgflux.runtime.service import AgentService, AgentSession
from msgflux.runtime.service.store import SQLiteServiceStore


class FailCompletedOnce(InMemoryCheckpointStore):
    def __init__(self):
        super().__init__()
        self.failure = OSError("terminal completed commit rejected")
        self.fail = True

    def commit_state(self, namespace, thread_id, run_id, state, **kwargs):
        if state.get("status") == "completed" and self.fail:
            self.fail = False
            raise self.failure
        return super().commit_state(namespace, thread_id, run_id, state, **kwargs)


def _text(text):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


def _tool_call(name):
    response = ModelResponse()
    response.set_response_type("tool_call")
    calls = ToolCallAggregator()
    calls.process(0, "call_1", name, "{}")
    response.add(calls)
    response.reasoning = None
    response.metadata = {}
    return response


class ScriptedModel:
    model_type = "chat_completion"

    def __init__(self):
        self.responses = [
            _tool_call("count_effect"),
            _text("first final"),
            _text("recovered final"),
        ]
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)

    async def acall(self, **kwargs):
        return self(**kwargs)


def _make_agent(store, model, effects):
    def count_effect() -> str:
        """Count one irreversible tool effect for the resume regression."""
        effects.append("done")
        return "effect persisted"

    return Agent(name="main", model=model, tools=[count_effect], checkpoint_store=store)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_agent_terminal_checkpoint_prewrite_failure_stays_resumable(asynchronous):
    store = FailCompletedOnce()
    model = ScriptedModel()
    effects = []
    agent = _make_agent(store, model, effects)
    history = ChatMessages()
    scope = ExecutionScope(namespace="main", thread_id="thread", run_id="same-run")
    try:
        with pytest.raises(OSError) as raised:
            if asynchronous:
                asyncio.run(
                    agent.acall("original input", messages=history, scope=scope)
                )
            else:
                agent("original input", messages=history, scope=scope)
        assert raised.value is store.failure
        failed = store.load_state("main", "thread", "same-run")
        assert failed["status"] == "failed"
        turns = [
            item for item in failed["messages"]["items"] if item.get("type") == "turn"
        ]
        assert turns[-1]["event"] == "fail"
        assert history.turns[-1]["status"] == "failed"
        assert (
            sum(
                item.get("type") == "function_call_output"
                for item in failed["messages"]["items"]
            )
            == 1
        )
        assert effects == ["done"]

        if asynchronous:
            result = asyncio.run(agent.acall("ignored retry", scope=scope))
        else:
            result = agent("ignored retry", scope=scope)
        assert result == "recovered final"
        recovered = store.load_state("main", "thread", "same-run")
        assert recovered["status"] == "completed"
        assert effects == ["done"]
        assert len(model.calls) == 3
    finally:
        asyncio.run(agent.aclose())


def test_terminal_checkpoint_publishes_into_hook_visible_history_after_commit():
    history = ChatMessages()
    observed = []

    def before(ctx):
        observed.append(
            ("before", ctx.messages, ctx.messages.get_active_turn()["status"])
        )

    def after(ctx):
        observed.append(("after", ctx.messages, ctx.messages.turns[-1]["status"]))

    model = Mock(model_type="chat_completion", return_value=_text("done"))
    agent = Agent(
        name="agent",
        model=model,
        checkpoint_store=InMemoryCheckpointStore(),
        hooks=[
            Hook(event="before_run_end", handler=before),
            Hook(event="after_run_end", handler=after),
        ],
    )
    try:
        assert agent("work", messages=history) == "done"
        assert [entry[0] for entry in observed] == ["before", "after"]
        assert all(entry[1] is history for entry in observed)
        assert [entry[2] for entry in observed] == ["in_progress", "completed"]
        assert history.turns[-1]["status"] == "completed"
    finally:
        asyncio.run(agent.aclose())


def test_non_durable_terminal_does_not_copy_history_after_before_hook(monkeypatch):
    history = ChatMessages()
    agent = Agent(
        name="agent",
        model=Mock(model_type="chat_completion", return_value=_text("done")),
    )

    def before(ctx):
        monkeypatch.setattr(
            ChatMessages,
            "copy",
            lambda _self: pytest.fail("non-durable finalization copied history"),
        )

    agent.hooks = [Hook(event="before_run_end", handler=before)]
    try:
        assert agent("work", messages=history) == "done"
        assert history.turns[-1]["status"] == "completed"
    finally:
        asyncio.run(agent.aclose())


@pytest.mark.asyncio
async def test_service_terminal_checkpoint_prewrite_failure_resumes_same_run():
    checkpoints = FailCompletedOnce()
    model = ScriptedModel()
    effects = []
    agent = _make_agent(checkpoints, model, effects)
    service = AgentService(store=SQLiteServiceStore())
    service.register(
        "main", lambda _thread: AgentSession(agent, checkpoint_store=checkpoints)
    )
    thread = await service.open_thread("main", thread_id="service-thread")
    try:
        accepted = await service.prompt(
            thread.thread_id, "original input", request_id="req"
        )
        failed = await service.wait(thread.thread_id, "req")
        assert failed.status == "failed"
        saved = checkpoints.load_state("main", thread.thread_id, accepted.run_id)
        turns = [
            item for item in saved["messages"]["items"] if item.get("type") == "turn"
        ]
        assert saved["status"] == "failed" and turns[-1]["event"] == "fail"
        assert effects == ["done"]

        resumed = await service.resume(thread.thread_id, "req")
        assert resumed.run_id == accepted.run_id
        completed = await service.wait(thread.thread_id, "req")
        assert completed.status == "completed"
        assert effects == ["done"]
        assert len(model.calls) == 3
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_inbox_ack_oserror_after_commit_does_not_demote_completed(monkeypatch):
    checkpoints = InMemoryCheckpointStore()
    inbox = AgentInbox(
        store=InMemoryAgentInboxStore(),
        namespace="agent",
        thread_id="thread",
        run_id="run",
    )
    inbox.set_reasoning_effort("high")
    model = Mock(model_type="chat_completion")
    agent = Agent(
        name="agent", model=model, agent_inbox=inbox, checkpoint_store=checkpoints
    )
    response = _text("done")
    agent.generator.aforward = AsyncMock(return_value=response)

    def failed_ack(self, _ids, **_kwargs):
        raise OSError("injected inbox ack failure")

    monkeypatch.setattr(AgentInbox, "ack", failed_ack)
    try:
        with pytest.warns(RuntimeWarning, match="Inbox acknowledgement failed"):
            assert (
                await agent.acall(
                    "work",
                    scope=ExecutionScope(
                        namespace="agent", thread_id="thread", run_id="run"
                    ),
                )
                == "done"
            )
        saved = checkpoints.load_state("agent", "thread", "run")
        assert saved["status"] == "completed"
        turns = [
            item for item in saved["messages"]["items"] if item.get("type") == "turn"
        ]
        assert turns[-1]["event"] == "complete"
        assert [
            item["reasoning_effort"]
            for item in saved["messages"]["items"]
            if item.get("type") == "model_configuration"
        ] == ["high"]
        assert len(inbox.peek()) == 1
    finally:
        await agent.aclose()
