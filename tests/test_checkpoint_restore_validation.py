"""Malformed checkpoint timelines are rejected before Agent resume."""

from copy import deepcopy
from unittest.mock import AsyncMock, Mock

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent
from msgflux.runtime import ExecutionScope
from msgflux.runtime.service import (
    AgentService,
    AgentSession,
    ServiceRecoveryRequiredError,
    SQLiteServiceStore,
)


class _CountingModel:
    model_type = "chat_completion"

    def __init__(self):
        self.calls = 0

    def __call__(self, **_kwargs):
        self.calls += 1
        return _response(f"response {self.calls}")

    async def acall(self, **_kwargs):
        self.calls += 1
        return _response(f"response {self.calls}")


def _response(content):
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(content)
    response.reasoning = None
    response.metadata = {}
    return response


def _paused_history(run_id="run"):
    messages = ChatMessages(thread_id="thread", namespace="agent")
    messages.begin_turn(turn_id=run_id)
    messages.add_user("saved prompt")
    return messages


def _store_paused_checkpoint(store, *, run_id="run"):
    store.save_state(
        "agent",
        "thread",
        run_id,
        {
            "status": "paused",
            "messages": _paused_history(run_id)._to_state(),
            "runtime": {},
            "scope": {"run_id": run_id},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "malformed_messages",
    [{"items": "corrupt"}, {"items": [None]}],
)
async def test_agent_rejects_malformed_checkpoint_before_model_call(
    asynchronous, malformed_messages
):
    store = InMemoryCheckpointStore()
    _store_paused_checkpoint(store)
    original_load = store.load_state
    model = _CountingModel()
    agent = Agent(name="agent", model=model, checkpoint_store=store)

    def corrupt_load(namespace, thread_id, run_id):
        state = original_load(namespace, thread_id, run_id)
        if state is not None and run_id == "run":
            state = deepcopy(state)
            state["messages"] = malformed_messages
        return state

    store.load_state = corrupt_load
    scope = ExecutionScope(thread_id="thread", namespace="agent", run_id="run")
    try:
        with pytest.raises(ValueError):
            if asynchronous:
                await agent.acall("ignored new input", scope=scope)
            else:
                agent("ignored new input", scope=scope)

        assert model.calls == 0
    finally:
        store.load_state = original_load
        await agent.aclose()

    saved = original_load("agent", "thread", "run")
    assert saved["status"] == "paused"
    assert any(
        item.get("content") == "saved prompt" for item in saved["messages"]["items"]
    )


def test_valid_empty_checkpoint_histories_remain_supported():
    validate = Agent._validated_checkpoint_messages

    assert validate({}) == {"items": []}
    assert validate({"messages": {}}) == {"items": []}
    assert validate({"messages": {"items": []}}) == {"items": []}
    with pytest.raises(ValueError, match="mapping"):
        validate({"messages": []})


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_agent_rejects_malformed_latest_history_before_continuing_thread(
    asynchronous,
):
    store = InMemoryCheckpointStore()
    _store_paused_checkpoint(store)
    original_latest = store.load_latest_run
    model = _CountingModel()
    agent = Agent(name="agent", model=model, checkpoint_store=store)

    def corrupt(state):
        state = deepcopy(state)
        state["messages"] = {"items": ["corrupt item"]}
        return state

    def corrupt_latest(namespace, thread_id):
        state = original_latest(namespace, thread_id)
        return corrupt(state) if state is not None else None

    store.load_latest_run = corrupt_latest
    scope = ExecutionScope(thread_id="thread", namespace="agent", run_id="new-run")
    try:
        with pytest.raises(ValueError, match="mappings"):
            if asynchronous:
                await agent.acall("new input", scope=scope)
            else:
                agent("new input", scope=scope)
        assert model.calls == 0
        assert store.load_state("agent", "thread", "new-run") is None
    finally:
        store.load_latest_run = original_latest
        await agent.aclose()


@pytest.mark.asyncio
async def test_service_refuses_malformed_checkpoint_before_resume_admission():
    checkpoints = InMemoryCheckpointStore()
    agent = Agent(name="agent", model=Mock(model_type="chat_completion"))
    agent.generator.aforward = AsyncMock(
        side_effect=[
            TaskPauseRequestedError("task", "pause before completion"),
            _response("must not execute"),
        ]
    )
    journal = SQLiteServiceStore()
    service = AgentService(store=journal)
    service.register(
        "main", lambda _thread: AgentSession(agent, checkpoint_store=checkpoints)
    )
    thread = await service.open_thread("main", thread_id="thread")
    original_load = checkpoints.load_state
    try:
        accepted = await service.prompt(
            thread.thread_id, "saved prompt", request_id="request"
        )
        paused = await service.wait(thread.thread_id, "request")
        assert paused.status == "paused"
        assert accepted.run_id == paused.run_id

        def corrupt_load(namespace, thread_id, run_id):
            state = original_load(namespace, thread_id, run_id)
            if state is not None and run_id == accepted.run_id:
                state = deepcopy(state)
                state["messages"] = {"items": "corrupt"}
            return state

        checkpoints.load_state = corrupt_load
        with pytest.raises(ServiceRecoveryRequiredError) as raised:
            await service.resume(thread.thread_id, "request")
        assert isinstance(raised.value.__cause__, ValueError)
        assert "Checkpoint messages `items` must be a list." in str(raised.value)
        assert service.receipt(thread.thread_id, "request").status == "paused"
        assert agent.generator.aforward.await_count == 1
    finally:
        checkpoints.load_state = original_load
        await service.aclose()
        journal.close()
        await agent.aclose()

    saved = original_load("agent", thread.thread_id, accepted.run_id)
    assert saved["status"] == "paused"
    assert any(
        item.get("content") == "saved prompt" for item in saved["messages"]["items"]
    )
