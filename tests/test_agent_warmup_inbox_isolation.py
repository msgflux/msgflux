"""Warmup must not consume, claim, or acknowledge pending inbox notifications."""

import asyncio
from typing import Any

import pytest

from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.exceptions import TaskInterruptRequestedError, TaskPauseRequestedError
from msgflux.models.response import ModelResponse
from msgflux.nn import Agent


class WarmupModel:
    model_type = "chat_completion"
    provider = "test"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> ModelResponse:
        self.calls.append(kwargs)
        response = ModelResponse()
        response.set_response_type("text_generation")
        response.add("ok")
        return response

    async def acall(self, **kwargs: Any) -> ModelResponse:
        return self(**kwargs)

    def warmup_system_prompt(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"warmed": True}

    async def awarmup_system_prompt(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"warmed": True}


def _publish(inbox, kind):
    if kind == "user":
        return inbox.user_message("important steering message")
    if kind == "pause":
        return inbox.pause(reason="operator requested pause")
    return inbox.interrupt(reason="operator requested interrupt")


def _run_warmup(agent, *, asynchronous):
    if asynchronous:
        return asyncio.run(agent.awarmup_system_prompt())
    return agent.warmup_system_prompt()


def _assert_pending(inbox, notification):
    assert [item.notification_id for item in inbox.peek()] == [
        notification.notification_id
    ]
    assert inbox.claimed_ids() == set()
    assert inbox.delivered_ids() == set()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("checkpoint_store", [False, True])
@pytest.mark.parametrize("kind", ["user", "pause", "interrupt"])
def test_warmup_preserves_pending_inbox_notifications(
    kind, checkpoint_store, asynchronous
):
    model = WarmupModel()
    agent = Agent(
        name="warmup_inbox",
        model=model,
        system_prompt="sys",
        checkpoint_store=InMemoryCheckpointStore() if checkpoint_store else None,
    )
    inbox = agent.agent_inbox
    notification = _publish(inbox, kind)

    try:
        if kind == "user":
            assert _run_warmup(agent, asynchronous=asynchronous) == {"warmed": True}
            assert len(model.calls) == 1
            assert "messages" not in model.calls[0]
        else:
            error = (
                TaskPauseRequestedError
                if kind == "pause"
                else TaskInterruptRequestedError
            )
            with pytest.raises(error):
                _run_warmup(agent, asynchronous=asynchronous)
            assert model.calls == []

        _assert_pending(inbox, notification)
    finally:
        if asynchronous:
            asyncio.run(agent.aclose())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("checkpoint_store", [False, True])
def test_real_turn_after_warmup_claims_pending_user_message(
    checkpoint_store, asynchronous
):
    model = WarmupModel()
    agent = Agent(
        name="warmup_inbox",
        model=model,
        system_prompt="sys",
        checkpoint_store=InMemoryCheckpointStore() if checkpoint_store else None,
    )
    inbox = agent.agent_inbox
    notification = inbox.user_message("important steering message")

    try:
        assert _run_warmup(agent, asynchronous=asynchronous) == {"warmed": True}
        _assert_pending(inbox, notification)

        if asynchronous:
            result = asyncio.run(agent.acall("work"))
        else:
            result = agent("work")

        assert result == "ok"
        sent = model.calls[-1]["messages"]
        rendered = sent.to_chatml() if hasattr(sent, "to_chatml") else sent
        assert any("important steering message" in str(item) for item in rendered)
        assert inbox.peek() == []
        assert inbox.claimed_ids() == set()
    finally:
        if asynchronous:
            asyncio.run(agent.aclose())
