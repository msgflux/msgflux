from typing import Any

import pytest

from msgflux.nn.modules.agent import Agent


class WarmupModel:
    model_type = "chat_completion"
    provider = "test"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, str]:
        return {"response": "ok"}

    async def acall(self, **kwargs: Any) -> dict[str, str]:
        return {"response": "ok"}

    def warmup_system_prompt(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"warmed": True}

    async def awarmup_system_prompt(self, **kwargs: Any) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"warmed": True}


def lookup_ticket(ticket_id: str) -> str:
    return f"{ticket_id}: open"


def test_agent_warmup_system_prompt_uses_only_system_prompt_and_tools():
    model = WarmupModel()
    agent = Agent(
        name="support_agent",
        model=model,
        system_message="You are a support agent for {{ product }}.",
        tools=[lookup_ticket],
    )

    result = agent.warmup_system_prompt(vars={"product": "msgflux"})

    assert result == {"warmed": True}
    assert len(model.calls) == 1
    call = model.calls[0]
    assert "msgflux" in call["system_prompt"]
    assert call["tool_definitions"].schemas[0]["function"]["name"] == "lookup_ticket"
    assert "messages" not in call
    assert "generation_schema" not in call
    assert "typed_parser" not in call


@pytest.mark.asyncio
async def test_agent_async_warmup_system_prompt_applies_tool_filter():
    model = WarmupModel()
    agent = Agent(
        name="support_agent",
        model=model,
        system_message="You are a support agent.",
        tools=[lookup_ticket],
    )

    result = await agent.awarmup_system_prompt(tool_filter={"block": "*"})

    assert result == {"warmed": True}
    assert len(model.calls) == 1
    assert model.calls[0]["tool_definitions"] is None


def test_agent_warmup_system_prompt_can_spawn_background(monkeypatch):
    model = WarmupModel()
    agent = Agent(
        name="support_agent",
        model=model,
        system_message="You are a support agent.",
    )
    captured: dict[str, Any] = {}

    def fake_spawn(to_send, *args: Any, **kwargs: Any) -> None:
        captured["to_send"] = to_send
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("msgflux.nn.modules.agent.spawn", fake_spawn)

    result = agent.warmup_system_prompt(background=True)

    assert result is None
    assert captured["to_send"] == agent._warmup_system_prompt
    assert captured["kwargs"] == {
        "vars": None,
        "tool_filter": None,
        "model_preference": None,
    }
