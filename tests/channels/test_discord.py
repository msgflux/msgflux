import json
import time

import pytest

from msgflux.channels import ChannelRegistry, DiscordInteractionsAdapter
from msgflux.channels.http.app import create_app
from msgflux.channels.social.discord import adapter as discord_adapter_module


class EchoAgent:
    name = "support"

    def __init__(self):
        self.calls = []

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        content = kwargs["messages"][0]["content"]
        return {"answer": f"echo: {content}"}


def _discord_ping():
    return {"id": "I123", "application_id": "A123", "type": 1, "token": "tok"}


def _discord_command(prompt="hello", *, attachments=None):
    data = {
        "id": "C123",
        "name": "ask",
        "type": 1,
        "options": [{"type": 3, "name": "prompt", "value": prompt}],
    }
    if attachments is not None:
        data["resolved"] = {"attachments": attachments}
    return {
        "id": "I123",
        "application_id": "A123",
        "type": 2,
        "token": "interaction-token",
        "channel_id": "CH123",
        "guild_id": "G123",
        "member": {"user": {"id": "U123", "username": "ada"}},
        "data": data,
    }


@pytest.mark.asyncio
async def test_discord_adapter_returns_ping_pong():
    adapter = DiscordInteractionsAdapter()

    response = await adapter.webhook_response(json.dumps(_discord_ping()).encode())

    assert response.payload == {"type": 1}
    assert response.continue_processing is False


@pytest.mark.asyncio
async def test_discord_adapter_defers_application_command_and_continues():
    adapter = DiscordInteractionsAdapter()

    response = await adapter.webhook_response(json.dumps(_discord_command()).encode())

    assert response.payload == {"type": 5}
    assert response.continue_processing is True


@pytest.mark.asyncio
async def test_discord_adapter_decodes_command_options_and_attachments():
    adapter = DiscordInteractionsAdapter()
    attachment = {
        "id": "ATT123",
        "filename": "photo.png",
        "content_type": "image/png",
        "url": "https://cdn.discordapp.com/attachments/photo.png",
    }

    messages = await adapter.decode(
        json.dumps(
            _discord_command("inspect this", attachments={"ATT123": attachment})
        ).encode()
    )

    assert len(messages) == 1
    message = messages[0]
    assert message.id == "discord:I123"
    assert message.channel == "discord"
    assert message.session_id == "discord:G123:CH123:U123"
    assert message.conversation_id == "CH123"
    assert message.sender_id == "U123"
    assert message.text == "inspect this"
    assert message.metadata["application_id"] == "A123"
    assert message.metadata["interaction_token"] == "interaction-token"
    assert message.metadata["command_name"] == "ask"
    assert message.metadata["option_values"] == {"prompt": "inspect this"}
    assert message.attachments[0].type == "image"
    assert message.attachments[0].payload["id"] == "ATT123"


def test_discord_webhook_defers_and_processes_command():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    sent = []
    agent = EchoAgent()
    registry = ChannelRegistry()
    registry.agent(agent)
    registry.social_adapter(
        "discord",
        DiscordInteractionsAdapter(
            sender=lambda outbound, _context: sent.append(outbound)
        ),
    )

    @registry.social_route(channel="discord")
    def route_discord(message, context):
        return "support"

    with TestClient(create_app(registry)) as client:
        response = client.post(
            "/social/discord/webhook", json=_discord_command("hello")
        )
        assert response.status_code == 200
        assert response.json() == {"type": 5}

        deadline = time.time() + 2
        while not sent and time.time() < deadline:
            time.sleep(0.01)

    assert [message.text for message in sent] == ["echo: hello"]
    assert agent.calls[0]["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_discord_adapter_sends_followup_message(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return b'{"id": "M123"}'

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setattr(discord_adapter_module, "urlopen", fake_urlopen)

    adapter = DiscordInteractionsAdapter(bot_token="bot-token", timeout_s=3)
    outbound = discord_adapter_module.OutboundSocialMessage(
        channel="discord",
        conversation_id="CH123",
        text="hello",
        metadata={
            "application_id": "A123",
            "interaction_token": "interaction-token",
        },
    )

    await adapter.send(outbound)

    request, timeout = requests[0]
    assert timeout == 3
    assert request.full_url == (
        "https://discord.com/api/v10/webhooks/A123/interaction-token"
    )
    assert request.get_header("Authorization") == "Bot bot-token"
    assert json.loads(request.data.decode("utf-8")) == {"content": "hello"}
