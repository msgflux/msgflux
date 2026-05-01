import json
import time

import pytest

from msgflux.channels import ChannelRegistry, TelegramAdapter
from msgflux.channels.http.app import create_app


class EchoAgent:
    name = "support"

    def __init__(self):
        self.calls = []

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        content = kwargs["messages"][0]["content"]
        return {"answer": f"echo: {content}"}


def _telegram_payload(text="hello"):
    return {
        "update_id": 1001,
        "message": {
            "message_id": 42,
            "from": {
                "id": 123,
                "is_bot": False,
                "first_name": "Ada",
                "username": "ada",
            },
            "chat": {"id": 456, "type": "private"},
            "date": 1710000000,
            "text": text,
        },
    }


@pytest.mark.asyncio
async def test_telegram_adapter_decodes_text_message():
    adapter = TelegramAdapter(secret_token="secret")

    messages = await adapter.decode(json.dumps(_telegram_payload()).encode())

    assert len(messages) == 1
    message = messages[0]
    assert message.id == "telegram:1001:42"
    assert message.channel == "telegram"
    assert message.session_id == "telegram:456"
    assert message.conversation_id == "456"
    assert message.sender_id == "123"
    assert message.text == "hello"
    assert message.metadata["chat_type"] == "private"


def test_registry_social_route_registers_adapter_and_route():
    registry = ChannelRegistry()
    adapter = TelegramAdapter()

    registry.social_adapter("telegram", adapter)

    @registry.social_route(channel="telegram")
    def route(message, context):
        return "support"

    boundary = registry.social_boundary()
    assert boundary.adapters()["telegram"] is adapter


def test_telegram_webhook_acknowledges_and_processes_message():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    sent = []
    agent = EchoAgent()
    registry = ChannelRegistry()
    registry.agent(agent)
    registry.social_adapter(
        "telegram",
        TelegramAdapter(
            secret_token="secret",
            sender=lambda outbound, _context: sent.append(outbound),
        ),
    )

    @registry.social_route(channel="telegram")
    def route_telegram(message, context):
        return "support"

    with TestClient(create_app(registry)) as client:
        home = client.get("/")
        assert home.json()["social"] == {"telegram": "/social/telegram/webhook"}

        response = client.post(
            "/social/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
            json=_telegram_payload("hello from telegram"),
        )
        assert response.status_code == 200
        assert response.json() == {"status": "accepted", "events": 1}

        deadline = time.time() + 2
        while not sent and time.time() < deadline:
            time.sleep(0.01)

    assert len(sent) == 1
    assert sent[0].conversation_id == "456"
    assert sent[0].text == "echo: hello from telegram"
    assert agent.calls[0]["messages"] == [
        {"role": "user", "content": "hello from telegram"}
    ]
    assert agent.calls[0]["vars"]["session_id"] == "telegram:456"
    assert agent.calls[0]["vars"]["social_channel"] == "telegram"


def test_telegram_webhook_rejects_invalid_secret():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    sent = []
    registry = ChannelRegistry()
    registry.agent(EchoAgent())
    registry.social_adapter(
        "telegram",
        TelegramAdapter(
            secret_token="secret",
            sender=lambda outbound, _context: sent.append(outbound),
        ),
    )

    @registry.social_route(channel="telegram")
    def route_telegram(message, context):
        return "support"

    with TestClient(create_app(registry)) as client:
        response = client.post(
            "/social/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
            json=_telegram_payload(),
        )

    assert response.status_code == 403
    assert response.json()["error"]["code"] == "forbidden"
    assert sent == []
