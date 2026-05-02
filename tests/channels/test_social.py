import asyncio
import json
import threading
import time

import pytest

from msgflux.channels import ChannelRegistry, OutboundSocialMessage, TelegramAdapter
from msgflux.channels.http.app import create_app
from msgflux.channels import social as social_module


class EchoAgent:
    name = "support"

    def __init__(self):
        self.calls = []

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        content = kwargs["messages"][0]["content"]
        return {"answer": f"echo: {content}", "reasoning": "internal"}


class SlowAgent:
    name = "support"

    def __init__(self):
        self.started = threading.Event()
        self.cancelled = threading.Event()

    async def acall(self, **kwargs):
        self.started.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return {"answer": "finished"}


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


def test_telegram_social_command_responds_without_agent_call():
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

    @registry.social_command("/start", channel="telegram")
    def start_command(message, context):
        return "authenticated"

    @registry.social_route(channel="telegram")
    def route_telegram(message, context):
        return "support"

    with TestClient(create_app(registry)) as client:
        response = client.post(
            "/social/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
            json=_telegram_payload("/start"),
        )
        assert response.status_code == 200

        deadline = time.time() + 2
        while not sent and time.time() < deadline:
            time.sleep(0.01)

    assert len(sent) == 1
    assert sent[0].channel == "telegram"
    assert sent[0].conversation_id == "456"
    assert sent[0].text == "authenticated"
    assert agent.calls == []


def test_telegram_social_command_can_return_outbound_from_context():
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

    @registry.social_command("help", channel="telegram")
    def help_command(message, context):
        return OutboundSocialMessage.from_context(
            context,
            "help text",
            metadata={"command": "/help"},
        )

    with TestClient(create_app(registry)) as client:
        response = client.post(
            "/social/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
            json=_telegram_payload("/help"),
        )
        assert response.status_code == 200

        deadline = time.time() + 2
        while not sent and time.time() < deadline:
            time.sleep(0.01)

    assert len(sent) == 1
    assert sent[0].conversation_id == "456"
    assert sent[0].text == "help text"
    assert sent[0].metadata == {"command": "/help"}


def test_telegram_social_command_can_fall_through_to_route():
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

    @registry.social_command("/start", channel="telegram")
    def start_command(message, context):
        return False

    @registry.social_route(channel="telegram")
    def route_telegram(message, context):
        return "support"

    with TestClient(create_app(registry)) as client:
        response = client.post(
            "/social/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
            json=_telegram_payload("/start route me"),
        )
        assert response.status_code == 200

        deadline = time.time() + 2
        while not sent and time.time() < deadline:
            time.sleep(0.01)

    assert sent[0].text == "echo: /start route me"
    assert agent.calls[0]["messages"] == [
        {"role": "user", "content": "/start route me"}
    ]


def test_telegram_builtin_cancel_stops_active_session_task():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    sent = []
    agent = SlowAgent()
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
        first = client.post(
            "/social/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
            json=_telegram_payload("long running"),
        )
        assert first.status_code == 200
        assert agent.started.wait(timeout=2)

        cancel_payload = _telegram_payload("/cancel")
        cancel_payload["update_id"] = 1002
        cancel_payload["message"]["message_id"] = 43
        cancel = client.post(
            "/social/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
            json=cancel_payload,
        )
        assert cancel.status_code == 200

        deadline = time.time() + 2
        while not sent and time.time() < deadline:
            time.sleep(0.01)

    assert agent.cancelled.is_set()
    assert [message.text for message in sent] == ["Cancelled the active request."]


def test_telegram_webhook_acknowledges_and_processes_message():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    sent = []
    route_contexts = []
    agent = EchoAgent()
    registry = ChannelRegistry()
    registry.defaults(vars={"tenant": "default"})
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
        route_contexts.append(context)
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
    assert agent.calls[0]["vars"] == {"tenant": "default"}
    assert route_contexts[0].message.session_id == "telegram:456"
    assert route_contexts[0].state["conversation_id"] == "456"


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


def test_telegram_webhook_uses_registry_auth_and_authorize():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    sent = []
    auth_contexts = []
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

    @registry.auth
    def auth(http_request, message, context):
        auth_contexts.append((http_request, message, context))
        if context.channel == "social:telegram" and message.sender_id == "123":
            return {"sender_id": message.sender_id}
        return False

    @registry.authorize(agent="support")
    def authorize(message, context, principal):
        return principal["sender_id"] == context.state["sender_id"]

    @registry.social_route(channel="telegram")
    def route_telegram(message, context):
        return "support"

    with TestClient(create_app(registry)) as client:
        response = client.post(
            "/social/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
            json=_telegram_payload("authenticated"),
        )
        assert response.status_code == 200

        deadline = time.time() + 2
        while not sent and time.time() < deadline:
            time.sleep(0.01)

    assert len(sent) == 1
    assert sent[0].text == "echo: authenticated"
    assert agent.calls[0]["messages"] == [{"role": "user", "content": "authenticated"}]
    http_request, message, context = auth_contexts[0]
    assert http_request is None
    assert message.sender_id == "123"
    assert context.channel == "social:telegram"
    assert context.state["principal"] == {"sender_id": "123"}
    assert context.state["social_message"] is message


def test_telegram_webhook_drops_unauthorized_social_event():
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

    @registry.auth
    def auth(http_request, message, context):
        return False

    @registry.social_route(channel="telegram")
    def route_telegram(message, context):
        return "support"

    with TestClient(create_app(registry)) as client:
        response = client.post(
            "/social/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
            json=_telegram_payload("blocked"),
        )
        assert response.status_code == 200

        time.sleep(0.05)

    assert sent == []
    assert agent.calls == []


def test_telegram_webhook_applies_social_rate_limits():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    sent = []
    agent = EchoAgent()
    registry = ChannelRegistry()
    registry.agent(agent)
    registry.rate_limit(
        name="telegram-sender-minute",
        agent="support",
        requests=1,
        window_s=60,
        by=lambda message, context: context.state["sender_id"],
    )
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
        first = client.post(
            "/social/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
            json=_telegram_payload("first"),
        )
        assert first.status_code == 200

        deadline = time.time() + 2
        while len(sent) < 1 and time.time() < deadline:
            time.sleep(0.01)

        second_payload = _telegram_payload("second")
        second_payload["update_id"] = 1002
        second_payload["message"]["message_id"] = 43
        second = client.post(
            "/social/telegram/webhook",
            headers={"X-Telegram-Bot-Api-Secret-Token": "secret"},
            json=second_payload,
        )
        assert second.status_code == 200

        time.sleep(0.05)

    assert [message.text for message in sent] == ["echo: first"]
    assert [call["messages"][0]["content"] for call in agent.calls] == ["first"]


@pytest.mark.asyncio
async def test_telegram_adapter_sets_webhook_with_secret_env(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return b'{"ok": true, "result": true}'

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "webhook-secret")
    monkeypatch.setattr(social_module, "urlopen", fake_urlopen)

    result = await TelegramAdapter(timeout_s=3).set_webhook(
        "https://example.com/social/telegram/webhook"
    )

    assert result == {"ok": True, "result": True}
    request, timeout = requests[0]
    assert timeout == 3
    assert request.full_url == "https://api.telegram.org/botbot-token/setWebhook"
    assert request.get_header("Content-type") == "application/json"
    payload = json.loads(request.data.decode("utf-8"))
    assert payload == {
        "url": "https://example.com/social/telegram/webhook",
        "secret_token": "webhook-secret",
    }
