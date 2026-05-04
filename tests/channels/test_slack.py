import hashlib
import hmac
import json
import time

import pytest

from msgflux.channels import ChannelRegistry, SlackAdapter
from msgflux.channels.http.app import create_app
from msgflux.channels.social.slack import adapter as slack_adapter_module


class EchoAgent:
    name = "support"

    def __init__(self):
        self.calls = []

    async def acall(self, **kwargs):
        self.calls.append(kwargs)
        content = kwargs["messages"][0]["content"]
        return {"answer": f"echo: {content}"}


def _slack_event(text="hello", *, thread_ts=None, subtype=None, bot_id=None):
    event = {
        "type": "message",
        "user": "U123",
        "text": text,
        "channel": "C123",
        "ts": "1710000000.000100",
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    if subtype is not None:
        event["subtype"] = subtype
    if bot_id is not None:
        event["bot_id"] = bot_id
    return {
        "token": "deprecated",
        "team_id": "T123",
        "api_app_id": "A123",
        "event": event,
        "type": "event_callback",
        "event_id": "Ev123",
        "event_time": 1710000000,
    }


def _signed_headers(body: bytes, secret="secret", timestamp=None):
    timestamp = str(timestamp or int(time.time()))
    base = f"v0:{timestamp}:".encode() + body
    digest = hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
    return {
        "X-Slack-Request-Timestamp": timestamp,
        "X-Slack-Signature": f"v0={digest}",
    }


@pytest.mark.asyncio
async def test_slack_adapter_verifies_request_signature():
    adapter = SlackAdapter(signing_secret="secret")
    body = json.dumps(_slack_event()).encode()

    class Request:
        headers = _signed_headers(body)

    assert await adapter.verify(Request(), body) is True

    class BadRequest:
        headers = {**_signed_headers(body), "X-Slack-Signature": "v0=bad"}

    assert await adapter.verify(BadRequest(), body) is False


@pytest.mark.asyncio
async def test_slack_adapter_rejects_stale_signature():
    adapter = SlackAdapter(signing_secret="secret")
    body = json.dumps(_slack_event()).encode()

    class Request:
        headers = _signed_headers(body, timestamp=1)

    assert await adapter.verify(Request(), body) is False


@pytest.mark.asyncio
async def test_slack_adapter_returns_url_verification_challenge():
    adapter = SlackAdapter()
    body = json.dumps({"type": "url_verification", "challenge": "abc"}).encode()

    response = await adapter.webhook_response(body)

    assert response.payload == {"challenge": "abc"}
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_slack_adapter_decodes_message_event():
    adapter = SlackAdapter()

    messages = await adapter.decode(json.dumps(_slack_event()).encode())

    assert len(messages) == 1
    message = messages[0]
    assert message.id == "slack:Ev123"
    assert message.channel == "slack"
    assert message.session_id == "slack:T123:C123:1710000000.000100"
    assert message.conversation_id == "C123"
    assert message.sender_id == "U123"
    assert message.text == "hello"
    assert message.metadata["thread_ts"] == "1710000000.000100"


@pytest.mark.asyncio
async def test_slack_adapter_decodes_threaded_message_event():
    adapter = SlackAdapter()

    messages = await adapter.decode(
        json.dumps(_slack_event(thread_ts="1709999999.000200")).encode()
    )

    assert messages[0].session_id == "slack:T123:C123:1709999999.000200"
    assert messages[0].metadata["thread_ts"] == "1709999999.000200"


@pytest.mark.asyncio
async def test_slack_adapter_ignores_bot_messages():
    adapter = SlackAdapter()

    assert (
        await adapter.decode(json.dumps(_slack_event(subtype="bot_message")).encode())
        == []
    )
    assert await adapter.decode(json.dumps(_slack_event(bot_id="B123")).encode()) == []


def test_slack_url_verification_webhook_response():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    registry = ChannelRegistry()
    registry.social_adapter("slack", SlackAdapter(signing_secret="secret"))

    body = json.dumps({"type": "url_verification", "challenge": "abc"}).encode()
    headers = _signed_headers(body)
    with TestClient(create_app(registry)) as client:
        response = client.post(
            "/social/slack/webhook",
            headers={**headers, "Content-Type": "application/json"},
            content=body,
        )

    assert response.status_code == 200
    assert response.json() == {"challenge": "abc"}


def test_slack_webhook_acknowledges_and_processes_message():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    sent = []
    agent = EchoAgent()
    registry = ChannelRegistry()
    registry.agent(agent)
    registry.social_adapter(
        "slack",
        SlackAdapter(
            signing_secret="secret",
            sender=lambda outbound, _context: sent.append(outbound),
        ),
    )

    @registry.social_route(channel="slack")
    def route_slack(message, context):
        return "support"

    body = json.dumps(_slack_event("hello from slack")).encode()
    headers = _signed_headers(body)
    with TestClient(create_app(registry)) as client:
        response = client.post(
            "/social/slack/webhook",
            headers={**headers, "Content-Type": "application/json"},
            content=body,
        )
        assert response.status_code == 200
        assert response.json() == {"status": "accepted", "events": 1}

        deadline = time.time() + 2
        while not sent and time.time() < deadline:
            time.sleep(0.01)

    assert len(sent) == 1
    assert sent[0].conversation_id == "C123"
    assert sent[0].text == "echo: hello from slack"
    assert agent.calls[0]["messages"] == [
        {"role": "user", "content": "hello from slack"}
    ]


def test_slack_webhook_deduplicates_retried_event():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    sent = []
    agent = EchoAgent()
    registry = ChannelRegistry()
    registry.agent(agent)
    registry.social_adapter(
        "slack",
        SlackAdapter(
            signing_secret="secret",
            sender=lambda outbound, _context: sent.append(outbound),
        ),
    )

    @registry.social_route(channel="slack")
    def route_slack(message, context):
        return "support"

    body = json.dumps(_slack_event("retry me")).encode()
    headers = _signed_headers(body)
    with TestClient(create_app(registry)) as client:
        first = client.post(
            "/social/slack/webhook",
            headers={**headers, "Content-Type": "application/json"},
            content=body,
        )
        second = client.post(
            "/social/slack/webhook",
            headers={**headers, "Content-Type": "application/json"},
            content=body,
        )
        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json() == {"status": "accepted", "events": 1}
        assert second.json() == {"status": "accepted", "events": 0}

        deadline = time.time() + 2
        while not sent and time.time() < deadline:
            time.sleep(0.01)

    assert [message.text for message in sent] == ["echo: retry me"]
    assert len(agent.calls) == 1


@pytest.mark.asyncio
async def test_slack_adapter_send_posts_threaded_message(monkeypatch):
    requests = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return b'{"ok": true, "ts": "1710000001.000100"}'

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-token")
    monkeypatch.setattr(slack_adapter_module, "urlopen", fake_urlopen)

    adapter = SlackAdapter(timeout_s=3)
    outbound = slack_adapter_module.OutboundSocialMessage(
        channel="slack",
        conversation_id="C123",
        text="hello",
        metadata={"thread_ts": "1710000000.000100"},
    )

    await adapter.send(outbound)

    request, timeout = requests[0]
    assert timeout == 3
    assert request.full_url == "https://slack.com/api/chat.postMessage"
    assert request.get_header("Authorization") == "Bearer xoxb-token"
    payload = json.loads(request.data.decode("utf-8"))
    assert payload == {
        "channel": "C123",
        "text": "hello",
        "thread_ts": "1710000000.000100",
    }
