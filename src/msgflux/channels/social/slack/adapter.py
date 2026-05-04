import asyncio
import hashlib
import hmac
import os
import time
from collections.abc import Mapping as ABCMapping
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest
from urllib.request import urlopen

import msgspec

from msgflux.channels.exceptions import ChannelError
from msgflux.channels.registry import Processor, call_processor
from msgflux.channels.social.types import (
    OutboundSocialMessage,
    SocialMessage,
    SocialWebhookResponse,
)

DEFAULT_SLACK_BOT_TOKEN_ENV = "SLACK_BOT_TOKEN"  # noqa: S105
DEFAULT_SLACK_SIGNING_SECRET_ENV = "SLACK_SIGNING_SECRET"  # noqa: S105
SLACK_SIGNATURE_VERSION = "v0"
SLACK_SIGNATURE_TOLERANCE_S = 60 * 5


class SlackAdapter:
    def __init__(
        self,
        *,
        bot_token: Optional[str] = None,
        bot_token_env: Optional[str] = None,
        signing_secret: Optional[str] = None,
        signing_secret_env: Optional[str] = None,
        sender: Optional[Processor] = None,
        timeout_s: float = 10.0,
        signature_tolerance_s: int = SLACK_SIGNATURE_TOLERANCE_S,
    ) -> None:
        self.bot_token = bot_token
        self.bot_token_env = bot_token_env or DEFAULT_SLACK_BOT_TOKEN_ENV
        self.signing_secret = signing_secret
        self.signing_secret_env = signing_secret_env or DEFAULT_SLACK_SIGNING_SECRET_ENV
        self.sender = sender
        self.timeout_s = timeout_s
        self.signature_tolerance_s = signature_tolerance_s

    async def verify(self, http_request: Any = None, body: bytes = b"") -> bool:
        secret = self._signing_secret()
        if not secret:
            return True
        headers = _headers(http_request)
        timestamp = headers.get("x-slack-request-timestamp")
        signature = headers.get("x-slack-signature")
        if not timestamp or not signature:
            return False
        try:
            request_ts = int(timestamp)
        except ValueError:
            return False
        if abs(int(time.time()) - request_ts) > self.signature_tolerance_s:
            return False

        base = f"{SLACK_SIGNATURE_VERSION}:{timestamp}:".encode() + body
        digest = hmac.new(secret.encode(), base, hashlib.sha256).hexdigest()
        expected = f"{SLACK_SIGNATURE_VERSION}={digest}"
        return hmac.compare_digest(expected, signature)

    async def webhook_response(
        self,
        body: bytes,
        _http_request: Any = None,
    ) -> Optional[SocialWebhookResponse]:
        payload = _decode_slack_payload(body)
        if payload.get("type") != "url_verification":
            return None
        challenge = payload.get("challenge")
        if not isinstance(challenge, str):
            raise ChannelError("Slack url_verification payload is missing challenge")
        return SocialWebhookResponse(payload={"challenge": challenge})

    async def decode(
        self,
        body: bytes,
        _http_request: Any = None,
    ) -> List[SocialMessage]:
        payload = _decode_slack_payload(body)
        if payload.get("type") != "event_callback":
            return []
        event = payload.get("event")
        if not isinstance(event, ABCMapping):
            return []
        if event.get("type") != "message":
            return []
        if event.get("subtype") in {"bot_message", "message_deleted"}:
            return []
        if event.get("bot_id") or event.get("user") is None:
            return []

        text = event.get("text")
        if text is None:
            return []

        team_id = str(payload.get("team_id") or event.get("team") or "unknown")
        channel_id = str(event.get("channel") or "unknown")
        user_id = str(event.get("user"))
        ts = str(event.get("ts") or payload.get("event_id") or "unknown")
        thread_ts = str(event.get("thread_ts") or ts)
        event_id = str(payload.get("event_id") or f"{channel_id}:{ts}")

        return [
            SocialMessage(
                id=f"slack:{event_id}",
                channel="slack",
                session_id=f"slack:{team_id}:{channel_id}:{thread_ts}",
                conversation_id=channel_id,
                sender_id=user_id,
                text=str(text),
                metadata={
                    "team_id": team_id,
                    "channel_id": channel_id,
                    "user_id": user_id,
                    "ts": ts,
                    "thread_ts": thread_ts,
                    "event_type": event.get("type"),
                },
                raw=dict(payload),
            )
        ]

    async def send(self, outbound: OutboundSocialMessage, context: Any = None) -> None:
        if self.sender is not None:
            await call_processor(self.sender, outbound, context)
            return

        payload: Dict[str, Any] = {
            "channel": outbound.conversation_id,
            "text": outbound.text,
        }
        thread_ts = _thread_ts(outbound, context)
        if thread_ts:
            payload["thread_ts"] = thread_ts

        await asyncio.to_thread(
            _post_slack_api,
            self._bot_token(),
            "chat.postMessage",
            payload,
            self.timeout_s,
        )

    def _bot_token(self) -> str:
        token = self.bot_token or os.getenv(self.bot_token_env, "")
        if not token:
            raise ChannelError("Slack bot token is not configured")
        return token

    def _signing_secret(self) -> str:
        return self.signing_secret or os.getenv(self.signing_secret_env, "")


def _decode_slack_payload(body: bytes) -> Dict[str, Any]:
    payload = msgspec.json.decode(body)
    if not isinstance(payload, ABCMapping):
        raise ChannelError("Slack webhook payload must be a JSON object")
    return dict(payload)


def _thread_ts(outbound: OutboundSocialMessage, context: Any = None) -> str:
    if outbound.metadata.get("thread_ts"):
        return str(outbound.metadata["thread_ts"])
    message = getattr(context, "message", None)
    metadata = getattr(message, "metadata", {}) if message is not None else {}
    if isinstance(metadata, ABCMapping) and metadata.get("thread_ts"):
        return str(metadata["thread_ts"])
    return ""


def _post_slack_api(
    token: str,
    method: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    data = msgspec.json.encode(payload)
    request = URLRequest(
        f"https://slack.com/api/{method}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            body = response.read()
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise ChannelError(
            f"Slack API `{method}` failed with HTTP {e.code}: {detail}"
        ) from e
    except URLError as e:
        raise ChannelError(f"Slack API `{method}` failed: {e.reason}") from e

    result = msgspec.json.decode(body) if body else {}
    if not isinstance(result, ABCMapping):
        raise ChannelError(f"Slack API `{method}` returned an invalid response")
    if result.get("ok") is False:
        error = result.get("error") or "unknown_error"
        raise ChannelError(f"Slack API `{method}` failed: {error}")
    return dict(result)


def _headers(http_request: Any = None) -> Dict[str, str]:
    if http_request is None:
        return {}
    raw_headers = getattr(http_request, "headers", {})
    return {str(key).lower(): str(value) for key, value in raw_headers.items()}
