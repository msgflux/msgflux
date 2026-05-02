import asyncio
import os
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
    SocialAttachment,
    SocialContext,
    SocialMessage,
)

DEFAULT_TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"  # noqa: S105
DEFAULT_TELEGRAM_WEBHOOK_SECRET_ENV = "TELEGRAM_WEBHOOK_SECRET"  # noqa: S105


class TelegramAdapter:
    def __init__(
        self,
        *,
        bot_token: Optional[str] = None,
        bot_token_env: Optional[str] = None,
        secret_token: Optional[str] = None,
        secret_token_env: Optional[str] = None,
        sender: Optional[Processor] = None,
        timeout_s: float = 10.0,
    ) -> None:
        self.bot_token = bot_token
        self.bot_token_env = bot_token_env or DEFAULT_TELEGRAM_BOT_TOKEN_ENV
        self.secret_token = secret_token
        self.secret_token_env = secret_token_env or DEFAULT_TELEGRAM_WEBHOOK_SECRET_ENV
        self.sender = sender
        self.timeout_s = timeout_s

    async def set_webhook(
        self,
        url: str,
        *,
        secret_token: Optional[str] = None,
        drop_pending_updates: Optional[bool] = None,
        allowed_updates: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"url": url}
        secret = secret_token if secret_token is not None else self._secret_token()
        if secret:
            payload["secret_token"] = secret
        if drop_pending_updates is not None:
            payload["drop_pending_updates"] = drop_pending_updates
        if allowed_updates is not None:
            payload["allowed_updates"] = allowed_updates
        return await asyncio.to_thread(
            _post_telegram_api,
            self._bot_token(),
            "setWebhook",
            payload,
            self.timeout_s,
        )

    async def delete_webhook(
        self,
        *,
        drop_pending_updates: Optional[bool] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if drop_pending_updates is not None:
            payload["drop_pending_updates"] = drop_pending_updates
        return await asyncio.to_thread(
            _post_telegram_api,
            self._bot_token(),
            "deleteWebhook",
            payload,
            self.timeout_s,
        )

    async def get_webhook_info(self) -> Dict[str, Any]:
        return await asyncio.to_thread(
            _post_telegram_api,
            self._bot_token(),
            "getWebhookInfo",
            {},
            self.timeout_s,
        )

    async def verify(self, http_request: Any = None, _body: bytes = b"") -> bool:
        expected = self._secret_token()
        if not expected:
            return True
        headers = (
            getattr(http_request, "headers", {}) if http_request is not None else {}
        )
        return headers.get("x-telegram-bot-api-secret-token") == expected

    async def decode(
        self,
        body: bytes,
        _http_request: Any = None,
    ) -> List[SocialMessage]:
        payload = msgspec.json.decode(body)
        if not isinstance(payload, ABCMapping):
            raise ChannelError("Telegram webhook payload must be a JSON object")

        telegram_message = payload.get("message") or payload.get("edited_message")
        if not isinstance(telegram_message, ABCMapping):
            return []

        text = telegram_message.get("text") or telegram_message.get("caption")
        chat = telegram_message.get("chat")
        sender = telegram_message.get("from") or {}
        if not isinstance(chat, ABCMapping) or not chat.get("id"):
            return []

        chat_id = str(chat["id"])
        sender_id = str(sender.get("id") or chat_id)
        message_id = str(telegram_message.get("message_id") or payload.get("update_id"))
        update_id = str(payload.get("update_id") or message_id)
        attachments = _telegram_attachments(telegram_message)
        if text is None and not attachments:
            return []

        return [
            SocialMessage(
                id=f"telegram:{update_id}:{message_id}",
                channel="telegram",
                session_id=f"telegram:{chat_id}",
                conversation_id=chat_id,
                sender_id=sender_id,
                text=str(text) if text is not None else None,
                attachments=attachments,
                metadata={
                    "update_id": update_id,
                    "message_id": message_id,
                    "chat_type": chat.get("type"),
                    "username": sender.get("username"),
                    "first_name": sender.get("first_name"),
                },
                raw=dict(payload),
            )
        ]

    async def send(
        self,
        outbound: OutboundSocialMessage,
        _context: SocialContext = None,
    ) -> None:
        if self.sender is not None:
            await call_processor(self.sender, outbound, _context)
            return

        for chunk in _telegram_text_chunks(outbound.text):
            await asyncio.to_thread(
                _post_telegram_message,
                self._bot_token(),
                outbound.conversation_id,
                chunk,
                self.timeout_s,
            )

    def _bot_token(self) -> str:
        token = self.bot_token or os.getenv(self.bot_token_env, "")
        if not token:
            raise ChannelError("Telegram bot token is not configured")
        return token

    def _secret_token(self) -> str:
        return self.secret_token or os.getenv(self.secret_token_env, "")


def _telegram_attachments(message: ABCMapping) -> List[SocialAttachment]:
    attachments = []
    for key in ("photo", "document", "audio", "voice", "video", "sticker"):
        if key in message:
            attachments.append(SocialAttachment(type=key, payload=message[key]))
    return attachments


def _telegram_text_chunks(text: str) -> List[str]:
    if not text:
        return []
    limit = 4096
    return [text[index : index + limit] for index in range(0, len(text), limit)]


def _post_telegram_message(
    token: str,
    chat_id: str,
    text: str,
    timeout_s: float,
) -> None:
    _post_telegram_api(
        token,
        "sendMessage",
        {"chat_id": chat_id, "text": text},
        timeout_s,
    )


def _post_telegram_api(
    token: str,
    method: str,
    payload: Dict[str, Any],
    timeout_s: float,
) -> Dict[str, Any]:
    data = msgspec.json.encode(payload)
    request = URLRequest(
        f"https://api.telegram.org/bot{token}/{method}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            body = response.read()
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise ChannelError(
            f"Telegram API `{method}` failed with HTTP {e.code}: {detail}"
        ) from e
    except URLError as e:
        raise ChannelError(f"Telegram API `{method}` failed: {e.reason}") from e

    result = msgspec.json.decode(body) if body else {}
    if not isinstance(result, ABCMapping):
        raise ChannelError(f"Telegram API `{method}` returned an invalid response")
    if result.get("ok") is False:
        description = result.get("description") or "unknown error"
        raise ChannelError(f"Telegram API `{method}` failed: {description}")
    return dict(result)
