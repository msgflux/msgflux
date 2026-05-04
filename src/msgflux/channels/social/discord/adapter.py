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
    SocialMessage,
    SocialWebhookResponse,
)

DEFAULT_DISCORD_PUBLIC_KEY_ENV = "DISCORD_PUBLIC_KEY"
DEFAULT_DISCORD_BOT_TOKEN_ENV = "DISCORD_BOT_TOKEN"  # noqa: S105
DISCORD_API_BASE_URL = "https://discord.com/api/v10"
DISCORD_INTERACTION_PING = 1
DISCORD_INTERACTION_APPLICATION_COMMAND = 2
DISCORD_RESPONSE_PONG = 1
DISCORD_RESPONSE_DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE = 5


class DiscordInteractionsAdapter:
    def __init__(
        self,
        *,
        public_key: Optional[str] = None,
        public_key_env: Optional[str] = None,
        bot_token: Optional[str] = None,
        bot_token_env: Optional[str] = None,
        sender: Optional[Processor] = None,
        timeout_s: float = 10.0,
    ) -> None:
        self.public_key = public_key
        self.public_key_env = public_key_env or DEFAULT_DISCORD_PUBLIC_KEY_ENV
        self.bot_token = bot_token
        self.bot_token_env = bot_token_env or DEFAULT_DISCORD_BOT_TOKEN_ENV
        self.sender = sender
        self.timeout_s = timeout_s

    async def verify(self, http_request: Any = None, body: bytes = b"") -> bool:
        public_key = self._public_key()
        if not public_key:
            return True

        headers = _headers(http_request)
        signature = headers.get("x-signature-ed25519")
        timestamp = headers.get("x-signature-timestamp")
        if not signature or not timestamp:
            return False

        try:
            from cryptography.exceptions import InvalidSignature  # noqa: PLC0415
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (  # noqa: PLC0415
                Ed25519PublicKey,
            )
        except ImportError as e:
            raise ChannelError(
                "Discord signature verification requires `cryptography`."
            ) from e

        try:
            verifier = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key))
            verifier.verify(bytes.fromhex(signature), timestamp.encode() + body)
        except (InvalidSignature, ValueError):
            return False
        return True

    async def webhook_response(
        self,
        body: bytes,
        _http_request: Any = None,
    ) -> Optional[SocialWebhookResponse]:
        payload = _decode_discord_payload(body)
        interaction_type = payload.get("type")
        if interaction_type == DISCORD_INTERACTION_PING:
            return SocialWebhookResponse(payload={"type": DISCORD_RESPONSE_PONG})
        if interaction_type == DISCORD_INTERACTION_APPLICATION_COMMAND:
            return SocialWebhookResponse(
                payload={"type": DISCORD_RESPONSE_DEFERRED_CHANNEL_MESSAGE_WITH_SOURCE},
                continue_processing=True,
            )
        return None

    async def decode(
        self,
        body: bytes,
        _http_request: Any = None,
    ) -> List[SocialMessage]:
        payload = _decode_discord_payload(body)
        if payload.get("type") != DISCORD_INTERACTION_APPLICATION_COMMAND:
            return []

        data = payload.get("data")
        if not isinstance(data, ABCMapping):
            return []

        user = _interaction_user(payload)
        user_id = str(user.get("id") or "unknown")
        guild_id = str(payload.get("guild_id") or "dm")
        channel_id = str(payload.get("channel_id") or "unknown")
        interaction_id = str(payload.get("id") or "unknown")
        application_id = str(payload.get("application_id") or "")
        token = str(payload.get("token") or "")
        command_name = str(data.get("name") or "command")
        text = _command_text(data)
        attachments = _discord_attachments(payload)

        return [
            SocialMessage(
                id=f"discord:{interaction_id}",
                channel="discord",
                session_id=f"discord:{guild_id}:{channel_id}:{user_id}",
                conversation_id=channel_id,
                sender_id=user_id,
                text=text,
                attachments=attachments,
                metadata={
                    "application_id": application_id,
                    "interaction_id": interaction_id,
                    "interaction_token": token,
                    "guild_id": guild_id,
                    "channel_id": channel_id,
                    "user_id": user_id,
                    "command_name": command_name,
                    "option_values": _option_values(data),
                },
                raw=dict(payload),
            )
        ]

    async def send(self, outbound: OutboundSocialMessage, context: Any = None) -> None:
        if self.sender is not None:
            await call_processor(self.sender, outbound, context)
            return

        metadata = _outbound_metadata(outbound, context)
        application_id = str(metadata.get("application_id") or "")
        interaction_token = str(metadata.get("interaction_token") or "")
        if not application_id or not interaction_token:
            raise ChannelError(
                "Discord outbound messages require application_id and interaction_token"
            )

        await asyncio.to_thread(
            _post_discord_webhook,
            application_id,
            interaction_token,
            {"content": outbound.text},
            self._bot_token(),
            self.timeout_s,
        )

    def _public_key(self) -> str:
        return self.public_key or os.getenv(self.public_key_env, "")

    def _bot_token(self) -> str:
        return self.bot_token or os.getenv(self.bot_token_env, "")


def _decode_discord_payload(body: bytes) -> Dict[str, Any]:
    payload = msgspec.json.decode(body)
    if not isinstance(payload, ABCMapping):
        raise ChannelError("Discord interaction payload must be a JSON object")
    return dict(payload)


def _interaction_user(payload: ABCMapping) -> Dict[str, Any]:
    member = payload.get("member")
    if isinstance(member, ABCMapping) and isinstance(member.get("user"), ABCMapping):
        return dict(member["user"])
    user = payload.get("user")
    return dict(user) if isinstance(user, ABCMapping) else {}


def _command_text(data: ABCMapping) -> str:
    values = _option_values(data)
    for key in ("prompt", "message", "text", "query", "input"):
        value = values.get(key)
        if value is not None:
            return str(value)
    if values:
        return "\n".join(f"{key}: {value}" for key, value in values.items())
    return str(data.get("name") or "")


def _option_values(data: ABCMapping) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    _collect_option_values(data.get("options"), values)
    return values


def _collect_option_values(options: Any, values: Dict[str, Any]) -> None:
    if not isinstance(options, list):
        return
    for option in options:
        if not isinstance(option, ABCMapping):
            continue
        name = option.get("name")
        if isinstance(name, str) and "value" in option:
            values[name] = option.get("value")
        _collect_option_values(option.get("options"), values)


def _discord_attachments(payload: ABCMapping) -> List[SocialAttachment]:
    data = payload.get("data")
    if not isinstance(data, ABCMapping):
        return []
    resolved = data.get("resolved")
    if not isinstance(resolved, ABCMapping):
        return []
    attachments = resolved.get("attachments")
    if not isinstance(attachments, ABCMapping):
        return []

    social_attachments = []
    for attachment in attachments.values():
        if not isinstance(attachment, ABCMapping):
            continue
        social_attachments.append(
            SocialAttachment(
                type=_discord_attachment_type(attachment),
                payload=dict(attachment),
            )
        )
    return social_attachments


def _discord_attachment_type(attachment: ABCMapping) -> str:
    content_type = str(attachment.get("content_type") or "").lower()
    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("audio/"):
        return "audio"
    if content_type.startswith("video/"):
        return "video"
    return "file"


def _outbound_metadata(
    outbound: OutboundSocialMessage,
    context: Any = None,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    message = getattr(context, "message", None)
    message_metadata = getattr(message, "metadata", {}) if message is not None else {}
    if isinstance(message_metadata, ABCMapping):
        metadata.update(message_metadata)
    metadata.update(outbound.metadata)
    return metadata


def _post_discord_webhook(
    application_id: str,
    interaction_token: str,
    payload: Dict[str, Any],
    bot_token: str,
    timeout_s: float,
) -> Dict[str, Any]:
    data = msgspec.json.encode(payload)
    headers = {"Content-Type": "application/json"}
    if bot_token:
        headers["Authorization"] = f"Bot {bot_token}"
    request = URLRequest(  # noqa: S310
        f"{DISCORD_API_BASE_URL}/webhooks/{application_id}/{interaction_token}",
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            body = response.read()
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise ChannelError(
            f"Discord webhook followup failed with HTTP {e.code}: {detail}"
        ) from e
    except URLError as e:
        raise ChannelError(f"Discord webhook followup failed: {e.reason}") from e

    result = msgspec.json.decode(body) if body else {}
    if result and not isinstance(result, ABCMapping):
        raise ChannelError("Discord webhook followup returned an invalid response")
    return dict(result)


def _headers(http_request: Any = None) -> Dict[str, str]:
    if http_request is None:
        return {}
    raw_headers = getattr(http_request, "headers", {})
    return {str(key).lower(): str(value) for key, value in raw_headers.items()}
