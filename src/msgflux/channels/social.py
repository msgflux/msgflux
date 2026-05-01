import asyncio
import os
from collections.abc import Mapping as ABCMapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.request import Request as URLRequest
from urllib.request import urlopen

import msgspec

from msgflux.channels.exceptions import ChannelError, ForbiddenError
from msgflux.channels.registry import (
    AgentRun,
    ChannelContext,
    Processor,
    call_processor,
)
from msgflux.logger import logger

DEFAULT_SOCIAL_ROUTE = "*"
DEFAULT_TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"  # noqa: S105
DEFAULT_TELEGRAM_WEBHOOK_SECRET_ENV = "TELEGRAM_WEBHOOK_SECRET"  # noqa: S105


@dataclass
class SocialAttachment:
    type: str
    payload: Any


@dataclass
class SocialMessage:
    id: str
    channel: str
    session_id: str
    conversation_id: str
    sender_id: str
    text: Optional[str] = None
    attachments: List[SocialAttachment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocialContext:
    channel: str
    adapter: Any
    message: SocialMessage
    agent_name: Optional[str] = None
    state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocialEvent:
    channel: str
    adapter: Any
    message: SocialMessage


@dataclass
class OutboundSocialMessage:
    channel: str
    conversation_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class InMemorySocialEventBus:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    async def publish(self, event: SocialEvent) -> None:
        await self._queue.put(event)

    async def get(self) -> Any:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    async def drain(self) -> None:
        await self._queue.join()

    async def close(self) -> None:
        await self._queue.put(None)


class SocialBoundary:
    def __init__(self, registry: Any, event_bus: Optional[Any] = None) -> None:
        self._registry = registry
        self._event_bus = event_bus or InMemorySocialEventBus()
        self._adapters: Dict[str, Any] = {}
        self._routes: Dict[str, List[Processor]] = {}
        self._consumer_task: Optional[asyncio.Task[Any]] = None

    def adapter(self, channel: str, adapter: Any) -> Any:
        channel_key = _normalize_channel(channel)
        if channel_key in self._adapters:
            raise ValueError(f"Social adapter `{channel_key}` is already registered")
        self._adapters[channel_key] = adapter
        return adapter

    def adapters(self) -> Dict[str, Any]:
        return dict(self._adapters)

    def has_adapters(self) -> bool:
        return bool(self._adapters)

    def route(
        self,
        target: str | Processor | None = None,
        *,
        channel: str = DEFAULT_SOCIAL_ROUTE,
    ) -> Processor | Callable[[Processor], Processor]:
        if callable(target) and not isinstance(target, str):
            processor = target
            self._routes.setdefault(DEFAULT_SOCIAL_ROUTE, []).append(processor)
            return processor

        key = _normalize_channel(target if isinstance(target, str) else channel)

        def decorator(processor: Processor) -> Processor:
            self._routes.setdefault(key, []).append(processor)
            return processor

        return decorator

    async def handle_webhook(
        self,
        channel: str,
        body: bytes,
        http_request: Any = None,
    ) -> int:
        channel_key = _normalize_channel(channel)
        adapter = self._adapters.get(channel_key)
        if adapter is None:
            raise ChannelError(f"Social adapter `{channel_key}` is not registered")

        is_verified = await call_processor(
            adapter.verify,
            http_request,
            body,
        )
        if is_verified is False:
            raise ForbiddenError("Invalid social webhook signature")

        messages = await call_processor(adapter.decode, body, http_request)
        count = 0
        for message in messages or []:
            await self._event_bus.publish(
                SocialEvent(channel=channel_key, adapter=adapter, message=message)
            )
            count += 1
        return count

    async def start(self) -> None:
        if not self._adapters or self._consumer_task is not None:
            return
        self._consumer_task = asyncio.create_task(self._consume_loop())

    async def stop(self) -> None:
        if self._consumer_task is None:
            return
        await self._event_bus.close()
        with suppress(asyncio.CancelledError):
            await self._consumer_task
        self._consumer_task = None

    async def drain(self) -> None:
        await self._event_bus.drain()

    async def process_event(self, event: SocialEvent) -> None:
        social_context = SocialContext(
            channel=event.channel,
            adapter=event.adapter,
            message=event.message,
            state={
                "session_id": event.message.session_id,
                "conversation_id": event.message.conversation_id,
                "sender_id": event.message.sender_id,
            },
        )
        agent_name = await self._route_message(event.message, social_context)
        if not agent_name:
            return
        social_context.agent_name = str(agent_name)

        channel_context = ChannelContext(
            channel=f"social:{event.channel}",
            agent_name=social_context.agent_name,
            request_id=event.message.id,
            request=event.message,
            state={
                **social_context.state,
                "social_context": social_context,
                "social_channel": event.channel,
                "session_id": event.message.session_id,
                "conversation_id": event.message.conversation_id,
                "sender_id": event.message.sender_id,
            },
        )
        run = None
        try:
            await self._registry.run_hooks(
                "request_start",
                event.message,
                channel_context,
            )
            agent = self._registry.get_agent(social_context.agent_name)
            run = await self._prepare_run(event.message, channel_context)
            output = await agent.acall(
                messages=run.messages,
                vars=run.vars,
                model_preference=run.model_preference,
                tool_filter=run.tool_filter,
                stream=False,
                **run.kwargs,
            )
            output = await self._apply_post_processors(
                social_context.agent_name,
                output,
                channel_context,
                run,
            )
            text = _social_output_text(output)
            if text:
                await call_processor(
                    event.adapter.send,
                    OutboundSocialMessage(
                        channel=event.channel,
                        conversation_id=event.message.conversation_id,
                        text=text,
                        metadata={
                            "session_id": event.message.session_id,
                            "sender_id": event.message.sender_id,
                        },
                    ),
                    social_context,
                )
            await self._registry.run_hooks(
                "request_end",
                event.message,
                channel_context,
                run,
                output,
                None,
            )
        except asyncio.CancelledError as e:
            await self._registry.run_hooks(
                "request_end",
                event.message,
                channel_context,
                run,
                None,
                e,
            )
            raise
        except Exception as e:
            await self._registry.run_hooks(
                "request_end",
                event.message,
                channel_context,
                run,
                None,
                e,
            )
            raise

    async def _consume_loop(self) -> None:
        while True:
            event = await self._event_bus.get()
            try:
                if event is None:
                    return
                await self.process_event(event)
            except Exception:
                logger.exception("Social event processing failed")
            finally:
                self._event_bus.task_done()

    async def _route_message(
        self,
        message: SocialMessage,
        context: SocialContext,
    ) -> Optional[str]:
        routes = [
            *self._routes.get(message.channel, []),
            *self._routes.get(DEFAULT_SOCIAL_ROUTE, []),
        ]
        for route in routes:
            agent_name = await call_processor(route, message, context)
            if agent_name:
                return str(agent_name)
        return None

    async def _prepare_run(
        self,
        message: SocialMessage,
        context: ChannelContext,
    ) -> AgentRun:
        defaults = self._registry.run_defaults(context.agent_name)
        run = AgentRun(
            messages=[{"role": "user", "content": message.text or ""}],
            vars={
                **defaults.vars,
                "session_id": message.session_id,
                "social_channel": message.channel,
                "conversation_id": message.conversation_id,
                "sender_id": message.sender_id,
            },
            stream=False,
            model_preference=defaults.model_preference,
            tool_filter=defaults.tool_filter,
            kwargs=dict(defaults.kwargs),
            policies=_run_policies(defaults),
        )
        for processor in self._registry.pre_processors(context.agent_name):
            update = await call_processor(processor, message, context, run)
            run = _apply_run_update(run, update)
        return run

    async def _apply_post_processors(
        self,
        agent_name: str,
        output: Any,
        context: ChannelContext,
        run: AgentRun,
    ) -> Any:
        for processor in self._registry.post_processors(agent_name):
            update = await call_processor(processor, output, context, run)
            if update is not None:
                output = update
        return output


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

    async def verify(self, http_request: Any = None, _body: bytes = b"") -> bool:
        expected = self.secret_token or os.getenv(self.secret_token_env, "")
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

        token = self.bot_token or os.getenv(self.bot_token_env, "")
        if not token:
            raise ChannelError("Telegram bot token is not configured")

        for chunk in _telegram_text_chunks(outbound.text):
            await asyncio.to_thread(
                _post_telegram_message,
                token,
                outbound.conversation_id,
                chunk,
                self.timeout_s,
            )


def _normalize_channel(channel: str) -> str:
    key = str(channel).strip().lower()
    if not key:
        raise ValueError("Social channel must not be empty")
    return key


def _run_policies(defaults: Any) -> Dict[str, Any]:
    policies = {}
    if defaults.stream_policy is not None:
        policies["stream"] = defaults.stream_policy
    return policies


def _apply_run_update(run: AgentRun, update: Any) -> AgentRun:
    if update is None:
        return run
    if isinstance(update, AgentRun):
        return update
    if not isinstance(update, ABCMapping):
        raise ChannelError(
            "Social pre processors must return None, AgentRun, or a mapping"
        )

    field_updates = {
        "messages": list,
        "vars": _as_dict,
        "stream": _identity,
        "model_preference": _identity,
        "tool_filter": _identity,
        "kwargs": _as_dict,
    }
    for field_name, transform in field_updates.items():
        if field_name in update:
            setattr(run, field_name, transform(update[field_name]))
    return run


def _as_dict(value: Any) -> Dict[str, Any]:
    return dict(value or {})


def _identity(value: Any) -> Any:
    return value


def _social_output_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, ABCMapping):
        for key in ("answer", "response", "content", "text"):
            value = output.get(key)
            if value is not None:
                return str(value)
    return str(output)


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
    encoder = msgspec.json.Encoder()
    data = encoder.encode({"chat_id": chat_id, "text": text})
    request = URLRequest(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310
        response.read()
