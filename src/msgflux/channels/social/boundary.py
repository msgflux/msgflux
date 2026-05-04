import asyncio
from collections.abc import Mapping as ABCMapping
from contextlib import suppress
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional

from msgflux.channels.exceptions import (
    ChannelError,
    ForbiddenError,
    RateLimitExceededError,
    UnauthorizedError,
)
from msgflux.channels.registry import (
    AgentRun,
    ChannelContext,
    Processor,
    call_processor,
)
from msgflux.channels.social.bus import InMemorySocialEventBus
from msgflux.channels.social.types import (
    OutboundSocialMessage,
    SocialContext,
    SocialEvent,
    SocialMessage,
    SocialWebhookResponse,
)
from msgflux.logger import logger

DEFAULT_SOCIAL_ROUTE = "*"


class SocialBoundary:
    def __init__(self, registry: Any, event_bus: Optional[Any] = None) -> None:
        self._registry = registry
        self._event_bus = event_bus or InMemorySocialEventBus()
        self._adapters: Dict[str, Any] = {}
        self._routes: Dict[str, List[Processor]] = {}
        self._commands: Dict[str, Dict[str, List[Processor]]] = {}
        self._active_tasks: Dict[str, asyncio.Task[Any]] = {}
        self._pending_events: Dict[str, List[SocialEvent]] = {}
        self._pending_tasks: Dict[str, asyncio.Task[Any]] = {}
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

    def command(
        self,
        command: str | List[str],
        handler: Optional[Processor] = None,
        *,
        channel: str = DEFAULT_SOCIAL_ROUTE,
    ) -> Processor | Callable[[Processor], Processor]:
        command_keys = _normalize_commands(command)
        channel_key = _normalize_channel(channel)

        def decorator(processor: Processor) -> Processor:
            for command_key in command_keys:
                self._commands.setdefault(channel_key, {}).setdefault(
                    command_key,
                    [],
                ).append(processor)
            return processor

        if handler is not None:
            return decorator(handler)
        return decorator

    async def handle_webhook(
        self,
        channel: str,
        body: bytes,
        http_request: Any = None,
    ) -> SocialWebhookResponse:
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

        webhook_response = await self._webhook_response(adapter, body, http_request)
        if webhook_response is not None and not webhook_response.continue_processing:
            return webhook_response

        messages = await call_processor(adapter.decode, body, http_request)
        count = 0
        for message in messages or []:
            if await self._message_seen(message):
                continue
            await self._event_bus.publish(
                SocialEvent(channel=channel_key, adapter=adapter, message=message)
            )
            count += 1
        accepted_response = SocialWebhookResponse(
            payload={"status": "accepted", "events": count},
            events=count,
        )
        if webhook_response is not None:
            webhook_response.events = count
            return webhook_response
        return accepted_response

    async def _webhook_response(
        self,
        adapter: Any,
        body: bytes,
        http_request: Any = None,
    ) -> Optional[SocialWebhookResponse]:
        handler = getattr(adapter, "webhook_response", None)
        if handler is None:
            return None
        response = await call_processor(handler, body, http_request)
        if response is None:
            return None
        if isinstance(response, SocialWebhookResponse):
            return response
        if isinstance(response, ABCMapping):
            return SocialWebhookResponse(payload=dict(response))
        raise ChannelError(
            "Social webhook_response must return None, SocialWebhookResponse, "
            "or a mapping"
        )

    async def _message_seen(self, message: SocialMessage) -> bool:
        ttl_s = getattr(self._registry.settings(), "social_dedup_ttl_s", None)
        if ttl_s is None or ttl_s <= 0:
            return False
        key = f"{message.channel}:{message.id}"
        store = self._registry.social_dedup_store()
        seen_or_mark = getattr(store, "seen_or_mark", None)
        handler = seen_or_mark if callable(seen_or_mark) else store
        result = await call_processor(handler, key, float(ttl_s))
        return bool(result)

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
        for task in list(self._active_tasks.values()):
            task.cancel()
        for task in list(self._pending_tasks.values()):
            task.cancel()
        for task in list(self._active_tasks.values()):
            with suppress(asyncio.CancelledError):
                await task
        for task in list(self._pending_tasks.values()):
            with suppress(asyncio.CancelledError):
                await task
        self._active_tasks.clear()
        self._pending_events.clear()
        self._pending_tasks.clear()
        self._consumer_task = None

    async def drain(self) -> None:
        await self._event_bus.drain()
        pending_tasks = list(self._pending_tasks.values())
        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        tasks = list(self._active_tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def active_task(self, session_id: str) -> Optional[asyncio.Task[Any]]:
        task = self._active_tasks.get(str(session_id))
        if task is None or task.done():
            return None
        return task

    def cancel_session(self, session_id: str) -> bool:
        cancelled = False
        pending_task = self._pending_tasks.pop(str(session_id), None)
        if pending_task is not None:
            pending_task.cancel()
            self._pending_events.pop(str(session_id), None)
            cancelled = True

        task = self.active_task(session_id)
        if task is not None:
            task.cancel()
            cancelled = True
        return cancelled

    async def send(
        self,
        context: SocialContext,
        message: str | OutboundSocialMessage,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        outbound = self._outbound_from_context(context, message, metadata=metadata)
        await call_processor(context.adapter.send, outbound, context)

    async def process_event(self, event: SocialEvent) -> None:
        social_context = SocialContext(
            channel=event.channel,
            adapter=event.adapter,
            message=event.message,
            boundary=self,
            state={
                "session_id": event.message.session_id,
                "conversation_id": event.message.conversation_id,
                "sender_id": event.message.sender_id,
            },
        )
        try:
            await self._authenticate_message(event.message, social_context)
        except ChannelError as e:
            await self._send_configured_error(social_context, e)
            return

        try:
            command_handled = await self._handle_command(event.message, social_context)
        except ChannelError as e:
            await self._send_configured_error(social_context, e)
            return
        if command_handled:
            return

        active_task = self.active_task(event.message.session_id)
        if active_task is not None:
            await self._send_text(
                social_context,
                "A request is already running for this session. "
                "Send /cancel to stop it.",
            )
            return

        if self._social_debounce_s() > 0:
            self._schedule_debounced_event(event)
            return

        await self._start_agent_event(event, social_context)

    async def _start_agent_event(
        self,
        event: SocialEvent,
        social_context: Optional[SocialContext] = None,
    ) -> None:
        if social_context is None:
            social_context = SocialContext(
                channel=event.channel,
                adapter=event.adapter,
                message=event.message,
                boundary=self,
                state={
                    "session_id": event.message.session_id,
                    "conversation_id": event.message.conversation_id,
                    "sender_id": event.message.sender_id,
                },
            )
            try:
                await self._authenticate_message(event.message, social_context)
            except ChannelError as e:
                await self._send_configured_error(social_context, e)
                return

        agent_name = await self._route_message(event.message, social_context)
        if not agent_name:
            return
        social_context.agent_name = str(agent_name)

        active_task = self.active_task(event.message.session_id)
        if active_task is not None:
            await self._send_text(
                social_context,
                "A request is already running for this session. "
                "Send /cancel to stop it.",
            )
            return

        task = asyncio.create_task(self._process_agent_event(event, social_context))
        self._active_tasks[event.message.session_id] = task
        task.add_done_callback(
            lambda completed, session_id=event.message.session_id: self._forget_task(
                session_id,
                completed,
            )
        )

    def _schedule_debounced_event(self, event: SocialEvent) -> None:
        session_id = event.message.session_id
        self._pending_events.setdefault(session_id, []).append(event)

        task = self._pending_tasks.pop(session_id, None)
        if task is not None:
            task.cancel()

        self._pending_tasks[session_id] = asyncio.create_task(
            self._process_debounced_session(session_id)
        )

    async def _process_debounced_session(self, session_id: str) -> None:
        try:
            await asyncio.sleep(self._social_debounce_s())
            events = self._pending_events.pop(session_id, [])
            if not events:
                return
            await self._start_agent_event(_merge_social_events(events))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Debounced social event processing failed")
        finally:
            task = self._pending_tasks.get(session_id)
            if task is asyncio.current_task():
                self._pending_tasks.pop(session_id, None)

    def _social_debounce_s(self) -> float:
        value = getattr(self._registry.settings(), "social_debounce_s", None)
        return float(value or 0)

    async def _process_agent_event(
        self,
        event: SocialEvent,
        social_context: SocialContext,
    ) -> None:
        channel_context = ChannelContext(
            channel=f"social:{event.channel}",
            agent_name=social_context.agent_name,
            request_id=event.message.id,
            request=event.message,
            state={
                **social_context.state,
                "social_context": social_context,
                "social_channel": event.channel,
                "social_message": event.message,
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
            await self._authorize_event(event.message, channel_context)
            await self._check_rate_limits(event.message, channel_context)
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
        except ChannelError as e:
            await self._send_configured_error(social_context, e)
            await self._registry.run_hooks(
                "request_end",
                event.message,
                channel_context,
                run,
                None,
                e,
            )
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

    def _forget_task(self, session_id: str, task: asyncio.Task[Any]) -> None:
        if self._active_tasks.get(session_id) is task:
            self._active_tasks.pop(session_id, None)
        with suppress(asyncio.CancelledError):
            task.exception()

    async def _authenticate_message(
        self,
        message: SocialMessage,
        social_context: SocialContext,
    ) -> None:
        context = ChannelContext(
            channel=f"social:{message.channel}",
            agent_name="",
            request_id=message.id,
            request=message,
            state={
                **social_context.state,
                "social_context": social_context,
                "social_channel": message.channel,
                "social_message": message,
                "session_id": message.session_id,
                "conversation_id": message.conversation_id,
                "sender_id": message.sender_id,
            },
        )
        principal = None
        auth_handler = self._registry.auth_handler()
        if auth_handler is not None:
            principal = await call_processor(auth_handler, None, message, context)
            if principal is False:
                raise UnauthorizedError("Unauthorized")

        context.state["principal"] = principal
        context.state["auth"] = principal
        social_context.state.update(context.state)

    async def _authorize_event(
        self,
        message: SocialMessage,
        context: ChannelContext,
    ) -> None:
        principal = context.state.get("principal")
        for authorizer in self._registry.authorizers(context.agent_name):
            result = await call_processor(authorizer, message, context, principal)
            if result is False:
                raise ForbiddenError("Forbidden")
            if isinstance(result, ABCMapping):
                context.state.update(result)

    async def _check_rate_limits(
        self,
        message: SocialMessage,
        context: ChannelContext,
    ) -> None:
        await self._registry.check_rate_limits(message, context, None)

    async def _send_configured_error(
        self,
        context: SocialContext,
        exc: ChannelError,
    ) -> None:
        message = None
        if isinstance(exc, UnauthorizedError):
            message = self._registry.settings().social_unauthorized_message
        elif isinstance(exc, ForbiddenError):
            message = self._registry.settings().social_forbidden_message
        elif isinstance(exc, RateLimitExceededError):
            message = self._registry.settings().social_rate_limit_message

        if message:
            await self._send_text(context, message)

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

    async def _handle_command(
        self,
        message: SocialMessage,
        context: SocialContext,
    ) -> bool:
        command_name = _message_command(message)
        if command_name is None:
            return False

        handlers = [
            *self._commands.get(message.channel, {}).get(command_name, []),
            *self._commands.get(DEFAULT_SOCIAL_ROUTE, {}).get(command_name, []),
        ]
        if not handlers:
            return await self._handle_builtin_command(command_name, message, context)

        await self._check_command_rate_limits(message, context)
        for handler in handlers:
            result = await call_processor(handler, message, context)
            if isinstance(result, OutboundSocialMessage | str):
                await self.send(context, result)
            elif _is_outbound_sequence(result):
                for item in result:
                    await self.send(context, item)
            elif result is False:
                return False
            elif result is not None:
                raise ChannelError(
                    "Social command handlers must return None, False, str, "
                    "OutboundSocialMessage, or a sequence of str/OutboundSocialMessage"
                )
        return True

    async def _handle_builtin_command(
        self,
        command_name: str,
        message: SocialMessage,
        context: SocialContext,
    ) -> bool:
        if command_name not in {"/cancel", "/stop"}:
            return False

        await self._check_command_rate_limits(message, context)
        cancelled = self.cancel_session(message.session_id)
        if cancelled:
            await self._send_text(context, "Cancelled the active request.")
        else:
            await self._send_text(context, "No active request to cancel.")
        return True

    async def _check_command_rate_limits(
        self,
        message: SocialMessage,
        social_context: SocialContext,
    ) -> None:
        context = ChannelContext(
            channel=f"social:{message.channel}",
            agent_name="",
            request_id=message.id,
            request=message,
            state={
                **social_context.state,
                "social_context": social_context,
                "social_channel": message.channel,
                "social_message": message,
                "session_id": message.session_id,
                "conversation_id": message.conversation_id,
                "sender_id": message.sender_id,
            },
        )
        await self._registry.check_rate_limits(message, context, None)

    async def _send_text(self, context: SocialContext, text: str) -> None:
        await self.send(context, text)

    def _outbound_from_context(
        self,
        context: SocialContext,
        message: str | OutboundSocialMessage,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OutboundSocialMessage:
        if isinstance(message, OutboundSocialMessage):
            if metadata:
                message.metadata.update(metadata)
            return message
        if isinstance(message, str):
            return OutboundSocialMessage.from_context(
                context,
                message,
                metadata=metadata,
            )
        raise ChannelError("Social send expects str or OutboundSocialMessage")

    async def _prepare_run(
        self,
        message: SocialMessage,
        context: ChannelContext,
    ) -> AgentRun:
        defaults = self._registry.run_defaults(context.agent_name)
        run = AgentRun(
            messages=[{"role": "user", "content": _social_message_content(message)}],
            vars=dict(defaults.vars),
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


def _normalize_channel(channel: str) -> str:
    key = str(channel).strip().lower()
    if not key:
        raise ValueError("Social channel must not be empty")
    return key


def _normalize_command(command: str) -> str:
    value = str(command).strip().lower()
    if not value:
        raise ValueError("Social command must not be empty")
    if not value.startswith("/"):
        value = f"/{value}"
    return value


def _normalize_commands(command: str | List[str]) -> List[str]:
    if isinstance(command, str):
        return [_normalize_command(command)]
    commands = [_normalize_command(item) for item in command]
    if not commands:
        raise ValueError("Social command list must not be empty")
    return commands


def _is_outbound_sequence(value: Any) -> bool:
    if not isinstance(value, list | tuple):
        return False
    return all(isinstance(item, str | OutboundSocialMessage) for item in value)


def _message_command(message: SocialMessage) -> Optional[str]:
    text = (message.text or "").strip()
    if not text.startswith("/"):
        return None
    return text.split(maxsplit=1)[0].split("@", maxsplit=1)[0].lower()


def _merge_social_events(events: List[SocialEvent]) -> SocialEvent:
    if len(events) == 1:
        return events[0]

    first = events[0]
    messages = [event.message for event in events]
    last_message = messages[-1]
    text_parts = [message.text for message in messages if message.text]
    attachments = [
        attachment for message in messages for attachment in message.attachments
    ]
    message = replace(
        last_message,
        text="\n".join(text_parts) if text_parts else last_message.text,
        content=None,
        attachments=attachments,
        metadata={
            **last_message.metadata,
            "batched": True,
            "batch_size": len(messages),
            "batch_message_ids": [message.id for message in messages],
        },
        raw={"messages": [message.raw for message in messages]},
    )
    return SocialEvent(channel=first.channel, adapter=first.adapter, message=message)


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


def _social_message_content(message: SocialMessage) -> Any:
    if message.content is not None:
        return message.content
    return message.text or ""


def _social_output_text(output: Any) -> str:
    if output is None:
        return ""
    consume = getattr(output, "consume", None)
    if callable(consume):
        output = consume()
    if isinstance(output, ABCMapping):
        for key in ("answer", "response", "content", "text"):
            value = output.get(key)
            if value is not None:
                return str(value)
    return str(output)
