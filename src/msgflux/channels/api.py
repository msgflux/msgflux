"""Small authorization and presentation boundary around AgentService."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Awaitable, Callable
from typing import TypeAlias

import msgspec

from msgflux.runtime.service.records import AdmissionReceipt
from msgflux.runtime.service.store import validate_identifier

from .records import ChannelAdmission, ChannelContext, ChannelReply, ChannelRequest

AuthorizeCallback: TypeAlias = Callable[
    [ChannelContext, ChannelRequest], bool | Awaitable[bool]
]
PreprocessorCallback: TypeAlias = Callable[
    [ChannelContext, ChannelRequest], ChannelRequest | Awaitable[ChannelRequest]
]
PostprocessorCallback: TypeAlias = Callable[[ChannelContext, str], str | Awaitable[str]]
CommandCallback: TypeAlias = Callable[
    [ChannelContext, ChannelRequest, str], str | Awaitable[str]
]


class ChannelPermissionError(PermissionError):
    """Raised when a channel request is denied by its host authorization policy."""


async def _resolve(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value


def _validate_context(context: ChannelContext) -> None:
    if not isinstance(context, ChannelContext):
        raise TypeError("context must be a ChannelContext")
    for field in ("channel", "principal", "request_id"):
        validate_identifier(getattr(context, field), field)


def _validate_request(request: ChannelRequest) -> None:
    if not isinstance(request, ChannelRequest):
        raise TypeError("request must be a ChannelRequest")
    validate_identifier(request.agent_id, "agent_id")
    validate_identifier(request.thread_id, "thread_id")
    if not isinstance(request.prompt, str):
        raise TypeError("prompt must be a string")


def _validate_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


class AgentChannel:
    """Adapt an authorized channel request to a borrowed Agent service.

    The service is borrowed and must expose ``open_thread`` and ``prompt`` with
    the same signatures as ``AgentService``. This adapter never owns its lifecycle.
    """

    def __init__(
        self,
        service: object,
        *,
        name: str,
        authorize: AuthorizeCallback,
    ) -> None:
        validate_identifier(name, "name")
        if not callable(authorize):
            raise TypeError("authorize must be callable")
        if not callable(getattr(service, "open_thread", None)) or not callable(
            getattr(service, "prompt", None)
        ):
            raise TypeError("service must provide open_thread and prompt")
        self.service = service
        self._name = name
        self._authorize_callback = authorize
        self._preprocessors: list[PreprocessorCallback] = []
        self._postprocessors: list[PostprocessorCallback] = []
        self._commands: dict[str, CommandCallback] = {}

    @property
    def name(self) -> str:
        """Stable origin namespace used for request identities."""
        return self._name

    def register_preprocessor(self, fn: PreprocessorCallback) -> PreprocessorCallback:
        """Append a sync or async request transform, returning it for decorators."""
        if not callable(fn):
            raise TypeError("preprocessor must be callable")
        self._preprocessors.append(fn)
        return fn

    def register_postprocessor(
        self, fn: PostprocessorCallback
    ) -> PostprocessorCallback:
        """Append a sync or async text transform for presentation only."""
        if not callable(fn):
            raise TypeError("postprocessor must be callable")
        self._postprocessors.append(fn)
        return fn

    def register_command(self, name: str, fn: CommandCallback) -> CommandCallback:
        """Register a local text command by exact name, without a leading slash."""
        validate_identifier(name, "command name")
        if name.startswith("/") or any(character.isspace() for character in name):
            raise ValueError(
                "command name must be a single token without a leading slash"
            )
        if not callable(fn):
            raise TypeError("command must be callable")
        if name in self._commands:
            raise ValueError(f"Command {name!r} is already registered")
        self._commands[name] = fn
        return fn

    async def _check_authorization(
        self, context: ChannelContext, request: ChannelRequest
    ) -> None:
        allowed = await _resolve(self._authorize_callback(context, request))
        if type(allowed) is not bool:
            raise TypeError("authorize callback must return bool")
        if not allowed:
            raise ChannelPermissionError("Channel request is not authorized")

    async def _preprocess(
        self,
        context: ChannelContext,
        request: ChannelRequest,
        callbacks: tuple[PreprocessorCallback, ...],
    ) -> ChannelRequest:
        for callback in callbacks:
            updated = await _resolve(callback(context, request))
            if not isinstance(updated, ChannelRequest):
                raise TypeError("preprocessor must return a ChannelRequest")
            _validate_request(updated)
            request = updated
        return request

    async def _postprocess(
        self,
        context: ChannelContext,
        content: str,
        callbacks: tuple[PostprocessorCallback, ...],
    ) -> str:
        content = _validate_text(content, "content")
        for callback in callbacks:
            content = await _resolve(callback(context, content))
            content = _validate_text(content, "postprocessor result")
        return content

    async def prompt(
        self,
        request: ChannelRequest,
        *,
        principal: str,
        request_id: str,
    ) -> ChannelAdmission | ChannelReply:
        """Authorize and process one source message without waiting for execution.

        Registered commands return a reply without Agent admission. Other inputs
        return a service receipt. Origin identity is scoped to this channel and
        principal; retries must keep the processed destination and prompt stable.
        """
        _validate_request(request)
        validate_identifier(principal, "principal")
        validate_identifier(request_id, "request_id")
        context = ChannelContext(self.name, principal, request_id)
        _validate_context(context)

        preprocessors = tuple(self._preprocessors)
        postprocessors = tuple(self._postprocessors)
        commands = self._commands.copy()

        await self._check_authorization(context, request)
        request = await self._preprocess(context, request, preprocessors)
        await self._check_authorization(context, request)

        parts = request.prompt.split(maxsplit=1)
        first = parts[0] if parts else ""
        arguments = parts[1] if len(parts) > 1 else ""
        command_name = first[1:] if first.startswith("/") else ""
        command = commands.get(command_name) if command_name else None
        if command is not None:
            output = await _resolve(command(context, request, arguments))
            output = _validate_text(output, "command result")
            output = await self._postprocess(context, output, postprocessors)
            return ChannelReply(context, request, output)

        identity = msgspec.json.encode(
            (context.channel, context.principal, context.request_id)
        )
        service_request_id = "channel:" + hashlib.sha256(identity).hexdigest()
        await self.service.open_thread(request.agent_id, thread_id=request.thread_id)
        receipt = await self.service.prompt(
            request.thread_id, request.prompt, request_id=service_request_id
        )
        if not isinstance(receipt, AdmissionReceipt):
            raise TypeError("service prompt must return an AdmissionReceipt")
        return ChannelAdmission(context, request, receipt)

    async def format(self, admission: ChannelAdmission, content: str) -> ChannelReply:
        """Format observed answer text without changing its canonical history."""
        if not isinstance(admission, ChannelAdmission):
            raise TypeError("admission must be a ChannelAdmission")
        _validate_context(admission.context)
        _validate_request(admission.request)
        if admission.context.channel != self.name:
            raise ValueError("admission belongs to another channel")
        if not isinstance(admission.receipt, AdmissionReceipt):
            raise TypeError("admission receipt must be an AdmissionReceipt")
        postprocessors = tuple(self._postprocessors)
        formatted = await self._postprocess(admission.context, content, postprocessors)
        return ChannelReply(admission.context, admission.request, formatted)
