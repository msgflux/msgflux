"""Contracts for trusted channel callbacks and request normalization."""

import asyncio
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest

from msgflux.channels import (
    AgentChannel,
    ChannelAdmission,
    ChannelPermissionError,
    ChannelReply,
    ChannelRequest,
)
from msgflux.runtime.service import AdmissionReceipt


def _channel(authorize=None):
    service = Mock()
    service.open_thread = AsyncMock()
    service.prompt = AsyncMock(
        return_value=AdmissionReceipt("thread", "internal", "run", "accepted")
    )
    channel = AgentChannel(
        service,
        name="telegram",
        authorize=authorize or (lambda _context, _request: True),
    )
    return channel, service


@pytest.mark.asyncio
async def test_callbacks_are_ordered_and_final_destination_is_authorized():
    calls = []

    async def authorize(context, request):
        calls.append(("authorize", request.thread_id))
        assert context.principal == "alice"
        return True

    channel, service = _channel(authorize)

    def first(context, request):
        calls.append(("first", request.prompt))
        assert context.channel == "telegram"
        return msgspec.structs.replace(request, prompt="normalized")

    async def second(context, request):
        calls.append(("second", request.prompt))
        return msgspec.structs.replace(request, thread_id="other")

    assert channel.register_preprocessor(first) is first
    assert channel.register_preprocessor(second) is second
    result = await channel.prompt(
        ChannelRequest("main", "thread", "hello"),
        principal="alice",
        request_id="delivery-1",
    )
    assert isinstance(result, ChannelAdmission)
    assert result.context.request_id == "delivery-1"
    assert result.request == ChannelRequest("main", "other", "normalized")
    assert calls == [
        ("authorize", "thread"),
        ("first", "hello"),
        ("second", "normalized"),
        ("authorize", "other"),
    ]
    service.open_thread.assert_awaited_once_with("main", thread_id="other")
    assert service.prompt.await_args.args == ("other", "normalized")


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [False, None, 1, "yes"])
async def test_authorizer_must_explicitly_allow_before_callbacks(answer):
    channel, service = _channel(lambda *_args: answer)
    processor = Mock()
    channel.register_preprocessor(processor)
    with pytest.raises((ChannelPermissionError, TypeError)):
        await channel.prompt(
            ChannelRequest("main", "thread", "hello"),
            principal="alice",
            request_id="delivery-1",
        )
    processor.assert_not_called()
    service.open_thread.assert_not_awaited()
    service.prompt.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "update",
    [None, {"prompt": "hello"}, ChannelRequest("main", "", "hello")],
)
async def test_invalid_preprocessor_results_do_not_reach_service(update):
    channel, service = _channel()
    channel.register_preprocessor(lambda *_args: update)
    with pytest.raises((TypeError, ValueError)):
        await channel.prompt(
            ChannelRequest("main", "thread", "hello"),
            principal="alice",
            request_id="delivery-1",
        )
    service.open_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_slash_text_is_an_ordinary_prompt():
    channel, service = _channel()
    result = await channel.prompt(
        ChannelRequest("main", "thread", "/usr/bin/python explain this path"),
        principal="alice",
        request_id="delivery-1",
    )
    assert isinstance(result, ChannelAdmission)
    assert service.prompt.await_args.args[1] == "/usr/bin/python explain this path"


@pytest.mark.asyncio
async def test_commands_use_effective_prompt_and_ordered_presentation_processors():
    channel, service = _channel()
    seen = []

    def command(context, request, arguments):
        seen.append((context.principal, request.prompt, arguments))
        return "answer"

    assert channel.register_command("help", command) is command
    channel.register_preprocessor(
        lambda _context, request: msgspec.structs.replace(
            request, prompt="/help two words"
        )
    )
    channel.register_postprocessor(lambda _context, text: text.upper())

    async def suffix(context, text):
        return text + "!"

    assert channel.register_postprocessor(suffix) is suffix
    result = await channel.prompt(
        ChannelRequest("main", "thread", "help me"),
        principal="alice",
        request_id="delivery-1",
    )
    assert isinstance(result, ChannelReply)
    assert result.content == "ANSWER!"
    assert seen == [("alice", "/help two words", "two words")]
    service.open_thread.assert_not_awaited()
    service.prompt.assert_not_awaited()


@pytest.mark.asyncio
async def test_postprocessor_must_return_text():
    channel, _service = _channel()
    admission = await channel.prompt(
        ChannelRequest("main", "thread", "hello"),
        principal="alice",
        request_id="delivery-1",
    )
    channel.register_postprocessor(lambda _context, _text: {"text": "bad"})
    with pytest.raises(TypeError):
        await channel.format(admission, "answer")


@pytest.mark.asyncio
async def test_cannot_format_an_admission_from_another_channel():
    channel, service = _channel()
    other = AgentChannel(service, name="slack", authorize=lambda *_args: True)
    admission = await channel.prompt(
        ChannelRequest("main", "thread", "hello"),
        principal="alice",
        request_id="delivery-1",
    )
    with pytest.raises(ValueError):
        await other.format(admission, "answer")


def test_command_registration_rejects_duplicates():
    channel, _service = _channel()
    channel.register_command("help", lambda *_args: "help")
    with pytest.raises(ValueError):
        channel.register_command("help", lambda *_args: "other")


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["agent_id", "thread_id", "principal", "request_id"])
async def test_identifiers_are_validated_before_authorization(field):
    authorize = Mock(return_value=True)
    channel, service = _channel(authorize)
    values = {
        "agent_id": "main",
        "thread_id": "thread",
        "principal": "alice",
        "request_id": "1",
    }
    values[field] = ""
    with pytest.raises(ValueError):
        await channel.prompt(
            ChannelRequest(values["agent_id"], values["thread_id"], "hello"),
            principal=values["principal"],
            request_id=values["request_id"],
        )
    authorize.assert_not_called()
    service.open_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_registration_during_authorization_affects_only_later_requests():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def authorize(_context, _request):
        entered.set()
        await release.wait()
        return True

    channel, service = _channel(authorize)
    first = asyncio.create_task(
        channel.prompt(
            ChannelRequest("main", "thread", "/help"),
            principal="alice",
            request_id="1",
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        channel.register_preprocessor(
            lambda _context, request: msgspec.structs.replace(request, prompt="/new")
        )
        channel.register_command("new", lambda *_args: "new command")
        channel.register_postprocessor(lambda _context, text: text.upper())
        release.set()
        admitted = await asyncio.wait_for(first, 2)
        assert isinstance(admitted, ChannelAdmission)
        assert admitted.request.prompt == "/help"
        service.prompt.assert_awaited_once()
        reply = await channel.prompt(
            ChannelRequest("main", "thread", "/help"),
            principal="alice",
            request_id="2",
        )
        assert isinstance(reply, ChannelReply)
        assert reply.content == "NEW COMMAND"
        service.prompt.assert_awaited_once()
    finally:
        release.set()
        if not first.done():
            first.cancel()
        await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_command_retries_require_host_idempotency():
    channel, service = _channel()
    handler = Mock(return_value="help")
    channel.register_command("help", handler)
    for _attempt in range(2):
        reply = await channel.prompt(
            ChannelRequest("main", "thread", "/help"),
            principal="alice",
            request_id="same-source-id",
        )
        assert isinstance(reply, ChannelReply)
    assert handler.call_count == 2
    service.prompt.assert_not_awaited()
