"""Immutable records passed through channel adapters."""

import msgspec

from msgflux.runtime.service.records import AdmissionReceipt


class ChannelContext(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Host supplied identity for one channel request."""

    channel: str
    principal: str
    request_id: str


class ChannelRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """The agent, conversation, and prompt selected by a channel caller."""

    agent_id: str
    thread_id: str
    prompt: str


class ChannelAdmission(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """A successfully authorized request admitted by the Agent service."""

    context: ChannelContext
    request: ChannelRequest
    receipt: AdmissionReceipt


class ChannelReply(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Text formatted for delivery back to the originating channel."""

    context: ChannelContext
    request: ChannelRequest
    content: str
