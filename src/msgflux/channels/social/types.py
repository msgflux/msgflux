from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    content: Any = None
    attachments: List[SocialAttachment] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocialContext:
    channel: str
    adapter: Any
    message: SocialMessage
    boundary: Any = None
    agent_name: Optional[str] = None
    state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SocialEvent:
    channel: str
    adapter: Any
    message: SocialMessage


@dataclass
class SocialWebhookResponse:
    status_code: int = 200
    payload: Dict[str, Any] = field(default_factory=dict)
    events: int = 0


@dataclass
class OutboundSocialMessage:
    channel: str
    conversation_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context(
        cls,
        context: SocialContext,
        text: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "OutboundSocialMessage":
        return cls(
            channel=context.message.channel,
            conversation_id=context.message.conversation_id,
            text=text,
            metadata=metadata or {},
        )
