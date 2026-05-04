from msgflux.channels.social.boundary import SocialBoundary
from msgflux.channels.social.bus import InMemorySocialDedupStore, InMemorySocialEventBus
from msgflux.channels.social.discord import DiscordInteractionsAdapter
from msgflux.channels.social.slack import SlackAdapter
from msgflux.channels.social.telegram import TelegramAdapter
from msgflux.channels.social.types import (
    OutboundSocialMessage,
    SocialAttachment,
    SocialContext,
    SocialEvent,
    SocialMessage,
    SocialWebhookResponse,
)

__all__ = [
    "InMemorySocialEventBus",
    "InMemorySocialDedupStore",
    "DiscordInteractionsAdapter",
    "OutboundSocialMessage",
    "SocialAttachment",
    "SocialBoundary",
    "SocialContext",
    "SocialEvent",
    "SocialMessage",
    "SocialWebhookResponse",
    "SlackAdapter",
    "TelegramAdapter",
]
