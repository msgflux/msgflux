"""Recoverable configuration of the last audited model response."""

import msgspec


class ModelConfiguration(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Model identity and distinct request/conversation reasoning settings.

    ``reasoning_effort`` is the effective value reported by the model adapter.
    ``UNSET`` means an older adapter did not capture a setting; ``None`` means
    the setting was captured and absent from that request.
    """

    provider: str
    model_id: str
    api_mode: str | None = None
    reasoning_effort: str | None = None
    request_reasoning_effort: str | None | msgspec.UnsetType = msgspec.UNSET
    conversation_reasoning_effort: str | None | msgspec.UnsetType = msgspec.UNSET
