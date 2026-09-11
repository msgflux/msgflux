"""Automatic conversation compaction as an optional Agent extension."""

from __future__ import annotations

from dataclasses import replace

import msgspec

from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.hooks import BeforeCompaction, Hook

CONTEXT_COMPACTION_CAPABILITY = "context_compaction"

__all__ = [
    "CompactionExtension",
    "CompactionPolicy",
    "CONTEXT_COMPACTION_CAPABILITY",
]


class CompactionPolicy(msgspec.Struct, frozen=True, kw_only=True):
    """Threshold policy used by :class:`CompactionExtension`."""

    trigger_ratio: float = 0.8
    reserved_output_tokens: int = 4096
    safety_margin_tokens: int = 1024
    context_capacity: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.trigger_ratio <= 1:
            raise ValueError("`trigger_ratio` must be greater than 0 and at most 1")
        for name in ("reserved_output_tokens", "safety_margin_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"`{name}` must be a non-negative integer")
        if self.context_capacity is not None and (
            not isinstance(self.context_capacity, int) or self.context_capacity <= 0
        ):
            raise ValueError("`context_capacity` must be a positive integer or None")


class CompactionExtension(AgentExtension):
    """Enable automatic append-only compaction for an Agent."""

    def __init__(self, policy: CompactionPolicy | None = None) -> None:
        super().__init__("compaction")
        self.policy = policy or CompactionPolicy()

    def hooks(self):
        return (Hook(event="before_compaction", handler=self._decide),)

    def capabilities(self):
        return (CONTEXT_COMPACTION_CAPABILITY,)

    def _decide(self, ctx: BeforeCompaction) -> BeforeCompaction:
        capacity = self.policy.context_capacity or ctx.context_capacity
        if capacity is None:
            return replace(ctx, context_capacity=None, action="skip")
        ratio_limit = int(capacity * self.policy.trigger_ratio)
        reserve_limit = (
            capacity
            - self.policy.reserved_output_tokens
            - self.policy.safety_margin_tokens
        )
        trigger = max(1, min(ratio_limit, reserve_limit))
        action = "compact" if ctx.estimated_input_tokens >= trigger else "skip"
        return replace(
            ctx,
            context_capacity=capacity,
            trigger_tokens=trigger,
            action=action,
        )
