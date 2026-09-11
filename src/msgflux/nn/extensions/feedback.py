"""Built-in Agent extensions for tool feedback decisions."""

from __future__ import annotations

from dataclasses import replace

from msgflux.core.dotdict import dotdict
from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.hooks import Hook, ToolFeedbackContext

__all__ = ["DefaultToolFeedbackExtension"]


class DefaultToolFeedbackExtension(AgentExtension):
    """Apply the Agent's default return policies for tool outcomes."""

    return_modes = frozenset({"direct", "handoff", "call_as_response"})

    def __init__(self) -> None:
        super().__init__("tool_feedback")

    def hooks(self):
        return (Hook(event="resolve_tool_feedback", handler=self._resolve),)

    def _resolve(self, ctx: ToolFeedbackContext) -> ToolFeedbackContext:
        if ctx.action != "continue" or not ctx.outcomes:
            return ctx

        modes = {outcome.feedback.name for outcome in ctx.outcomes}
        if not modes.issubset(self.return_modes):
            return ctx
        if len(modes) != 1:
            formatted = ", ".join(sorted(modes))
            raise ValueError(
                "One model response produced incompatible return feedback "
                f"modes: {formatted}"
            )

        intents_by_id = {intent.id: intent for intent in ctx.intents}
        tool_calls = []
        for outcome in ctx.outcomes:
            intent = intents_by_id.get(outcome.intent_id)
            if intent is None:
                raise ValueError(
                    f"Missing tool intent for outcome `{outcome.intent_id}`"
                )
            tool_calls.append(
                {
                    "id": intent.id,
                    "name": intent.name,
                    "parameters": dict(intent.arguments),
                    "result": outcome.result,
                    "error": (
                        outcome.error.message if outcome.error is not None else None
                    ),
                }
            )

        return replace(
            ctx,
            action="return",
            output=dotdict(
                tool_responses={
                    "tool_calls": tool_calls,
                    "reasoning": ctx.reasoning,
                }
            ),
        )
