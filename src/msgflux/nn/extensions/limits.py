"""Terminal tool-round policy, independent of the model wire protocol."""

from dataclasses import replace

from msgflux.core.dotdict import dotdict
from msgflux.nn.extensions.base import AgentExtension
from msgflux.nn.hooks import Hook
from msgflux.nn.hooks.events import ContinuationContext, ModelContext
from msgflux.runtime.agent_run import get_agent_run
from msgflux.tools.responses import ToolResponses


class ToolTurnLimitExtension(AgentExtension):
    """Stop after a settled batch budget and optionally warn before the last batch.

    One round is one model-produced batch, regardless of its number of calls.
    The exhausted run returns its last tool results and a structured stop reason;
    no extra model request is issued to synthesize a final answer.
    """

    def __init__(self, limit: int, *, warn_remaining: int = 1):
        super().__init__("tool_turn_limit")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if (
            isinstance(warn_remaining, bool)
            or not isinstance(warn_remaining, int)
            or warn_remaining < 0
        ):
            raise ValueError("warn_remaining must be a non-negative integer")
        self.limit = limit
        self.warn_remaining = warn_remaining

    def hooks(self):
        return (
            Hook(event="resolve_continuation", handler=self._decide),
            Hook(event="transform_system_prompt", handler=self._warn),
        )

    def _decide(self, ctx: ContinuationContext) -> ContinuationContext:
        state = self.durable_state()
        completed = state.get("completed_tool_turns", 0)
        if ctx.phase == "after_tools":
            completed += 1
            state["completed_tool_turns"] = completed
            state["last_tool_results"] = ToolResponses.from_outcomes(
                ctx.intents, ctx.outcomes
            ).to_dict()["tool_calls"]
        if completed < self.limit:
            return ctx
        return replace(
            ctx,
            action="return",
            stop_reason="tool_turn_limit",
            output=dotdict(
                stop_reason="tool_turn_limit",
                completed_tool_turns=completed,
                tool_responses={"tool_calls": state.get("last_tool_results", [])},
            ),
        )

    def _warn(self, ctx: ModelContext) -> ModelContext:
        if get_agent_run() is None:
            return ctx
        remaining = self.limit - self.durable_state().get("completed_tool_turns", 0)
        if self.warn_remaining == 0 or not 0 < remaining <= self.warn_remaining:
            return ctx
        notice = (
            f"Tool budget: {remaining} round(s) remaining. "
            "After the last tool round this run ends immediately; use it to "
            "complete the task. There will be no further model request."
        )
        return replace(
            ctx, system_prompt=f"{ctx.system_prompt or ''}\n\n{notice}".strip()
        )
