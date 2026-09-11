"""Shared loop boundaries independent of tool protocol and budget policy."""

# ruff: noqa: A002

from msgflux.models.response import ModelResponse
from msgflux.nn.hooks.events import ContinuationContext
from msgflux.runtime.context import get_execution_context


class _TerminalResponse(ModelResponse):
    """Runtime output that must not be decoded as a model flow-control step."""

    def __init__(self, decision: ContinuationContext):
        super().__init__()
        self.set_response_type("structured")
        self.add(decision.output)
        self.metadata = {"stop_reason": decision.stop_reason}


class AgentContinuationMixin:
    def _continuation_context(self, phase, messages, vars, intents=(), outcomes=()):
        return ContinuationContext(
            phase=phase,
            scope=get_execution_context()["scope"],
            messages=messages,
            vars=vars,
            intents=intents,
            outcomes=outcomes,
        )

    @staticmethod
    def _validate_continuation(context):
        if not isinstance(context, ContinuationContext):
            raise TypeError("resolve_continuation must return ContinuationContext")
        if context.action not in {"continue", "return"}:
            raise ValueError("Continuation action must be continue or return")
        if context.action == "return" and not context.stop_reason:
            raise ValueError("A terminal continuation requires a stop_reason")
        return context

    def _resolve_continuation(self, phase, messages, vars, intents=(), outcomes=()):
        context = self._run_lifecycle_hooks(
            "resolve_continuation",
            self._continuation_context(phase, messages, vars, intents, outcomes),
            stop_when=lambda current: current.action == "return",
        )
        return self._validate_continuation(context)

    async def _aresolve_continuation(
        self, phase, messages, vars, intents=(), outcomes=()
    ):
        context = await self._arun_lifecycle_hooks(
            "resolve_continuation",
            self._continuation_context(phase, messages, vars, intents, outcomes),
            stop_when=lambda current: current.action == "return",
        )
        return self._validate_continuation(context)
