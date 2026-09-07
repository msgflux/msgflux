"""Typed payloads for stable lifecycle hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Mapping

from msgflux.runtime.context import ExecutionScope

if TYPE_CHECKING:
    from msgflux.runtime.agent_inbox import AgentNotification
    from msgflux.tools.catalog import ToolCatalogView
    from msgflux.tools.runtime import ToolIntent, ToolOutcome

__all__ = [
    "AgentContext",
    "ConversationContext",
    "NotificationContext",
    "BeforeRun",
    "BeforeTool",
    "BeforeToolDispatch",
    "AfterTool",
    "BeforeResume",
    "BeforeCompaction",
    "OutputContext",
    "ModelContext",
    "ModelRequestContext",
    "ModelResponseContext",
    "RunEndContext",
    "ToolCatalogContext",
    "ToolFeedbackContext",
    "ContinuationContext",
]


@dataclass(frozen=True)
class BeforeRun:
    """Input accepted by a fresh Agent run."""

    message: Any
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeforeResume:
    """Durable state that may be transformed before execution resumes.

    Hooks may replace messages, model preference, or non-identity scope fields.
    The restored thread, namespace, and run identity cannot be changed.
    """

    scope: ExecutionScope
    messages: Any
    model_preference: str | None = None


@dataclass(frozen=True, kw_only=True)
class AgentContext:
    """Runtime context shared by Agent lifecycle payloads."""

    scope: ExecutionScope
    vars: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class BeforeCompaction(AgentContext):
    """Automatic compaction decision before a Model creates a context view."""

    messages: Any
    estimated_input_tokens: int
    estimate_source: Literal["provider", "heuristic"]
    context_capacity: int | None
    compacted_through_item_id: str
    reason: Literal["threshold", "manual", "overflow"] = "threshold"
    action: Literal["compact", "skip"] = "skip"
    trigger_tokens: int | None = None


@dataclass(frozen=True, kw_only=True)
class ConversationContext(AgentContext):
    """Conversation prepared for one model request."""

    messages: Any


@dataclass(frozen=True, kw_only=True)
class NotificationContext(AgentContext):
    """Pending non-control notifications before they enter model context."""

    notifications: tuple[AgentNotification, ...] = ()
    messages: Any = None


@dataclass(frozen=True, kw_only=True)
class ToolCatalogContext(AgentContext):
    """Logical tool catalog prepared for one model request."""

    catalog: ToolCatalogView
    messages: Any = None


@dataclass(frozen=True, kw_only=True)
class ModelContext(AgentContext):
    """Model-facing prompt and read-only catalog exposed to prompt extensions.

    ``transform_system_prompt`` consumes only ``system_prompt``. The catalog provides
    request context to prompt extensions and is not a catalog transformation
    boundary.
    """

    system_prompt: str
    tool_catalog: ToolCatalogView | None = None

    @property
    def tool_names(self) -> frozenset[str]:
        if self.tool_catalog is None:
            return frozenset()
        return frozenset(entry.name for entry in self.tool_catalog.visible_entries())


@dataclass(frozen=True, kw_only=True)
class ModelRequestContext(AgentContext):
    """Provider-neutral parameters immediately before an LM call."""

    messages: Any
    system_prompt: str | None = None
    tool_catalog: ToolCatalogView | None = None
    prefilling: str | None = None
    stream: bool = False
    generation_schema: Any = None
    model_preference: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_parameters(
        cls,
        parameters: Mapping[str, Any],
        *,
        scope: ExecutionScope,
        runtime_vars: Mapping[str, Any],
    ) -> ModelRequestContext:
        """Build the stable hook payload from Agent model parameters."""
        known = {
            "messages",
            "system_prompt",
            "tool_catalog",
            "prefilling",
            "stream",
            "generation_schema",
            "model_preference",
        }
        return cls(
            scope=scope,
            vars=runtime_vars,
            messages=parameters.get("messages"),
            system_prompt=parameters.get("system_prompt"),
            tool_catalog=parameters.get("tool_catalog"),
            prefilling=parameters.get("prefilling"),
            stream=bool(parameters.get("stream", False)),
            generation_schema=parameters.get("generation_schema"),
            model_preference=parameters.get("model_preference"),
            extra={key: value for key, value in parameters.items() if key not in known},
        )

    def to_parameters(self) -> dict[str, Any]:
        """Return the model call parameters represented by this context."""
        parameters = {
            **self.extra,
            "messages": self.messages,
            "system_prompt": self.system_prompt,
            "prefilling": self.prefilling,
            "stream": self.stream,
            "tool_catalog": self.tool_catalog,
            "generation_schema": self.generation_schema,
        }
        if self.model_preference is not None:
            parameters["model_preference"] = self.model_preference
        return parameters


@dataclass(frozen=True, kw_only=True)
class ModelResponseContext(AgentContext):
    """Settled non-streaming LM response before canonical history is updated."""

    response: Any
    request: ModelRequestContext


@dataclass(frozen=True, kw_only=True)
class OutputContext(AgentContext):
    """Settled Agent output before presentation to the caller."""

    output: Any


@dataclass(frozen=True, kw_only=True)
class ToolFeedbackContext(AgentContext):
    """Tool outcomes awaiting an Agent-level continuation decision."""

    intents: tuple[ToolIntent, ...]
    outcomes: tuple[ToolOutcome, ...]
    messages: Any
    reasoning: str | None = None
    action: Literal["continue", "return"] = "continue"
    output: Any = None


@dataclass(frozen=True, kw_only=True)
class ContinuationContext(AgentContext):
    """A safe boundary where execution policies may stop the Agent loop."""

    phase: Literal["before_request", "after_tools"]
    messages: Any
    intents: tuple[ToolIntent, ...] = ()
    outcomes: tuple[ToolOutcome, ...] = ()
    action: Literal["continue", "return"] = "continue"
    output: Any = None
    stop_reason: str | None = None


@dataclass(frozen=True, kw_only=True)
class RunEndContext(AgentContext):
    """Terminal Agent outcome around the durable run boundary."""

    outcome: Literal["completed", "failed", "interrupted", "paused"]
    messages: Any
    output: Any = None
    error: BaseException | None = None


@dataclass(frozen=True)
class BeforeTool:
    """A resolved and validated tool call awaiting execution."""

    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    block: str | None = None


@dataclass(frozen=True)
class BeforeToolDispatch:
    """A validated tool call immediately before dispatch."""

    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    dispatch_mode: Literal["foreground", "background", "detached"] = "foreground"
    block: str | None = None


@dataclass(frozen=True)
class AfterTool:
    """A tool outcome before it becomes an Agent tool result."""

    tool_call_id: str
    tool_name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    result: Any = None
    error: BaseException | str | None = None
