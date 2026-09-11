"""Canonical planning and policy payloads for tool execution."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Mapping

import msgspec

from msgflux.nn.modules.tool.definitions import DispatchSpec, ToolDefinition
from msgflux.tools.runtime import FeedbackSpec, ToolIntent, ToolOutcome, _copy_mapping


class ToolExecutionPlan(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved intent plus private runtime arguments and selected policies."""

    intent: ToolIntent
    definition: ToolDefinition
    visible_arguments: Mapping[str, Any] = msgspec.field(default_factory=dict)
    runtime_arguments: Mapping[str, Any] = msgspec.field(default_factory=dict)
    dispatch: DispatchSpec | str | None = None
    feedback: FeedbackSpec | str | None = None

    def __post_init__(self) -> None:
        if self.intent.name != self.definition.name:
            raise ValueError("Tool intent and definition names must match")
        msgspec.structs.force_setattr(
            self,
            "visible_arguments",
            _copy_mapping(self.visible_arguments, "visible_arguments"),
        )
        if not isinstance(self.runtime_arguments, Mapping):
            raise TypeError("`runtime_arguments` must be a mapping")
        msgspec.structs.force_setattr(
            self,
            "runtime_arguments",
            dict(self.runtime_arguments),
        )
        collisions = self.visible_arguments.keys() & self.runtime_arguments.keys()
        if collisions:
            formatted = ", ".join(f"`{name}`" for name in sorted(collisions))
            raise ValueError(
                "Tool arguments cannot be both visible and runtime-provided: "
                f"{formatted}"
            )
        msgspec.structs.force_setattr(
            self,
            "dispatch",
            DispatchSpec.coerce(self.dispatch)
            if self.dispatch is not None
            else self.definition.dispatch,
        )
        msgspec.structs.force_setattr(
            self,
            "feedback",
            FeedbackSpec.coerce(self.feedback)
            if self.feedback is not None
            else self.definition.feedback,
        )

    @property
    def call_arguments(self) -> dict[str, Any]:
        return {**self.visible_arguments, **self.runtime_arguments}


class ToolRuntimeContext(msgspec.Struct, frozen=True, kw_only=True):
    """Execution-local values available to opt-in context bindings."""

    values: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping):
            raise TypeError("`values` must be a mapping")
        msgspec.structs.force_setattr(self, "values", dict(self.values))

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def require(self, name: str) -> Any:
        try:
            return self.values[name]
        except KeyError as exc:
            raise RuntimeError(
                f"Runtime context value `{name}` is unavailable"
            ) from exc


ExecuteTool = Callable[[ToolExecutionPlan | None], Awaitable[ToolOutcome]]


class DispatchRequest(msgspec.Struct, frozen=True, kw_only=True):
    """Input delivered to a dispatch extension."""

    plan: ToolExecutionPlan
    context: ToolRuntimeContext
    execute: ExecuteTool


class BeforeToolPolicy(msgspec.Struct, frozen=True, kw_only=True):
    """Typed payload evaluated before runtime arguments are resolved."""

    intent: ToolIntent
    definition: ToolDefinition
    context: ToolRuntimeContext


class BeforeDispatchPolicy(msgspec.Struct, frozen=True, kw_only=True):
    """Typed payload evaluated after an execution plan is prepared."""

    plan: ToolExecutionPlan
    context: ToolRuntimeContext


class AfterToolPolicy(msgspec.Struct, frozen=True, kw_only=True):
    """Typed payload evaluated after a dispatcher produces an outcome."""

    plan: ToolExecutionPlan
    outcome: ToolOutcome
    context: ToolRuntimeContext


__all__ = [
    "AfterToolPolicy",
    "BeforeDispatchPolicy",
    "BeforeToolPolicy",
    "DispatchRequest",
    "ToolExecutionPlan",
    "ToolRuntimeContext",
]
