"""Provider-neutral contracts shared by Models, Agents, and ToolLibrary."""

from copy import deepcopy
from typing import Any, Mapping

import msgspec

_OUTCOME_STATUSES = {
    "blocked",
    "completed",
    "dispatched",
    "execution_failed",
    "interrupted",
    "invalid_arguments",
    "not_found",
}


def _require_name(value: Any, subject: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{subject}` must be a non-empty string")
    return value


def _copy_mapping(value: Mapping[str, Any], subject: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"`{subject}` must be a mapping")
    return deepcopy(dict(value))


class FeedbackSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Agent-facing handling requested after dispatch settles."""

    name: str = "model"
    options: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(
            self, "name", _require_name(self.name, "feedback.name")
        )
        msgspec.structs.force_setattr(
            self,
            "options",
            _copy_mapping(self.options, "feedback.options"),
        )

    @classmethod
    def coerce(cls, value: "FeedbackSpec | str | None") -> "FeedbackSpec":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(name=value)
        raise TypeError("`feedback` must be a FeedbackSpec, string, or None")


class ToolIntent(msgspec.Struct, frozen=True, kw_only=True):
    """Provider-neutral request to perform one named action."""

    id: str
    name: str
    arguments: Mapping[str, Any] = msgspec.field(default_factory=dict)
    parent_id: str | None = None

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(self, "id", _require_name(self.id, "id"))
        msgspec.structs.force_setattr(self, "name", _require_name(self.name, "name"))
        msgspec.structs.force_setattr(
            self,
            "arguments",
            _copy_mapping(self.arguments, "arguments"),
        )


class ToolError(msgspec.Struct, frozen=True, kw_only=True):
    """Structured failure independent from model-facing error rendering."""

    code: str
    message: str
    details: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(
            self, "code", _require_name(self.code, "error.code")
        )
        msgspec.structs.force_setattr(
            self,
            "message",
            _require_name(self.message, "error.message"),
        )
        msgspec.structs.force_setattr(
            self,
            "details",
            _copy_mapping(self.details, "error.details"),
        )


class ToolOutcome(msgspec.Struct, frozen=True, kw_only=True):
    """Canonical result returned by every tool dispatch extension."""

    intent_id: str
    tool_name: str
    status: str
    result: Any = None
    error: ToolError | None = None
    feedback: FeedbackSpec = msgspec.field(default_factory=FeedbackSpec)
    metadata: Mapping[str, Any] = msgspec.field(default_factory=dict)

    def __post_init__(self) -> None:
        msgspec.structs.force_setattr(
            self, "intent_id", _require_name(self.intent_id, "intent_id")
        )
        msgspec.structs.force_setattr(
            self, "tool_name", _require_name(self.tool_name, "tool_name")
        )
        if self.status not in _OUTCOME_STATUSES:
            expected = ", ".join(sorted(_OUTCOME_STATUSES))
            raise ValueError(
                f"Unsupported tool outcome status `{self.status}`: {expected}"
            )
        if self.status in _OUTCOME_STATUSES - {"completed", "dispatched"}:
            if not isinstance(self.error, ToolError):
                raise ValueError(f"Tool outcome `{self.status}` requires `error`")
        elif self.error is not None:
            raise ValueError(f"Tool outcome `{self.status}` cannot include `error`")
        if not isinstance(self.feedback, FeedbackSpec):
            raise TypeError("`feedback` must be a FeedbackSpec")
        msgspec.structs.force_setattr(
            self,
            "metadata",
            _copy_mapping(self.metadata, "outcome.metadata"),
        )

    @property
    def ok(self) -> bool:
        return self.status in {"completed", "dispatched"}

    @classmethod
    def completed(
        cls,
        intent: ToolIntent,
        result: Any,
        *,
        feedback: FeedbackSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolOutcome":
        return cls(
            intent_id=intent.id,
            tool_name=intent.name,
            status="completed",
            result=result,
            feedback=feedback or FeedbackSpec(),
            metadata=metadata or {},
        )

    @classmethod
    def dispatched(
        cls,
        intent: ToolIntent,
        result: Any = None,
        *,
        feedback: FeedbackSpec | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolOutcome":
        return cls(
            intent_id=intent.id,
            tool_name=intent.name,
            status="dispatched",
            result=result,
            feedback=feedback or FeedbackSpec(),
            metadata=metadata or {},
        )

    @classmethod
    def failed(
        cls,
        intent: ToolIntent,
        *,
        status: str,
        code: str,
        message: str,
        feedback: FeedbackSpec | None = None,
        details: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ToolOutcome":
        if status not in _OUTCOME_STATUSES - {"completed", "dispatched"}:
            raise ValueError(f"`{status}` is not a failure outcome status")
        return cls(
            intent_id=intent.id,
            tool_name=intent.name,
            status=status,
            error=ToolError(
                code=code,
                message=message,
                details=details or {},
            ),
            feedback=feedback or FeedbackSpec(),
            metadata=metadata or {},
        )


__all__ = ["FeedbackSpec", "ToolError", "ToolIntent", "ToolOutcome"]
