from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import msgspec

from msgflux.tools.runtime import ToolIntent, ToolOutcome


@dataclass
class ToolCall:
    """Represents the execution of a single tool call."""

    id: str
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class ToolResponses:
    """Represents the execution of tool calls."""

    return_directly: bool
    tool_calls: List[ToolCall] = field(default_factory=list)

    @classmethod
    def from_outcomes(
        cls, intents: tuple[ToolIntent, ...], outcomes: tuple[ToolOutcome, ...]
    ) -> "ToolResponses":
        """Project canonical execution results for a ToolFlowControl consumer."""
        if len(intents) != len(outcomes):
            raise ValueError("Each legacy tool call must have exactly one outcome")
        direct_modes = {"direct", "handoff", "call_as_response"}
        calls = []
        for intent, outcome in zip(intents, outcomes):
            if outcome.intent_id != intent.id:
                raise ValueError("Tool outcomes must preserve intent ordering")
            calls.append(
                ToolCall(
                    id=outcome.intent_id,
                    name=outcome.tool_name,
                    parameters=dict(
                        outcome.metadata.get("arguments", intent.arguments)
                    ),
                    result=outcome.result,
                    error=outcome.error.message if outcome.error is not None else None,
                )
            )
        return cls(
            return_directly=bool(outcomes)
            and all(
                outcome.status == "completed" and outcome.feedback.name in direct_modes
                for outcome in outcomes
            ),
            tool_calls=calls,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> bytes:
        """Returns a encoded-JSON."""
        return msgspec.json.encode(self.to_dict())

    def get_by_id(self, tool_id: str) -> Optional[ToolCall]:
        """Retrieve a tool_call by tool id."""
        return next((r for r in self.tool_calls if r.id == tool_id), None)

    def get_by_name(self, tool_name: str) -> Optional[ToolCall]:
        """Retrieve a tool_call by tool name."""
        return next((r for r in self.tool_calls if r.name == tool_name), None)
