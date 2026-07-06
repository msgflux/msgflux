from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import msgspec


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
