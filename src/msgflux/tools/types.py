from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass
class ToolMetadata:
    """Normalized metadata extracted from a Python callable tool."""

    name: str
    description: str
    annotations: Dict[str, Any]
    tool_config: Dict[str, Any]
    impl: Callable
    display_name: str | None = None
    usage_guidance: str | None = None
    source_tool: Any | None = None


class ToolBucket:
    """Base class for tools that absorb other tools by kind."""

    tool_kind = "bucket"
    capture_kind: str

    def add(self, tool: ToolMetadata) -> None:
        raise NotImplementedError


class ToolLibraryOperator:
    """Base class for runtime tools that operate through ToolLibraryHandle."""

    tool_kind = "runtime"
    inject_handle = True
